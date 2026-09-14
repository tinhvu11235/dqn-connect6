"""Connect6 environment with atomic one-stone actions.

The environment models a two-stone Connect6 turn as two consecutive calls to
``step`` by the same player. The first black move is the only exception and
places one stone before the turn switches to white.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


EMPTY = 0
BLACK = 1
WHITE = -1


@dataclass(frozen=True)
class GameSnapshot:
    board: np.ndarray
    current_player: int
    stones_left_in_turn: int
    last_action: int


def encode_board_state(
    board: np.ndarray,
    current_player: int,
    stones_left_in_turn: int,
    last_action: int = -1,
) -> np.ndarray:
    """Encode a board from the current player's perspective.

    Plane 0: stones owned by ``current_player``.
    Plane 1: opponent stones.
    Plane 2: normalized stones left in the current atomic turn phase.
    Plane 3: one-hot last move.
    """

    board = np.asarray(board)
    board_size = board.shape[0]
    state = np.zeros((4, board_size, board_size), dtype=np.float32)
    state[0] = board == current_player
    state[1] = board == -current_player
    state[2].fill(float(stones_left_in_turn) / 2.0)
    if last_action >= 0:
        row, col = divmod(int(last_action), board_size)
        if 0 <= row < board_size and 0 <= col < board_size:
            state[3, row, col] = 1.0
    return state


class Connect6Env:
    """Rules-only Connect6 environment.

    Actions are atomic: ``action = row * board_size + col``.
    """

    def __init__(
        self,
        board_size: int = 19,
        win_length: int = 6,
        force_center_opening: bool = False,
    ) -> None:
        if board_size < win_length:
            raise ValueError("board_size must be at least win_length")
        self.board_size = int(board_size)
        self.win_length = int(win_length)
        self.force_center_opening = bool(force_center_opening)
        self.reset()

    @property
    def num_actions(self) -> int:
        return self.board_size * self.board_size

    @property
    def nb_actions(self) -> int:
        return self.num_actions

    def reset(self) -> np.ndarray:
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        self.current_player = BLACK
        self.stones_left_in_turn = 1
        self.winner = 0
        self.done = False
        self.move_count = 0
        self.last_action = -1
        return self.encode_state()

    def snapshot(self) -> GameSnapshot:
        return GameSnapshot(
            board=self.board.copy(),
            current_player=int(self.current_player),
            stones_left_in_turn=int(self.stones_left_in_turn),
            last_action=int(self.last_action),
        )

    def encode_state(self) -> np.ndarray:
        return encode_board_state(
            self.board,
            self.current_player,
            self.stones_left_in_turn,
            self.last_action,
        )

    def encode_action(self, row: int, col: int) -> int:
        row = int(row)
        col = int(col)
        if not (0 <= row < self.board_size and 0 <= col < self.board_size):
            raise ValueError(f"action coordinates out of bounds: {(row, col)}")
        return row * self.board_size + col

    def decode_action(self, action: int) -> Tuple[int, int]:
        action = int(action)
        if not (0 <= action < self.num_actions):
            raise ValueError(f"action out of bounds: {action}")
        return divmod(action, self.board_size)

    def get_legal_actions(self) -> List[int]:
        if self.done:
            return []
        if self.force_center_opening and self.move_count == 0:
            center = self.board_size // 2
            return [self.encode_action(center, center)]
        return np.flatnonzero(self.board.reshape(-1) == EMPTY).astype(np.int64).tolist()

    def get_legal_mask(self) -> np.ndarray:
        mask = np.zeros(self.num_actions, dtype=np.bool_)
        legal_actions = self.get_legal_actions()
        if legal_actions:
            mask[np.asarray(legal_actions, dtype=np.int64)] = True
        return mask

    def is_terminal(self) -> bool:
        return bool(self.done)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, object]]:
        if self.done:
            raise RuntimeError("cannot step a terminal game")

        row, col = self.decode_action(action)
        legal_mask = self.get_legal_mask()
        if not legal_mask[int(action)]:
            raise ValueError(f"illegal action {action} at {(row, col)}")

        acting_player = int(self.current_player)
        stones_before = int(self.stones_left_in_turn)
        self.board[row, col] = acting_player
        self.last_action = int(action)
        self.move_count += 1

        reward = 0.0
        if self.check_win(acting_player, row, col):
            self.done = True
            self.winner = acting_player
            reward = 1.0
        elif self.move_count == self.num_actions:
            self.done = True
            self.winner = 0
        else:
            if self.stones_left_in_turn > 1:
                self.stones_left_in_turn -= 1
            else:
                self.current_player = -self.current_player
                self.stones_left_in_turn = 2

        same_player_next = (not self.done) and self.current_player == acting_player
        info: Dict[str, object] = {
            "acting_player": acting_player,
            "next_player": int(self.current_player),
            "same_player_next": bool(same_player_next),
            "stones_before": stones_before,
            "stones_left_in_turn": int(self.stones_left_in_turn),
            "move_count": int(self.move_count),
            "winner": int(self.winner),
        }
        return self.encode_state(), reward, bool(self.done), info

    def check_win(
        self,
        player: Optional[int] = None,
        row: Optional[int] = None,
        col: Optional[int] = None,
    ) -> bool:
        if player is None:
            player = self.current_player
        player = int(player)

        if row is None or col is None:
            positions = np.argwhere(self.board == player)
            return any(self.check_win(player, int(r), int(c)) for r, c in positions)

        if self.board[int(row), int(col)] != player:
            return False

        for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
            count = 1
            count += self._count_direction(player, int(row), int(col), dr, dc)
            count += self._count_direction(player, int(row), int(col), -dr, -dc)
            if count >= self.win_length:
                return True
        return False

    def _count_direction(self, player: int, row: int, col: int, dr: int, dc: int) -> int:
        count = 0
        row += dr
        col += dc
        while 0 <= row < self.board_size and 0 <= col < self.board_size:
            if self.board[row, col] != player:
                break
            count += 1
            row += dr
            col += dc
        return count

    def render(self) -> str:
        symbols = {EMPTY: ".", BLACK: "X", WHITE: "O"}
        rows = [" ".join(symbols[int(v)] for v in row) for row in self.board]
        return "\n".join(rows)

    def copy(self) -> "Connect6Env":
        clone = Connect6Env(
            board_size=self.board_size,
            win_length=self.win_length,
            force_center_opening=self.force_center_opening,
        )
        clone.board = self.board.copy()
        clone.current_player = int(self.current_player)
        clone.stones_left_in_turn = int(self.stones_left_in_turn)
        clone.winner = int(self.winner)
        clone.done = bool(self.done)
        clone.move_count = int(self.move_count)
        clone.last_action = int(self.last_action)
        return clone


# Backward-friendly alias for older imports that expected a Connect6 class.
Connect6 = Connect6Env
