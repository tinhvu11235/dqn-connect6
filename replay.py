"""Bitboard replay buffer for Connect6 DQN training."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from connect6 import GameSnapshot, encode_board_state
from symmetry import NUM_SYMMETRIES, transform_action, transform_board


class ReplayBuffer:
    def __init__(
        self,
        capacity: int = 50_000,
        board_size: int = 19,
        seed: Optional[int] = None,
    ) -> None:
        self.capacity = int(capacity)
        self.board_size = int(board_size)
        self.num_actions = self.board_size * self.board_size
        self.word_count = (self.num_actions + 63) // 64
        self.rng = np.random.default_rng(seed)
        self._word_indices = (np.arange(self.num_actions) // 64).astype(np.int64)
        self._bit_masks = np.left_shift(
            np.uint64(1),
            (np.arange(self.num_actions, dtype=np.uint64) % np.uint64(64)),
        ).astype(np.uint64)

        self.black_bits = np.zeros((self.capacity, self.word_count), dtype=np.uint64)
        self.white_bits = np.zeros_like(self.black_bits)
        self.next_black_bits = np.zeros_like(self.black_bits)
        self.next_white_bits = np.zeros_like(self.black_bits)
        self.current_players = np.zeros(self.capacity, dtype=np.int8)
        self.next_players = np.zeros(self.capacity, dtype=np.int8)
        self.stones_left = np.zeros(self.capacity, dtype=np.uint8)
        self.next_stones_left = np.zeros(self.capacity, dtype=np.uint8)
        self.last_actions = np.full(self.capacity, -1, dtype=np.int16)
        self.next_last_actions = np.full(self.capacity, -1, dtype=np.int16)
        self.actions = np.zeros(self.capacity, dtype=np.uint16)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.dones = np.zeros(self.capacity, dtype=np.bool_)
        self.same_player_next = np.zeros(self.capacity, dtype=np.bool_)
        self.pos = 0
        self.size = 0

    def __len__(self) -> int:
        return int(self.size)

    @property
    def storage_format(self) -> str:
        return "bitboard_uint64"

    @property
    def board_storage_bytes(self) -> int:
        return (
            self.black_bits.nbytes
            + self.white_bits.nbytes
            + self.next_black_bits.nbytes
            + self.next_white_bits.nbytes
        )

    @property
    def total_storage_bytes(self) -> int:
        arrays = [
            self.black_bits,
            self.white_bits,
            self.next_black_bits,
            self.next_white_bits,
            self.current_players,
            self.next_players,
            self.stones_left,
            self.next_stones_left,
            self.last_actions,
            self.next_last_actions,
            self.actions,
            self.rewards,
            self.dones,
            self.same_player_next,
        ]
        return int(sum(array.nbytes for array in arrays))

    def _pack_board(self, board: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        flat = np.asarray(board, dtype=np.int8).reshape(-1)
        if flat.size != self.num_actions:
            raise ValueError(
                f"board has {flat.size} cells, expected {self.num_actions}"
            )

        black = np.zeros(self.word_count, dtype=np.uint64)
        white = np.zeros(self.word_count, dtype=np.uint64)
        black_cells = flat == 1
        white_cells = flat == -1
        np.bitwise_or.at(black, self._word_indices[black_cells], self._bit_masks[black_cells])
        np.bitwise_or.at(white, self._word_indices[white_cells], self._bit_masks[white_cells])
        return black, white

    def _unpack_board(self, black: np.ndarray, white: np.ndarray) -> np.ndarray:
        flat = np.zeros(self.num_actions, dtype=np.int8)
        self._unpack_color(flat, black, 1)
        self._unpack_color(flat, white, -1)
        return flat.reshape(self.board_size, self.board_size)

    def _unpack_color(self, flat: np.ndarray, words: np.ndarray, value: int) -> None:
        for word_idx, word in enumerate(words):
            bits = int(word)
            base_action = word_idx * 64
            while bits:
                lowest_bit = bits & -bits
                action = base_action + lowest_bit.bit_length() - 1
                if action < self.num_actions:
                    flat[action] = value
                bits ^= lowest_bit

    def push(
        self,
        state: GameSnapshot,
        action: int,
        reward: float,
        next_state: GameSnapshot,
        done: bool,
        same_player_next: bool,
    ) -> None:
        idx = self.pos
        black, white = self._pack_board(state.board)
        next_black, next_white = self._pack_board(next_state.board)
        self.black_bits[idx] = black
        self.white_bits[idx] = white
        self.next_black_bits[idx] = next_black
        self.next_white_bits[idx] = next_white
        self.current_players[idx] = state.current_player
        self.next_players[idx] = next_state.current_player
        self.stones_left[idx] = state.stones_left_in_turn
        self.next_stones_left[idx] = next_state.stones_left_in_turn
        self.last_actions[idx] = state.last_action
        self.next_last_actions[idx] = next_state.last_action
        self.actions[idx] = int(action)
        self.rewards[idx] = float(reward)
        self.dones[idx] = bool(done)
        self.same_player_next[idx] = bool(same_player_next)

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, augment: bool = True) -> Dict[str, np.ndarray]:
        if self.size < batch_size:
            raise ValueError(f"not enough replay samples: have {self.size}, need {batch_size}")

        indices = self.rng.choice(self.size, size=batch_size, replace=False)
        states = np.empty((batch_size, 4, self.board_size, self.board_size), dtype=np.float32)
        next_states = np.empty_like(states)
        actions = np.empty(batch_size, dtype=np.int64)
        next_legal_masks = np.empty((batch_size, self.board_size * self.board_size), dtype=np.bool_)

        for out_idx, replay_idx in enumerate(indices):
            board = self._unpack_board(
                self.black_bits[replay_idx],
                self.white_bits[replay_idx],
            )
            next_board = self._unpack_board(
                self.next_black_bits[replay_idx],
                self.next_white_bits[replay_idx],
            )
            action = int(self.actions[replay_idx])
            last_action = int(self.last_actions[replay_idx])
            next_last_action = int(self.next_last_actions[replay_idx])

            if augment:
                symmetry = int(self.rng.integers(NUM_SYMMETRIES))
                board = transform_board(board, symmetry)
                next_board = transform_board(next_board, symmetry)
                action = transform_action(action, self.board_size, symmetry)
                last_action = transform_action(last_action, self.board_size, symmetry)
                next_last_action = transform_action(next_last_action, self.board_size, symmetry)

            states[out_idx] = encode_board_state(
                board,
                int(self.current_players[replay_idx]),
                int(self.stones_left[replay_idx]),
                last_action,
            )
            next_states[out_idx] = encode_board_state(
                next_board,
                int(self.next_players[replay_idx]),
                int(self.next_stones_left[replay_idx]),
                next_last_action,
            )
            actions[out_idx] = action
            if self.dones[replay_idx]:
                next_legal_masks[out_idx].fill(False)
            else:
                next_legal_masks[out_idx] = next_board.reshape(-1) == 0

        return {
            "states": states,
            "actions": actions,
            "rewards": self.rewards[indices].astype(np.float32, copy=True),
            "next_states": next_states,
            "dones": self.dones[indices].astype(np.bool_, copy=True),
            "same_player_next": self.same_player_next[indices].astype(np.bool_, copy=True),
            "next_legal_masks": next_legal_masks,
        }
