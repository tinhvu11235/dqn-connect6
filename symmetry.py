"""D4 board symmetries for square Connect6 boards."""

from __future__ import annotations

from typing import Tuple

import numpy as np


NUM_SYMMETRIES = 8


def transform_coords(row: int, col: int, board_size: int, symmetry: int) -> Tuple[int, int]:
    row = int(row)
    col = int(col)
    n = int(board_size)
    op = int(symmetry) % NUM_SYMMETRIES
    if op == 0:
        return row, col
    if op == 1:
        return col, n - 1 - row
    if op == 2:
        return n - 1 - row, n - 1 - col
    if op == 3:
        return n - 1 - col, row
    if op == 4:
        return row, n - 1 - col
    if op == 5:
        return n - 1 - row, col
    if op == 6:
        return col, row
    return n - 1 - col, n - 1 - row


def transform_action(action: int, board_size: int, symmetry: int) -> int:
    action = int(action)
    if action < 0:
        return action
    row, col = divmod(action, int(board_size))
    new_row, new_col = transform_coords(row, col, board_size, symmetry)
    return new_row * int(board_size) + new_col


def transform_board(board: np.ndarray, symmetry: int) -> np.ndarray:
    """Transform a 2D board or a CHW tensor using the selected D4 symmetry."""

    op = int(symmetry) % NUM_SYMMETRIES
    axes = (-2, -1)
    if op == 0:
        out = board
    elif op == 1:
        out = np.rot90(board, k=-1, axes=axes)
    elif op == 2:
        out = np.rot90(board, k=2, axes=axes)
    elif op == 3:
        out = np.rot90(board, k=1, axes=axes)
    elif op == 4:
        out = np.flip(board, axis=-1)
    elif op == 5:
        out = np.flip(board, axis=-2)
    elif op == 6:
        out = np.swapaxes(board, -2, -1)
    else:
        out = np.flip(np.swapaxes(board, -2, -1), axis=(-2, -1))
    return np.ascontiguousarray(out)
