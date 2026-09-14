"""Tkinter GUI for playing Connect6 against a trained checkpoint."""

from __future__ import annotations

import argparse
import tkinter as tk
from tkinter import messagebox
from typing import Callable, Optional

import numpy as np

from agent import DQNAgent, load_agent_from_checkpoint
from config import ModelConfig, TrainConfig, set_global_seed
from connect6 import BLACK, WHITE, Connect6Env
from evaluate import heuristic_policy


Policy = Callable[[Connect6Env], int]


class Connect6Gui:
    def __init__(
        self,
        root: tk.Tk,
        env: Connect6Env,
        ai_policy: Policy,
        human_player: int,
        cell_size: int = 32,
    ) -> None:
        self.root = root
        self.env = env
        self.ai_policy = ai_policy
        self.human_player = int(human_player)
        self.ai_player = -self.human_player
        self.cell_size = int(cell_size)
        self.margin = max(24, self.cell_size)
        self.board_pixels = self.margin * 2 + self.cell_size * (self.env.board_size - 1)
        self.busy = False

        self.root.title("Connect6 - Human vs AI")
        self.status = tk.StringVar()
        self.canvas = tk.Canvas(
            root,
            width=self.board_pixels,
            height=self.board_pixels,
            bg="#d8ad66",
            highlightthickness=0,
        )
        self.canvas.pack(padx=12, pady=(12, 6))
        self.canvas.bind("<Button-1>", self.on_click)

        controls = tk.Frame(root)
        controls.pack(fill=tk.X, padx=12, pady=(0, 12))
        tk.Button(controls, text="New game", command=self.new_game).pack(side=tk.LEFT)
        tk.Label(controls, textvariable=self.status, anchor="w").pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(12, 0)
        )

        self.draw()
        self.update_status()
        if self.env.current_player == self.ai_player:
            self.root.after(350, self.perform_ai_turn)

    def new_game(self) -> None:
        self.env.reset()
        self.busy = False
        self.draw()
        self.update_status()
        if self.env.current_player == self.ai_player:
            self.root.after(350, self.perform_ai_turn)

    def board_to_pixel(self, row: int, col: int) -> tuple[int, int]:
        return self.margin + col * self.cell_size, self.margin + row * self.cell_size

    def pixel_to_board(self, x: int, y: int) -> Optional[tuple[int, int]]:
        col = int(round((x - self.margin) / self.cell_size))
        row = int(round((y - self.margin) / self.cell_size))
        if not (0 <= row < self.env.board_size and 0 <= col < self.env.board_size):
            return None
        px, py = self.board_to_pixel(row, col)
        if abs(px - x) > self.cell_size * 0.45 or abs(py - y) > self.cell_size * 0.45:
            return None
        return row, col

    def on_click(self, event: tk.Event) -> None:
        if self.busy or self.env.is_terminal() or self.env.current_player != self.human_player:
            return
        coords = self.pixel_to_board(int(event.x), int(event.y))
        if coords is None:
            return
        row, col = coords
        action = self.env.encode_action(row, col)
        if action not in self.env.get_legal_actions():
            self.status.set("That intersection is already occupied.")
            return
        self.apply_action(action)
        if not self.env.is_terminal() and self.env.current_player == self.ai_player:
            self.busy = True
            self.root.after(350, self.perform_ai_turn)

    def perform_ai_turn(self) -> None:
        if self.env.is_terminal() or self.env.current_player != self.ai_player:
            self.busy = False
            self.update_status()
            return
        try:
            action = int(self.ai_policy(self.env))
        except Exception as exc:
            self.busy = False
            messagebox.showerror("AI error", str(exc))
            return
        self.apply_action(action)
        if not self.env.is_terminal() and self.env.current_player == self.ai_player:
            self.root.after(250, self.perform_ai_turn)
        else:
            self.busy = False
            self.update_status()

    def apply_action(self, action: int) -> None:
        try:
            self.env.step(action)
        except Exception as exc:
            messagebox.showerror("Illegal move", str(exc))
            return
        self.draw()
        self.update_status()

    def update_status(self) -> None:
        if self.env.is_terminal():
            if self.env.winner == 0:
                self.status.set("Draw.")
            elif self.env.winner == self.human_player:
                self.status.set("You win.")
            else:
                self.status.set("AI wins.")
            return

        turn = "Your turn" if self.env.current_player == self.human_player else "AI turn"
        color = "Black" if self.env.current_player == BLACK else "White"
        self.status.set(
            f"{turn} - {color}, stones left this turn: {self.env.stones_left_in_turn}"
        )

    def draw(self) -> None:
        self.canvas.delete("all")
        n = self.env.board_size
        start = self.margin
        end = self.margin + self.cell_size * (n - 1)
        for idx in range(n):
            p = self.margin + idx * self.cell_size
            self.canvas.create_line(start, p, end, p, fill="#4b3218")
            self.canvas.create_line(p, start, p, end, fill="#4b3218")

        star_points = self.star_points(n)
        for row, col in star_points:
            x, y = self.board_to_pixel(row, col)
            self.canvas.create_oval(x - 3, y - 3, x + 3, y + 3, fill="#4b3218", outline="")

        radius = max(8, int(self.cell_size * 0.42))
        for row in range(n):
            for col in range(n):
                value = int(self.env.board[row, col])
                if value == 0:
                    continue
                x, y = self.board_to_pixel(row, col)
                fill = "#111111" if value == BLACK else "#f5f0e6"
                outline = "#111111" if value == BLACK else "#4b3218"
                self.canvas.create_oval(
                    x - radius,
                    y - radius,
                    x + radius,
                    y + radius,
                    fill=fill,
                    outline=outline,
                    width=2,
                )

        if self.env.last_action >= 0:
            row, col = self.env.decode_action(self.env.last_action)
            x, y = self.board_to_pixel(row, col)
            self.canvas.create_rectangle(
                x - radius - 2,
                y - radius - 2,
                x + radius + 2,
                y + radius + 2,
                outline="#c51f1a",
                width=2,
            )

    @staticmethod
    def star_points(board_size: int) -> list[tuple[int, int]]:
        if board_size < 9:
            return []
        if board_size == 19:
            points = [3, 9, 15]
        else:
            edge = min(3, board_size // 4)
            center = board_size // 2
            points = sorted({edge, center, board_size - 1 - edge})
        return [(r, c) for r in points for c in points]


def make_policy(args: argparse.Namespace) -> tuple[Policy, int, int]:
    if args.checkpoint:
        agent, _ = load_agent_from_checkpoint(args.checkpoint, device=args.device, load_optimizer=False)

        def policy(env: Connect6Env) -> int:
            return agent.select_action(
                env,
                epsilon=0.0,
                candidate_mode=args.candidate_mode,
                candidate_radius=args.candidate_radius,
            )

        return policy, agent.train_config.board_size, agent.train_config.win_length

    rng = np.random.default_rng(args.seed)
    policy = heuristic_policy(rng, candidate_mode=args.candidate_mode)
    return policy, args.board_size, args.win_length


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Play Connect6 against an AI checkpoint.")
    parser.add_argument("--checkpoint", default=None, help="Path to a trained .pt checkpoint.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--human", default="black", choices=["black", "white"])
    parser.add_argument("--candidate-mode", default="all", choices=["all", "local"])
    parser.add_argument("--candidate-radius", type=int, default=2)
    parser.add_argument("--board-size", type=int, default=19)
    parser.add_argument("--win-length", type=int, default=6)
    parser.add_argument("--cell-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=123)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    set_global_seed(args.seed)
    ai_policy, board_size, win_length = make_policy(args)
    human_player = BLACK if args.human == "black" else WHITE
    env = Connect6Env(board_size=board_size, win_length=win_length)
    root = tk.Tk()
    Connect6Gui(root, env, ai_policy, human_player, cell_size=args.cell_size)
    root.mainloop()


if __name__ == "__main__":
    main()
