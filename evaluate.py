"""Evaluation helpers for Connect6 checkpoints."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Callable, Dict, Optional

import numpy as np

from agent import DQNAgent, candidate_actions, load_agent_from_checkpoint
from config import TrainConfig, configs_for_preset, set_global_seed
from connect6 import BLACK, Connect6Env, WHITE


Policy = Callable[[Connect6Env], int]


@dataclass
class MatchStats:
    wins: int = 0
    losses: int = 0
    draws: int = 0
    moves: int = 0

    @property
    def games(self) -> int:
        return self.wins + self.losses + self.draws

    @property
    def win_rate(self) -> float:
        return 0.0 if self.games == 0 else self.wins / self.games

    @property
    def avg_moves(self) -> float:
        return 0.0 if self.games == 0 else self.moves / self.games


def random_policy(rng: np.random.Generator) -> Policy:
    def choose(env: Connect6Env) -> int:
        return int(rng.choice(env.get_legal_actions()))

    return choose


def heuristic_policy(rng: np.random.Generator, candidate_mode: str = "local") -> Policy:
    def choose(env: Connect6Env) -> int:
        legal = env.get_legal_actions()
        player = int(env.current_player)
        for action in legal:
            row, col = env.decode_action(action)
            env.board[row, col] = player
            wins = env.check_win(player, row, col)
            env.board[row, col] = 0
            if wins:
                return int(action)
        for action in legal:
            row, col = env.decode_action(action)
            env.board[row, col] = -player
            blocks = env.check_win(-player, row, col)
            env.board[row, col] = 0
            if blocks:
                return int(action)

        candidates = candidate_actions(env, mode=candidate_mode, radius=2)
        best_score = -1
        best_actions = []
        for action in candidates:
            row, col = env.decode_action(action)
            score = 0
            for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                for sign in (1, -1):
                    r = row + sign * dr
                    c = col + sign * dc
                    run = 0
                    while 0 <= r < env.board_size and 0 <= c < env.board_size:
                        if env.board[r, c] != player:
                            break
                        run += 1
                        r += sign * dr
                        c += sign * dc
                    score += run * run
            if score > best_score:
                best_score = score
                best_actions = [action]
            elif score == best_score:
                best_actions.append(action)
        return int(rng.choice(best_actions if best_actions else legal))

    return choose


def network_policy(agent: DQNAgent, candidate_mode: str = "all") -> Policy:
    def choose(env: Connect6Env) -> int:
        return agent.select_action(env, epsilon=0.0, candidate_mode=candidate_mode)

    return choose


def play_game(
    train_config: TrainConfig,
    network_player: int,
    network: Policy,
    opponent: Policy,
) -> Dict[str, int]:
    env = Connect6Env(board_size=train_config.board_size, win_length=train_config.win_length)
    env.reset()
    while not env.is_terminal():
        action = network(env) if env.current_player == network_player else opponent(env)
        env.step(action)
    if env.winner == network_player:
        result = "win"
    elif env.winner == 0:
        result = "draw"
    else:
        result = "loss"
    return {"result": result, "moves": env.move_count, "winner": env.winner}


def evaluate_match(
    train_config: TrainConfig,
    games: int,
    network: Policy,
    opponent: Policy,
) -> MatchStats:
    stats = MatchStats()
    for game_idx in range(games):
        network_player = BLACK if game_idx % 2 == 0 else WHITE
        result = play_game(train_config, network_player, network, opponent)
        stats.moves += int(result["moves"])
        if result["result"] == "win":
            stats.wins += 1
        elif result["result"] == "loss":
            stats.losses += 1
        else:
            stats.draws += 1
    return stats


def greedy_self_play(train_config: TrainConfig, games: int, network: Policy) -> Dict[str, float]:
    black_wins = 0
    white_wins = 0
    draws = 0
    total_moves = 0
    for _ in range(games):
        env = Connect6Env(board_size=train_config.board_size, win_length=train_config.win_length)
        env.reset()
        while not env.is_terminal():
            env.step(network(env))
        black_wins += int(env.winner == BLACK)
        white_wins += int(env.winner == WHITE)
        draws += int(env.winner == 0)
        total_moves += env.move_count
    return {
        "black_wins": black_wins,
        "white_wins": white_wins,
        "draws": draws,
        "avg_moves": 0.0 if games == 0 else total_moves / games,
    }


def print_stats(name: str, stats: MatchStats) -> None:
    print(
        f"{name}: wins={stats.wins} losses={stats.losses} draws={stats.draws} "
        f"win_rate={stats.win_rate:.3f} avg_moves={stats.avg_moves:.1f}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a Connect6 DQN checkpoint.")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--preset", choices=["local_cpu", "local_gpu"], default="local_cpu")
    parser.add_argument("--board-size", type=int, default=19)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--smoke", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    set_global_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    if args.checkpoint:
        agent, _ = load_agent_from_checkpoint(args.checkpoint, device=args.device, load_optimizer=False)
        train_config = agent.train_config
    else:
        model_config, train_config = configs_for_preset(args.preset, board_size=args.board_size)
        agent = DQNAgent(model_config, train_config=train_config, device=args.device)

    games = min(args.games, 2) if args.smoke else args.games
    net = network_policy(agent, candidate_mode="all")
    print_stats("network_vs_random", evaluate_match(train_config, games, net, random_policy(rng)))
    print_stats("network_vs_heuristic", evaluate_match(train_config, games, net, heuristic_policy(rng)))
    self_play = greedy_self_play(train_config, games, net)
    print(
        "greedy_self_play: "
        f"black_wins={self_play['black_wins']} white_wins={self_play['white_wins']} "
        f"draws={self_play['draws']} avg_moves={self_play['avg_moves']:.1f}"
    )


if __name__ == "__main__":
    main()
