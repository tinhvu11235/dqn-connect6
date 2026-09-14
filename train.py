"""Self-play DQN training entrypoint for Connect6."""

from __future__ import annotations

import argparse
import os
import time
from typing import Optional

from tqdm import tqdm

from agent import DQNAgent, load_agent_from_checkpoint
from config import (
    TrainConfig,
    apply_smoke_overrides,
    configs_for_preset,
    epsilon_by_step,
    set_global_seed,
)
from connect6 import Connect6Env
from model import count_parameters
from replay import ReplayBuffer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a Connect6 DQN agent by self-play.")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--preset", choices=["local_cpu", "local_gpu"], default="local_cpu")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--board-size", type=int, default=19)
    parser.add_argument("--win-length", type=int, default=6)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--candidate-mode", choices=["all", "local"], default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--no-progress", action="store_true")
    return parser


def make_env(config: TrainConfig) -> Connect6Env:
    return Connect6Env(
        board_size=config.board_size,
        win_length=config.win_length,
        force_center_opening=config.force_center_opening,
    )


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes:d}m{secs:02d}s"
    return f"{secs:d}s"


def run_training(args: argparse.Namespace) -> str:
    model_config, train_config = configs_for_preset(args.preset, board_size=args.board_size)
    train_config.win_length = args.win_length
    train_config.checkpoint_dir = args.checkpoint_dir
    train_config.seed = args.seed
    if args.candidate_mode is not None:
        train_config.candidate_mode = args.candidate_mode
    episodes = int(args.episodes)

    if args.smoke:
        model_config, train_config = apply_smoke_overrides(model_config, train_config)
        episodes = min(episodes, 2)

    set_global_seed(train_config.seed)

    if args.resume:
        agent, checkpoint = load_agent_from_checkpoint(args.resume, device=args.device, load_optimizer=True)
        model_config = agent.model_config
        train_config = agent.train_config
        train_config.checkpoint_dir = args.checkpoint_dir
        if args.candidate_mode is not None:
            train_config.candidate_mode = args.candidate_mode
        start_episode = int(checkpoint.get("train_state", {}).get("episode", 0))
        total_steps = int(checkpoint.get("train_state", {}).get("total_steps", 0))
    else:
        agent = DQNAgent(model_config, train_config=train_config, device=args.device)
        start_episode = 0
        total_steps = 0

    replay = ReplayBuffer(
        capacity=train_config.replay_size,
        board_size=train_config.board_size,
        seed=train_config.seed,
    )
    env = make_env(train_config)
    os.makedirs(train_config.checkpoint_dir, exist_ok=True)

    print(
        "training Connect6 "
        f"board={train_config.board_size}x{train_config.board_size} "
        f"win={train_config.win_length} device={agent.device} "
        f"params={count_parameters(agent.online):,}"
    )
    print(
        "replay "
        f"format={replay.storage_format} words_per_color={replay.word_count} "
        f"capacity={replay.capacity:,} approx={replay.total_storage_bytes / (1024 ** 2):.1f} MB"
    )

    latest_path = os.path.join(train_config.checkpoint_dir, "latest.pt")
    last_loss: Optional[float] = None
    train_started_at = time.perf_counter()

    episode_iter = range(1, episodes + 1)
    progress = tqdm(
        episode_iter,
        total=episodes,
        desc="training",
        unit="episode",
        dynamic_ncols=True,
        disable=args.no_progress,
    )

    for local_episode in progress:
        episode_started_at = time.perf_counter()
        episode = start_episode + local_episode
        env.reset()
        episode_reward = 0.0
        steps_this_episode = 0

        while not env.is_terminal():
            epsilon = epsilon_by_step(total_steps, train_config)
            state = env.snapshot()
            action = agent.select_action(
                env,
                epsilon=epsilon,
                candidate_mode=train_config.candidate_mode,
                candidate_radius=train_config.candidate_radius,
            )
            _, reward, done, info = env.step(action)
            next_state = env.snapshot()
            replay.push(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
                same_player_next=bool(info["same_player_next"]),
            )

            total_steps += 1
            steps_this_episode += 1
            episode_reward += reward

            if (
                len(replay) >= train_config.replay_warmup
                and total_steps % train_config.train_every == 0
            ):
                last_loss = agent.optimize(replay, train_config.batch_size, augment=True)

        elapsed = time.perf_counter() - train_started_at
        episode_seconds = time.perf_counter() - episode_started_at
        avg_episode_seconds = elapsed / max(1, local_episode)
        eta_seconds = avg_episode_seconds * max(0, episodes - local_episode)

        if episode % max(1, args.save_every) == 0 or local_episode == episodes:
            agent.save_checkpoint(
                latest_path,
                train_state={
                    "episode": episode,
                    "total_steps": total_steps,
                    "last_loss": -1.0 if last_loss is None else last_loss,
                    "elapsed_seconds": elapsed,
                    "avg_episode_seconds": avg_episode_seconds,
                },
            )
            if local_episode != episodes:
                progress.write(f"saved checkpoint: {latest_path}")

        progress.set_postfix(
            {
                "ep": episode,
                "steps": steps_this_episode,
                "winner": env.winner,
                "replay": len(replay),
                "eps": f"{epsilon_by_step(total_steps, train_config):.3f}",
                "loss": "n/a" if last_loss is None else f"{last_loss:.5f}",
                "ep_time": format_duration(episode_seconds),
                "eta": format_duration(eta_seconds),
            }
        )

    print(f"saved checkpoint: {latest_path}")
    return latest_path


def main() -> None:
    args = build_parser().parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
