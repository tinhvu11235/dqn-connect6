"""Configuration helpers for local Connect6 training."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Any, Dict, Tuple

import numpy as np
import torch


@dataclass
class ModelConfig:
    board_size: int = 19
    input_channels: int = 4
    channels: int = 32
    num_blocks: int = 3

    @property
    def num_actions(self) -> int:
        return self.board_size * self.board_size


@dataclass
class TrainConfig:
    board_size: int = 19
    win_length: int = 6
    batch_size: int = 64
    replay_size: int = 50_000
    replay_warmup: int = 2_000
    gamma: float = 0.99
    learning_rate: float = 1e-4
    epsilon_start: float = 1.0
    epsilon_end: float = 0.10
    epsilon_decay_steps: int = 100_000
    train_every: int = 4
    target_update_interval: int = 1_000
    grad_clip: float = 5.0
    candidate_mode: str = "local"
    candidate_radius: int = 2
    force_center_opening: bool = False
    checkpoint_dir: str = "checkpoints"
    seed: int = 123


def configs_for_preset(preset: str, board_size: int = 19) -> Tuple[ModelConfig, TrainConfig]:
    preset = preset.lower()
    if preset == "local_gpu":
        model_config = ModelConfig(board_size=board_size, channels=64, num_blocks=4)
        train_config = TrainConfig(board_size=board_size, batch_size=128)
    elif preset == "local_cpu":
        model_config = ModelConfig(board_size=board_size, channels=32, num_blocks=3)
        train_config = TrainConfig(board_size=board_size, batch_size=64)
    else:
        raise ValueError(f"unknown preset: {preset}")
    return model_config, train_config


def apply_smoke_overrides(
    model_config: ModelConfig,
    train_config: TrainConfig,
) -> Tuple[ModelConfig, TrainConfig]:
    model_config.channels = min(model_config.channels, 16)
    model_config.num_blocks = min(model_config.num_blocks, 1)
    train_config.batch_size = 4
    train_config.replay_size = 256
    train_config.replay_warmup = 4
    train_config.train_every = 1
    train_config.target_update_interval = 8
    train_config.epsilon_decay_steps = 32
    train_config.candidate_mode = "local"
    return model_config, train_config


def select_device(requested: str = "auto") -> torch.device:
    requested = requested.lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def epsilon_by_step(step: int, config: TrainConfig) -> float:
    if config.epsilon_decay_steps <= 0:
        return config.epsilon_end
    fraction = min(1.0, max(0.0, step / float(config.epsilon_decay_steps)))
    return config.epsilon_start + fraction * (config.epsilon_end - config.epsilon_start)


def dataclass_from_dict(cls, data: Dict[str, Any]):
    allowed = {field.name for field in cls.__dataclass_fields__.values()}
    filtered = {key: value for key, value in data.items() if key in allowed}
    return cls(**filtered)


def to_dict(config) -> Dict[str, Any]:
    return asdict(config)
