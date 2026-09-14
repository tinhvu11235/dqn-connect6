"""DQN agent utilities for Connect6."""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn

from config import ModelConfig, TrainConfig, dataclass_from_dict, select_device, to_dict
from connect6 import Connect6Env
from model import QNetwork


def _would_win_after(board: np.ndarray, player: int, action: int, win_length: int) -> bool:
    board_size = board.shape[0]
    row, col = divmod(int(action), board_size)
    if board[row, col] != 0:
        return False

    def count_direction(dr: int, dc: int) -> int:
        count = 0
        r = row + dr
        c = col + dc
        while 0 <= r < board_size and 0 <= c < board_size:
            if board[r, c] != player:
                break
            count += 1
            r += dr
            c += dc
        return count

    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        count = 1 + count_direction(dr, dc) + count_direction(-dr, -dc)
        if count >= win_length:
            return True
    return False


def candidate_actions(
    env: Connect6Env,
    mode: str = "all",
    radius: int = 2,
    include_tactical: bool = True,
) -> List[int]:
    legal = env.get_legal_actions()
    if mode == "all" or not legal:
        return legal
    if mode != "local":
        raise ValueError(f"unknown candidate mode: {mode}")

    board_size = env.board_size
    occupied = np.argwhere(env.board != 0)
    legal_set = set(legal)

    if len(occupied) == 0:
        center = env.encode_action(board_size // 2, board_size // 2)
        return [center] if center in legal_set else legal

    candidates = set()
    for row, col in occupied:
        for r in range(max(0, int(row) - radius), min(board_size, int(row) + radius + 1)):
            for c in range(max(0, int(col) - radius), min(board_size, int(col) + radius + 1)):
                action = r * board_size + c
                if action in legal_set:
                    candidates.add(action)

    if include_tactical:
        player = int(env.current_player)
        opponent = -player
        for action in legal:
            if _would_win_after(env.board, player, action, env.win_length):
                candidates.add(action)
            elif _would_win_after(env.board, opponent, action, env.win_length):
                candidates.add(action)

    if not candidates:
        return legal
    return sorted(candidates)


def masked_argmax(q_values: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
    if q_values.shape != legal_mask.shape:
        raise ValueError("q_values and legal_mask must have the same shape")
    masked = q_values.masked_fill(~legal_mask.bool(), -1.0e9)
    return torch.argmax(masked, dim=-1)


def compute_dqn_targets(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    same_player_next: torch.Tensor,
    next_online_q: torch.Tensor,
    next_target_q: torch.Tensor,
    next_legal_masks: torch.Tensor,
    gamma: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    next_legal_masks = next_legal_masks.bool()
    has_legal = next_legal_masks.any(dim=1)
    masked_next_online = next_online_q.masked_fill(~next_legal_masks, -1.0e9)
    next_actions = torch.argmax(masked_next_online, dim=1)
    next_q = next_target_q.gather(1, next_actions.unsqueeze(1)).squeeze(1)
    next_q = torch.where(has_legal, next_q, torch.zeros_like(next_q))

    continuation_sign = torch.where(
        same_player_next.bool(),
        torch.ones_like(rewards),
        -torch.ones_like(rewards),
    )
    non_terminal = (~dones.bool()).to(dtype=rewards.dtype)
    targets = rewards + gamma * continuation_sign * next_q * non_terminal
    return targets, next_actions


class DQNAgent:
    def __init__(
        self,
        model_config: ModelConfig,
        train_config: Optional[TrainConfig] = None,
        device: str | torch.device = "auto",
    ) -> None:
        self.model_config = model_config
        self.train_config = train_config or TrainConfig(board_size=model_config.board_size)
        self.device = select_device(str(device)) if not isinstance(device, torch.device) else device
        self.online = QNetwork(model_config).to(self.device)
        self.target = QNetwork(model_config).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=self.train_config.learning_rate)
        self.loss_fn = nn.SmoothL1Loss()
        self.optimizer_steps = 0
        self.rng = np.random.default_rng(self.train_config.seed)

    def select_action(
        self,
        env: Connect6Env,
        epsilon: float = 0.0,
        candidate_mode: str = "all",
        candidate_radius: int = 2,
    ) -> int:
        candidates = candidate_actions(env, mode=candidate_mode, radius=candidate_radius)
        if not candidates:
            raise RuntimeError("no legal actions available")
        if self.rng.random() < epsilon:
            return int(self.rng.choice(candidates))

        state = torch.from_numpy(env.encode_state()).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.online(state).squeeze(0)
        mask = torch.zeros(env.num_actions, dtype=torch.bool, device=self.device)
        mask[torch.as_tensor(candidates, dtype=torch.long, device=self.device)] = True
        return int(masked_argmax(q_values, mask).item())

    def optimize(self, replay, batch_size: int, augment: bool = True) -> Optional[float]:
        if len(replay) < batch_size:
            return None

        batch = replay.sample(batch_size, augment=augment)
        states = torch.from_numpy(batch["states"]).to(self.device)
        actions = torch.from_numpy(batch["actions"]).long().to(self.device)
        rewards = torch.from_numpy(batch["rewards"]).to(self.device)
        next_states = torch.from_numpy(batch["next_states"]).to(self.device)
        dones = torch.from_numpy(batch["dones"]).to(self.device)
        same_player_next = torch.from_numpy(batch["same_player_next"]).to(self.device)
        next_legal_masks = torch.from_numpy(batch["next_legal_masks"]).to(self.device)

        q_values = self.online(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_online_q = self.online(next_states)
            next_target_q = self.target(next_states)
            targets, _ = compute_dqn_targets(
                rewards,
                dones,
                same_player_next,
                next_online_q,
                next_target_q,
                next_legal_masks,
                self.train_config.gamma,
            )

        loss = self.loss_fn(q_values, targets)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), self.train_config.grad_clip)
        self.optimizer.step()

        self.optimizer_steps += 1
        if self.optimizer_steps % self.train_config.target_update_interval == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.detach().cpu().item())

    def save_checkpoint(
        self,
        path: str,
        train_state: Optional[Dict[str, float]] = None,
    ) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = {
            "model_config": to_dict(self.model_config),
            "train_config": to_dict(self.train_config),
            "online_state_dict": self.online.state_dict(),
            "target_state_dict": self.target.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "optimizer_steps": self.optimizer_steps,
            "train_state": train_state or {},
        }
        torch.save(payload, path)

    def load_checkpoint(self, path: str, load_optimizer: bool = True) -> Dict[str, object]:
        checkpoint = torch.load(path, map_location=self.device)
        self.online.load_state_dict(checkpoint["online_state_dict"])
        self.target.load_state_dict(checkpoint.get("target_state_dict", checkpoint["online_state_dict"]))
        if load_optimizer and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.optimizer_steps = int(checkpoint.get("optimizer_steps", 0))
        return checkpoint


def load_agent_from_checkpoint(
    path: str,
    device: str | torch.device = "auto",
    load_optimizer: bool = False,
) -> Tuple[DQNAgent, Dict[str, object]]:
    device_obj = select_device(str(device)) if not isinstance(device, torch.device) else device
    checkpoint = torch.load(path, map_location=device_obj)
    model_config = dataclass_from_dict(ModelConfig, checkpoint.get("model_config", {}))
    train_config = dataclass_from_dict(TrainConfig, checkpoint.get("train_config", {}))
    agent = DQNAgent(model_config, train_config=train_config, device=device_obj)
    agent.online.load_state_dict(checkpoint["online_state_dict"])
    agent.target.load_state_dict(checkpoint.get("target_state_dict", checkpoint["online_state_dict"]))
    if load_optimizer and "optimizer_state_dict" in checkpoint:
        agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    agent.optimizer_steps = int(checkpoint.get("optimizer_steps", 0))
    return agent, checkpoint
