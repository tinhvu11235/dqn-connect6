# Connect6 DQN Baseline

This repository is a modern PyTorch baseline for reinforcement-learning
Connect6. The old Keras/Connect Four files and generated test artifacts have
been removed; the runnable path is:

- `connect6.py`: rules-only Connect6 environment.
- `model.py`: compact residual Q-network.
- `replay.py`: bitboard replay buffer with D4 augmentation.
- `agent.py`: epsilon-greedy Double DQN with legal-action masking.
- `train.py`: self-play training CLI.
- `evaluate.py`: checkpoint evaluation CLI.
- `play_gui.py`: Tkinter GUI for human vs AI play.

## Rules Implemented

- Board defaults to `19 x 19`.
- Internal board values are `0` empty, `1` black, `-1` white.
- Black moves first with exactly one stone.
- After the opening, each turn contains two atomic `step(action)` calls by the
  same player.
- If the first stone of a two-stone turn wins, the game ends immediately and
  the second stone is not placed.
- A player wins with six or more contiguous stones horizontally, vertically,
  diagonally `\`, or diagonally `/`.
- There is no gravity, no column top, and no forbidden cell from Connect Four.

## Action Space

Actions are single-stone atomic moves:

```text
action = row * board_size + col
```

On a `19 x 19` board, the network outputs `361` Q-values. Connect6 two-stone
turns are represented as two consecutive environment steps by the same player.
This avoids the roughly 65k pair-action output layer and keeps local CPU
training practical.

## State Encoding

The network receives a canonical `float32` tensor shaped `(4, board_size,
board_size)` from the perspective of the player to move:

- plane 0: current player's stones.
- plane 1: opponent stones.
- plane 2: `stones_left_in_turn / 2`.
- plane 3: one-hot last move.

A single shared network can therefore play both black and white.

## Bellman Target Sign

Because Connect6 can let the same player act twice in a row, the DQN target
must account for whether the next state is evaluated for the same player or
for the opponent:

```text
terminal:             target = reward
same player next:     target = reward + gamma * next_Q
opponent player next: target = reward - gamma * next_Q
```

`replay.py` stores `same_player_next`, and `agent.compute_dqn_targets` applies
this sign explicitly. Double DQN next-action selection masks illegal occupied
cells before `argmax`.

## Replay Storage

Replay stores boards as bitboards instead of raw `19 x 19` matrices. Each board
uses one `uint64` bitboard array for black stones and one for white stones. For
the default `19 x 19` board, each color needs `ceil(361 / 64) = 6` `uint64`
words. Current and next boards therefore use:

```text
4 * 6 * 8 = 192 bytes per transition
```

The buffer unpacks sampled bitboards back to temporary board tensors only when
building CNN batches and applying D4 augmentation.

## Install

```powershell
python -m pip install -r requirements.txt
```

## Smoke Training

This runs two tiny self-play episodes, performs optimizer updates, and writes a
checkpoint:

```powershell
python train.py --episodes 2 --smoke --device cpu --checkpoint-dir checkpoints/smoke
```

## CPU Training

```powershell
python train.py --episodes 100 --device cpu --preset local_cpu --checkpoint-dir checkpoints/cpu
```

Training uses a `tqdm` progress bar in the terminal. It shows completed
episodes, elapsed time, estimated remaining time, recent episode steps, winner,
replay size, epsilon, loss, per-episode time, and a checkpoint message whenever
`--save-every` is reached. Disable the progress bar when redirecting logs with:

```powershell
python train.py --episodes 100 --no-progress --device cpu --checkpoint-dir checkpoints/cpu
```

## GPU Training

```powershell
python train.py --episodes 10000 --device auto --preset local_gpu --checkpoint-dir checkpoints/gpu
```

`--device auto` uses CUDA when available and otherwise falls back to CPU.

## Resume

```powershell
python train.py --episodes 10000 --device auto --resume checkpoints/gpu/latest.pt --checkpoint-dir checkpoints/gpu
```

Checkpoints restore the online network, target network, optimizer, optimizer
step count, and basic training counters. The replay buffer is rebuilt after
resume to keep checkpoint files small.

## Evaluate

```powershell
python evaluate.py --checkpoint checkpoints/cpu/latest.pt --games 20 --device cpu
```

Evaluation reports:

- network vs random.
- network vs lightweight heuristic.
- greedy network self-play.

## Play Against The AI

After training, start the GUI with the checkpoint:

```powershell
python play_gui.py --checkpoint checkpoints/cpu/latest.pt --device cpu --human black
```

Use CUDA automatically if available:

```powershell
python play_gui.py --checkpoint checkpoints/gpu/latest.pt --device auto --human white
```

If `--human black`, you move first. If `--human white`, the AI makes the first
black opening move. Click board intersections to place stones. During normal
Connect6 turns, the same player clicks twice unless the first stone wins
immediately.

You can also open the board without a checkpoint; in that mode the AI uses a
simple heuristic fallback:

```powershell
python play_gui.py --human black
```

## Local Resource Estimate

The default `local_cpu` model has about 4.2 million parameters. Model plus
optimizer state is usually under 100 MB of RAM. The default bitboard replay is
roughly 10 MB for board data at `50_000` transitions, and about 11 MB including
stored scalar metadata. A normal CPU run should fit comfortably in a few
hundred MB.

The `local_gpu` preset uses wider channels and can use a few hundred MB of VRAM
depending on batch size and driver overhead.

## Remaining Bottlenecks

- Pure self-play with sparse terminal reward learns slowly on Connect6.
- Local candidate pruning improves early training speed but is not a strong
  search policy.
- The Q-network has no explicit threat head or policy/value split.
- Replay is uniform, not prioritized.
- No MCTS or snapshot opponent pool is enabled in phase 1.

## Phase 2 Ideas

- Add a stronger heuristic curriculum and threat-based reward shaping behind a
  disabled-by-default flag.
- Add prioritized replay.
- Add snapshot opponent self-play.
- Add policy/value heads and upgrade toward AlphaZero-style training.
- Add lightweight MCTS only after the rules, masks, and Bellman logic remain
  covered by tests.
