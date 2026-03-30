# Legged RL with MuJoCo

Train and run reinforcement learning policies for legged robots in MuJoCo,
with a custom Gymnasium environment for Unitree Go2 and a simple CLI for
training and evaluation.

## What This Project Does

- Provides a custom Gymnasium environment: `UnitreeGo2Env-v0`
- Trains policies with Stable-Baselines3 (`PPO` and `SAC`)
- Supports vectorized training environments and optional Intel XPU training
- Includes CLI to run a random agent or a trained model with rendering

## Prerequisites

- Python `3.12` (project requires `~=3.12`)
- Linux with MuJoCo-compatible graphics/runtime dependencies
- Poetry (recommended for dependency and script management)

## Installation

```bash
git clone https://github.com/kittvnebluda/legged-rl-mujoco.git
cd legged-rl-mujoco
poetry install
```

Run commands through Poetry:

```bash
poetry run lrl --help
```

### Optional: Intel XPU Dependencies

Install the optional XPU dependency group:

```bash
poetry install --with xpu
```

Then train with:

```bash
poetry run lrl train --device xpu
```

## Quickstart

1) Run a Random Agent (rendered)

```bash
poetry run lrl run
```

2) Train a Policy

```bash
poetry run lrl train --name go2 --algo ppo --total-timesteps 50000 --device cpu --n_envs 16
```

3) Run a Trained Model

```bash
poetry run lrl run models/go2_ppo.zip --algo ppo
```

If `--algo` is omitted, the runner tries to infer it from the model filename suffix (for example `_ppo.zip`).

## Train a Policy

The training command:

```bash
poetry run lrl train [OPTIONS]
```

Main options:

- `--name, -n` model name prefix (default: `go2`)
- `--algo, -a` algorithm (`ppo` or `sac`; default: `ppo`)
- `--total-timesteps, -t` number of training timesteps (default: `50000`)
- `--load-model, -l` continue from existing model file
- `--device, -d` compute device (`cpu`, `cuda` if available, or `xpu` when installed)
- `--n_envs` number of vectorized environments (default: `16`)
- `--seed` random seed (auto-generated when not provided)

## Run/Evaluate a Policy

Use:

```bash
poetry run lrl run [MODEL_PATH] [--algo ALGO]
```

- Without `MODEL_PATH`, it launches the random agent rollout.
- With `MODEL_PATH`, it loads a trained SB3 model and runs it in a rendered environment.
- During model execution, debug metrics are printed from the environment each step.

## Training Outputs

By default training writes:

- `models/<name>_<algo>.zip` - final saved model
- `checkpoints/<name>_<algo>_*.zip` - periodic checkpoints (every 50k steps)
- `tb_logs/mujoco_<algo>/` - TensorBoard logs

Visualize logs:

```bash
poetry run tensorboard --logdir tb_logs/mujoco_<algo>
```

## Environment Notes

Environment implementation lives in:

- `src/legged_rl/envs/unitree_go2_env.py`

Environment id:

- `UnitreeGo2Env-v0`

The environment uses:

- 12-dimensional continuous action space (`[-1, 1]`)
- MuJoCo simulation with configurable reward/cost weights
- Randomized command velocities and PD gain randomization for robustness

## Development

### Generate MuJoCo Typing Stubs

The MuJoCo Python package does not ship complete type hints. Generate local stubs:

```bash
poetry run pybind11-stubgen mujoco -o typings
```
