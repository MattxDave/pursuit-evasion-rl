# Pursuit-Evasion Hybrid RL (Multi-Agent)

A modular reinforcement learning system for training autonomous pursuers and evaders in multi-agent pursuit-evasion scenarios.

## 🎯 Features

### Single-Agent Mode (Original)
- **Classical Control**: Lead-intercept algorithm for optimal heading computation
- **RL Enhancement**: Neural network learns speed control and heading refinements
- Train a pursuer to intercept a moving evader

### Multi-Agent Mode (NEW)
- **Multiple Stations**: Scattered across the arena, each guarded by 2 pursuers
- **Smart Pursuers**: Radar detection, inter-pursuer communication, state machine (Guard/Intercept/Return/Recharge)
- **Smart Evaders**: RL-controlled evasion, flock formation (wedge/line/diamond), decoy sacrifice behavior
- **Coordination**: Pursuers decide when both should intercept vs. one stays on guard
- **Communication**: Threat broadcasting between pursuers, flock coordination between evaders

## 📁 Project Structure

```
pursuit_hybrid/
├── train.py              # Single-agent training
├── train_multi.py        # Multi-agent training (NEW)
├── simulate.py           # Single-agent visualization
├── simulate_multi.py     # Multi-agent visualization (NEW)
├── src/
│   ├── __init__.py
│   ├── config.py         # Single-agent config
│   ├── multi_config.py   # Multi-agent config (NEW)
│   ├── environment.py    # Single-agent environment
│   ├── multi_environment.py  # Multi-agent environment (NEW)
│   ├── agents.py         # Original Pursuer/Evader
│   ├── multi_agents.py   # Station, SmartPursuer, SmartEvader, FlockController (NEW)
│   ├── geometry.py       # Lead intercept, pure pursuit
│   └── rewards.py        # Reward calculation
├── logs/                 # Training logs & tensorboard
└── models/               # Saved model checkpoints
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install stable-baselines3 gymnasium numpy matplotlib pyyaml
```

### 2. Train a Model

```bash
# Default training (1M steps)
python train.py

# Quick test run (100K steps)
python train.py --preset fast

# Full training (3M steps)
python train.py --preset full

# Custom training
python train.py --timesteps 500000 --n-envs 16
```

### 3. Visualize Results

```bash
# Run simulation with trained model
python simulate.py

# Specific evader speed
python simulate.py --evader-speed 4.0

# Save video
python simulate.py --save-video pursuit.mp4

# Benchmark performance
python simulate.py --benchmark
```

---

## 🆕 Multi-Agent Mode

### Train Multi-Agent

```bash
# Train pursuer team (evaders use heuristics)
python train_multi.py --mode pursuers --timesteps 1000000 --n-envs 8

# Train evader flock (pursuers use heuristics)  
python train_multi.py --mode evaders --timesteps 1000000 --n-envs 8

# Self-play: alternating training
python train_multi.py --mode alternating --timesteps 2000000

# Custom scenario
python train_multi.py --mode pursuers \
    --num-stations 3 \
    --num-evaders 5 \
    --radar-range 20 \
    --arena-radius 60
```

### Simulate Multi-Agent

```bash
# Run with heuristic agents
python simulate_multi.py

# Run with trained pursuer model
python simulate_multi.py --pursuer-model models/pursuers_xxx/best/best_model.zip

# Run with both trained models
python simulate_multi.py \
    --pursuer-model models/pursuers_xxx/best/best_model.zip \
    --evader-model models/evaders_xxx/best/best_model.zip

# Run batch evaluation
python simulate_multi.py --episodes 50 --no-render

# Save animation
python simulate_multi.py --save-animation episode.gif

# Save trajectory plot
python simulate_multi.py --save-plot trajectories.png
```

### Multi-Agent System Design

#### Stations
- 2 stations (configurable) randomly placed on the arena
- Each station is guarded by 2 pursuers
- Pursuers return to stations to recharge

#### Pursuers (Guard Team)
- **Radar System**: Detects evaders within range (default 15m)
- **State Machine**:
  - `GUARDING`: At station, scanning for threats
  - `INTERCEPTING`: Chasing detected evader
  - `RETURNING`: Going back to station
  - `RECHARGING`: At station, restoring battery
- **Communication**: Broadcast threats to partner
- **Coordination**: Decide if one or both should intercept

#### Evaders (Flock)
- **RL-Controlled**: Each evader has own policy for evasion/goal-seeking
- **Formation**: Wedge, line, or diamond formation
- **Roles**:
  - `LEADER`: Front of formation, sets direction
  - `FOLLOWER`: Maintains formation
  - `DECOY`: Sacrifices to draw pursuers away
- **Communication**: Share detected threats with flock
- **Goal**: Reach the goal area on opposite side of arena

## Configuration

### Training Presets

| Preset | Timesteps | Envs | Description |
|--------|-----------|------|-------------|
| `fast` | 100K | 4 | Quick testing |
| `default` | 1M | 8 | Standard training |
| `full` | 3M | 16 | Best performance |

### Environment Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `radius` | 25.0 | Arena radius (meters) |
| `capture_radius` | 0.5 | Distance to catch evader |
| `pursuer_max_speed` | 3.0 | Pursuer max speed (m/s) |
| `evader_min_speed` | 0.6 | Evader min speed (m/s) |
| `evader_max_speed` | 5.0 | Evader max speed (m/s) |
| `heading_nudge_deg` | 15.0 | Max RL heading adjustment (±°) |

### Reward Structure

| Event | Reward |
|-------|--------|
| Capture | +15 |
| Evader escapes | -10 |
| Evader reaches goal | -12 |
| Timeout | -5 |
| Closing distance | +1.0 × Δdist |

##  How It Works

### Hybrid Architecture

1. **Classical Base**: Lead-intercept algorithm computes optimal heading to intercept moving target
2. **RL Control**: PPO learns:
   - **Throttle**: Speed as fraction of max (0.5 - 1.0)
   - **Heading Nudge**: Small angular adjustments (±15°)

### Observation Space (8D)

```
[rel_x, rel_y, evader_vx, evader_vy, dist_to_goal, goal_dir_x, goal_dir_y, speed_ratio]
```

### Action Space (2D)

```
[throttle, heading_nudge]
```

### Curriculum Learning

Training automatically increases difficulty:
- **Start**: 80% slow evaders (0.6-2.0 m/s)
- **End**: 90% fast evaders (3.0-5.0 m/s)

## 📊 Expected Performance

After training with `default` preset:

| Evader Speed | Capture Rate |
|--------------|--------------|
| 1.0 m/s | ~99% |
| 2.0 m/s | ~95% |
| 3.0 m/s | ~80% |
| 4.0 m/s | ~50% |
| 5.0 m/s | ~25% |

*Note: Capture becomes geometrically harder when evader exceeds pursuer speed (3.0 m/s)*

## 🔧 Development

### Module Overview

- **`config.py`**: Dataclass-based configuration with presets
- **`geometry.py`**: Mathematical functions (lead intercept, pure pursuit)
- **`agents.py`**: Agent state and dynamics
- **`rewards.py`**: Modular reward computation
- **`environment.py`**: Gymnasium-compatible environment

### Extending

```python
from src import PursuitEnv, Config

# Custom configuration
config = Config()
config.env.pursuer_max_speed = 4.0
config.rewards.capture = 20.0

# Create environment
env = PursuitEnv(config=config)
```

##  References

- Lead Intercept: Classical missile guidance
- PPO: [Schulman et al., 2017](https://arxiv.org/abs/1707.06347)
- Stable-Baselines3: [Documentation](https://stable-baselines3.readthedocs.io/)

## 🐳 Docker

### Build Image

```bash
docker build -t pursuit-evasion-rl .
```

### Run Training

```bash
# Quick training in container
docker run -v $(pwd)/models:/app/models -v $(pwd)/logs:/app/logs \
    pursuit-evasion-rl python train.py --timesteps 500000 --n-envs 8 --name docker_run

# Or use docker-compose
docker-compose up train
```

### Run Evaluation

```bash
# Compare hybrid vs classical controller
docker run -v $(pwd)/models:/app/models -v $(pwd)/eval_comparison:/app/eval_comparison \
    pursuit-evasion-rl python compare_hybrid_vs_classical.py \
    --model models/docker_run/best/best_model.zip \
    --episodes 50 --evader-speed 5.0 --no-plot
```

### Development Shell

```bash
# Interactive shell with source mounted
docker-compose run dev

# Or directly
docker run -it -v $(pwd):/app pursuit-evasion-rl /bin/bash
```

### Docker Compose Services

| Service | Description |
|---------|-------------|
| `train` | Run full training (1M steps) |
| `evaluate` | Compare hybrid vs classical |
| `dev` | Interactive development shell |

