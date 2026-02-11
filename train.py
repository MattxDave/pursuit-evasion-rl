#!/usr/bin/env python3
"""
Training Script
===============
Train pursuit-evasion agents using PPO.

Usage:
    python train.py                     # Default training
    python train.py --preset fast       # Quick test run
    python train.py --preset full       # Full training
    python train.py --timesteps 500000  # Custom timesteps
"""

import os
import sys
import argparse
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.utils import set_random_seed

from src.environment import PursuitEnv
from src.config import Config, get_config


def make_env(config: Config, seed: int, rank: int):
    """Create environment factory."""
    def _init():
        env = PursuitEnv(config=config)
        env.reset(seed=seed + rank)
        return Monitor(env)
    return _init


def train(config: Config, args):
    """Run training."""
    
    # Setup directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{config.experiment_name}_{timestamp}"
    log_dir = Path(config.log_dir) / run_name
    model_dir = Path(config.model_dir) / run_name
    
    log_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config.save(log_dir / "config.json")
    
    print("=" * 60)
    print(f"Training: {run_name}")
    print("=" * 60)
    print(f"Timesteps: {config.training.total_timesteps:,}")
    print(f"Environments: {config.training.n_envs}")
    print(f"Evader speed: {config.env.evader_min_speed}-{config.env.evader_max_speed} m/s")
    print(f"Pursuer max speed: {config.env.pursuer_max_speed} m/s")
    print(f"Heading nudge: ±{config.env.heading_nudge_deg}°")
    print(f"Log directory: {log_dir}")
    print(f"Model directory: {model_dir}")
    print("=" * 60)
    
    # Set seed
    set_random_seed(config.training.seed)
    
    # Create vectorized environments
    env_fns = [
        make_env(config, config.training.seed, i) 
        for i in range(config.training.n_envs)
    ]
    
    try:
        env = SubprocVecEnv(env_fns)
        print(f"Using SubprocVecEnv with {config.training.n_envs} processes")
    except Exception as e:
        print(f"SubprocVecEnv failed ({e}), falling back to DummyVecEnv")
        env = DummyVecEnv(env_fns)
    
    # Evaluation environment
    eval_env = DummyVecEnv([make_env(config, 9999, 0)])
    
    # Policy network architecture
    policy_kwargs = dict(
        net_arch=dict(
            pi=list(config.training.policy_layers),
            vf=list(config.training.value_layers),
        )
    )
    
    # Create PPO model
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        gamma=config.training.gamma,
        n_steps=config.training.n_steps,
        batch_size=config.training.batch_size,
        n_epochs=config.training.n_epochs,
        learning_rate=config.training.learning_rate,
        gae_lambda=config.training.gae_lambda,
        clip_range=config.training.clip_range,
        ent_coef=config.training.ent_coef,
        vf_coef=config.training.vf_coef,
        max_grad_norm=config.training.max_grad_norm,
        policy_kwargs=policy_kwargs,
        seed=config.training.seed,
        device=config.training.device,
        tensorboard_log=str(log_dir / "tensorboard"),
    )
    
    # Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(model_dir / "best"),
        log_path=str(log_dir),
        eval_freq=config.training.eval_freq,
        n_eval_episodes=config.training.eval_episodes,
        deterministic=True,
        render=False,
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=config.training.save_freq,
        save_path=str(model_dir / "checkpoints"),
        name_prefix="model",
    )
    
    # Train
    print("\nStarting training...")
    model.learn(
        total_timesteps=config.training.total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True,
    )
    
    # Save final model
    final_path = model_dir / "final_model"
    model.save(str(final_path))
    print(f"\nTraining complete! Final model saved to: {final_path}")
    
    # Cleanup
    env.close()
    eval_env.close()
    
    return model, str(model_dir)


def main():
    parser = argparse.ArgumentParser(description="Train pursuit-evasion agent")
    
    parser.add_argument("--preset", type=str, default="default",
                       choices=["default", "fast", "full"],
                       help="Training preset")
    parser.add_argument("--timesteps", type=int, default=None,
                       help="Override total timesteps")
    parser.add_argument("--n-envs", type=int, default=None,
                       help="Override number of environments")
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed")
    parser.add_argument("--name", type=str, default=None,
                       help="Experiment name")
    parser.add_argument("--evader-noise-std", type=float, default=None,
                        help="Evader heading noise std (radians) during training")
    parser.add_argument("--evader-noise-prob", type=float, default=None,
                        help="Per-step probability of applying evader heading noise during training")
    
    args = parser.parse_args()
    
    # Get config
    config = get_config(args.preset)
    
    # Apply overrides
    if args.timesteps:
        config.training.total_timesteps = args.timesteps
    if args.n_envs:
        config.training.n_envs = args.n_envs
    if args.seed:
        config.training.seed = args.seed
    if args.name:
        config.experiment_name = args.name
    if args.evader_noise_std is not None:
        config.evader_noise.heading_noise_std = args.evader_noise_std
    if args.evader_noise_prob is not None:
        config.evader_noise.noise_prob = args.evader_noise_prob
    
    # Run training
    train(config, args)


if __name__ == "__main__":
    main()
