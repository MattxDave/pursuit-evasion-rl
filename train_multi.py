"""
Multi-Agent Training Script
===========================
Train pursuers and evaders in the multi-agent pursuit-evasion environment.

Supports:
- Training pursuers only (evaders use heuristics)
- Training evaders only (pursuers use heuristics)
- Alternating training (self-play)
"""

import os
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    CallbackList,
)
from stable_baselines3.common.monitor import Monitor

from src.multi_config import MultiAgentConfig, get_default_multi_config
from src.multi_environment import (
    MultiAgentPursuitEnv,
    PursuerTrainingEnv,
    EvaderTrainingEnv,
)

console = Console()


def make_pursuer_env(config: MultiAgentConfig, seed: int = 0):
    """Create pursuer training environment."""
    def _init():
        env = PursuerTrainingEnv(config=config)
        env = Monitor(env)
        return env
    return _init


def make_evader_env(config: MultiAgentConfig, seed: int = 0):
    """Create evader training environment."""
    def _init():
        env = EvaderTrainingEnv(config=config)
        env = Monitor(env)
        return env
    return _init


def train_pursuers(config: MultiAgentConfig, args: argparse.Namespace):
    """Train pursuer policy."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.name}_pursuers_{timestamp}"
    
    log_dir = Path(config.log_dir) / run_name
    model_dir = Path(config.model_dir) / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    console.print(f"\n[bold cyan]{'='*60}[/bold cyan]")
    console.print(f"[bold]Training: {run_name} (PURSUERS)[/bold]")
    console.print(f"[bold cyan]{'='*60}[/bold cyan]")
    console.print(f"Timesteps: {args.timesteps:,}")
    console.print(f"Environments: {args.n_envs}")
    console.print(f"Stations: {config.station.num_stations}")
    console.print(f"Pursuers: {config.station.num_stations * config.station.pursuers_per_station}")
    console.print(f"Evaders: {config.evader.num_evaders}")
    console.print(f"Log directory: {log_dir}")
    console.print(f"Model directory: {model_dir}")
    console.print(f"[bold cyan]{'='*60}[/bold cyan]\n")
    
    # Create environments
    if args.n_envs > 1:
        env = SubprocVecEnv([make_pursuer_env(config, i) for i in range(args.n_envs)])
        console.print(f"Using SubprocVecEnv with {args.n_envs} processes")
    else:
        env = DummyVecEnv([make_pursuer_env(config)])
    
    # Evaluation environment
    eval_env = DummyVecEnv([make_pursuer_env(config)])
    
    # Create model
    policy_kwargs = dict(
        net_arch=dict(
            pi=list(config.training.policy_layers),
            vf=list(config.training.value_layers),
        )
    )
    
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=config.training.learning_rate,
        n_steps=config.training.n_steps,
        batch_size=config.training.batch_size,
        n_epochs=config.training.n_epochs,
        gamma=config.training.gamma,
        gae_lambda=config.training.gae_lambda,
        clip_range=config.training.clip_range,
        ent_coef=config.training.ent_coef,
        policy_kwargs=policy_kwargs,
        tensorboard_log=str(log_dir / "tensorboard"),
    )
    
    # Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(model_dir / "best"),
        log_path=str(log_dir / "eval"),
        eval_freq=config.training.eval_freq,
        n_eval_episodes=config.training.eval_episodes,
        deterministic=True,
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=config.training.save_freq,
        save_path=str(model_dir / "checkpoints"),
        name_prefix="pursuer_model",
    )
    
    callbacks = CallbackList([eval_callback, checkpoint_callback])
    
    # Train
    console.print("\n[bold green]Starting pursuer training...[/bold green]")
    model.learn(
        total_timesteps=args.timesteps,
        callback=callbacks,
        progress_bar=True,
    )
    
    # Save final model
    final_path = model_dir / "final" / "pursuer_model"
    final_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(final_path))
    console.print(f"\n[bold green]Training complete! Model saved to {final_path}[/bold green]")
    
    # Save config
    config.save(str(model_dir / "config.json"))
    
    env.close()
    eval_env.close()
    
    return model


def train_evaders(config: MultiAgentConfig, args: argparse.Namespace):
    """Train evader policy."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.name}_evaders_{timestamp}"
    
    log_dir = Path(config.log_dir) / run_name
    model_dir = Path(config.model_dir) / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    console.print(f"\n[bold cyan]{'='*60}[/bold cyan]")
    console.print(f"[bold]Training: {run_name} (EVADERS)[/bold]")
    console.print(f"[bold cyan]{'='*60}[/bold cyan]")
    console.print(f"Timesteps: {args.timesteps:,}")
    console.print(f"Environments: {args.n_envs}")
    console.print(f"Evaders: {config.evader.num_evaders}")
    console.print(f"Log directory: {log_dir}")
    console.print(f"Model directory: {model_dir}")
    console.print(f"[bold cyan]{'='*60}[/bold cyan]\n")
    
    # Create environments
    if args.n_envs > 1:
        env = SubprocVecEnv([make_evader_env(config, i) for i in range(args.n_envs)])
        console.print(f"Using SubprocVecEnv with {args.n_envs} processes")
    else:
        env = DummyVecEnv([make_evader_env(config)])
    
    eval_env = DummyVecEnv([make_evader_env(config)])
    
    policy_kwargs = dict(
        net_arch=dict(
            pi=list(config.training.policy_layers),
            vf=list(config.training.value_layers),
        )
    )
    
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=config.training.learning_rate,
        n_steps=config.training.n_steps,
        batch_size=config.training.batch_size,
        n_epochs=config.training.n_epochs,
        gamma=config.training.gamma,
        gae_lambda=config.training.gae_lambda,
        clip_range=config.training.clip_range,
        ent_coef=config.training.ent_coef,
        policy_kwargs=policy_kwargs,
        tensorboard_log=str(log_dir / "tensorboard"),
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(model_dir / "best"),
        log_path=str(log_dir / "eval"),
        eval_freq=config.training.eval_freq,
        n_eval_episodes=config.training.eval_episodes,
        deterministic=True,
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=config.training.save_freq,
        save_path=str(model_dir / "checkpoints"),
        name_prefix="evader_model",
    )
    
    callbacks = CallbackList([eval_callback, checkpoint_callback])
    
    console.print("\n[bold green]Starting evader training...[/bold green]")
    model.learn(
        total_timesteps=args.timesteps,
        callback=callbacks,
        progress_bar=True,
    )
    
    final_path = model_dir / "final" / "evader_model"
    final_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(final_path))
    console.print(f"\n[bold green]Training complete! Model saved to {final_path}[/bold green]")
    
    config.save(str(model_dir / "config.json"))
    
    env.close()
    eval_env.close()
    
    return model


def train_alternating(config: MultiAgentConfig, args: argparse.Namespace):
    """
    Train both pursuers and evaders with alternating phases.
    
    This creates a self-play dynamic where each side improves against the other.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.name}_selfplay_{timestamp}"
    
    log_dir = Path(config.log_dir) / run_name
    model_dir = Path(config.model_dir) / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    console.print(f"\n[bold magenta]{'='*60}[/bold magenta]")
    console.print(f"[bold]Alternating Training (Self-Play): {run_name}[/bold]")
    console.print(f"[bold magenta]{'='*60}[/bold magenta]")
    console.print(f"Total timesteps: {args.timesteps:,}")
    console.print(f"Alternating every: {config.training.alternating_steps:,} steps")
    console.print(f"[bold magenta]{'='*60}[/bold magenta]\n")
    
    # Initialize both models
    pursuer_env = SubprocVecEnv([make_pursuer_env(config, i) for i in range(args.n_envs)])
    evader_env = SubprocVecEnv([make_evader_env(config, i) for i in range(args.n_envs)])
    
    policy_kwargs = dict(
        net_arch=dict(
            pi=list(config.training.policy_layers),
            vf=list(config.training.value_layers),
        )
    )
    
    pursuer_model = PPO(
        "MlpPolicy",
        pursuer_env,
        verbose=0,
        learning_rate=config.training.learning_rate,
        n_steps=config.training.n_steps,
        batch_size=config.training.batch_size,
        policy_kwargs=policy_kwargs,
        tensorboard_log=str(log_dir / "tensorboard_pursuers"),
    )
    
    evader_model = PPO(
        "MlpPolicy",
        evader_env,
        verbose=0,
        learning_rate=config.training.learning_rate,
        n_steps=config.training.n_steps,
        batch_size=config.training.batch_size,
        policy_kwargs=policy_kwargs,
        tensorboard_log=str(log_dir / "tensorboard_evaders"),
    )
    
    # Alternating training loop
    total_steps = 0
    phase_steps = config.training.alternating_steps
    phase = 0
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Training...", total=args.timesteps)
        
        while total_steps < args.timesteps:
            if phase % 2 == 0:
                # Train pursuers
                console.print(f"\n[cyan]Phase {phase + 1}: Training PURSUERS[/cyan]")
                pursuer_model.learn(
                    total_timesteps=phase_steps,
                    reset_num_timesteps=False,
                    progress_bar=False,
                )
            else:
                # Train evaders
                console.print(f"\n[green]Phase {phase + 1}: Training EVADERS[/green]")
                evader_model.learn(
                    total_timesteps=phase_steps,
                    reset_num_timesteps=False,
                    progress_bar=False,
                )
            
            total_steps += phase_steps
            phase += 1
            progress.update(task, completed=total_steps)
            
            # Save checkpoints
            if phase % 4 == 0:  # Every 4 phases
                pursuer_model.save(str(model_dir / f"pursuer_phase_{phase}"))
                evader_model.save(str(model_dir / f"evader_phase_{phase}"))
    
    # Save final models
    pursuer_model.save(str(model_dir / "final" / "pursuer_model"))
    evader_model.save(str(model_dir / "final" / "evader_model"))
    config.save(str(model_dir / "config.json"))
    
    console.print(f"\n[bold green]Self-play training complete![/bold green]")
    console.print(f"Models saved to {model_dir}")
    
    pursuer_env.close()
    evader_env.close()
    
    return pursuer_model, evader_model


def main():
    parser = argparse.ArgumentParser(description="Train multi-agent pursuit-evasion")
    
    # Training mode
    parser.add_argument(
        "--mode", type=str, default="pursuers",
        choices=["pursuers", "evaders", "alternating"],
        help="Training mode: pursuers, evaders, or alternating (self-play)"
    )
    
    # Training parameters
    parser.add_argument("--timesteps", type=int, default=1_000_000, help="Total timesteps")
    parser.add_argument("--n-envs", type=int, default=8, help="Number of parallel environments")
    parser.add_argument("--name", type=str, default="multi_agent", help="Experiment name")
    
    # Environment parameters
    parser.add_argument("--num-stations", type=int, default=2, help="Number of stations")
    parser.add_argument("--num-evaders", type=int, default=3, help="Number of evaders")
    parser.add_argument("--arena-radius", type=float, default=50.0, help="Arena radius")
    
    # Pursuer parameters
    parser.add_argument("--radar-range", type=float, default=15.0, help="Pursuer radar range")
    parser.add_argument("--pursuer-speed", type=float, default=10.0, help="Pursuer max speed")
    
    # Evader parameters
    parser.add_argument("--evader-speed", type=float, default=6.0, help="Evader max speed")
    parser.add_argument("--detection-range", type=float, default=20.0, help="Evader detection range")
    
    args = parser.parse_args()
    
    # Build config
    config = get_default_multi_config()
    
    # Apply command line overrides
    config.station.num_stations = args.num_stations
    config.evader.num_evaders = args.num_evaders
    config.arena.radius = args.arena_radius
    config.pursuer.radar_range = args.radar_range
    config.pursuer.max_speed = args.pursuer_speed
    config.evader.max_speed = args.evader_speed
    config.evader.detection_range = args.detection_range
    
    # Train
    if args.mode == "pursuers":
        train_pursuers(config, args)
    elif args.mode == "evaders":
        train_evaders(config, args)
    elif args.mode == "alternating":
        train_alternating(config, args)


if __name__ == "__main__":
    main()
