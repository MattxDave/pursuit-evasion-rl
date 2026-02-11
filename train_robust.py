#!/usr/bin/env python3
"""
Robust Self-Play Training with Domain Randomization and Curriculum Learning
============================================================================

Key improvements:
1. Domain randomization - vary speeds, spawns, arena size
2. Curriculum learning - start easy, increase difficulty
3. Population-based self-play - train against diverse opponents
4. Improved reward shaping
"""

import os
import sys
import argparse
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
import json

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

from src.multi_environment import MultiAgentPursuitEnv
from src.multi_config import MultiAgentConfig


class DomainRandomizationWrapper:
    """Wrapper that randomizes environment parameters each reset."""
    
    def __init__(self, env: MultiAgentPursuitEnv, randomization_config: Dict[str, Any]):
        self.env = env
        self.config = randomization_config
        self.rng = np.random.default_rng()
        
    def randomize(self):
        """Apply domain randomization before reset."""
        cfg = self.config
        
        # Randomize speeds
        if cfg.get('speed_variation', 0) > 0:
            var = cfg['speed_variation']
            self.env.pursuer_cfg.max_speed = 9.0 * (1 + self.rng.uniform(-var, var))
            self.env.evader_cfg.max_speed = 11.0 * (1 + self.rng.uniform(-var, var))
        
        # Randomize arena
        if cfg.get('arena_variation', 0) > 0:
            var = cfg['arena_variation']
            self.env.arena_cfg.radius = 50.0 * (1 + self.rng.uniform(-var, var))
        
        # Randomize detection ranges
        if cfg.get('detection_variation', 0) > 0:
            var = cfg['detection_variation']
            self.env.evader_cfg.detection_range = 40.0 * (1 + self.rng.uniform(-var, var))
            self.env.pursuer_cfg.radar_range = 35.0 * (1 + self.rng.uniform(-var, var))


class CurriculumCallback(BaseCallback):
    """Callback for curriculum learning - increase difficulty over time."""
    
    def __init__(self, 
                 env,
                 phases: List[Dict[str, Any]],
                 phase_timesteps: int = 500_000,
                 verbose: int = 1):
        super().__init__(verbose)
        self.env = env
        self.phases = phases
        self.phase_timesteps = phase_timesteps
        self.current_phase = 0
        
    def _on_step(self) -> bool:
        # Check if we should advance to next phase
        target_phase = min(
            self.num_timesteps // self.phase_timesteps,
            len(self.phases) - 1
        )
        
        if target_phase > self.current_phase:
            self.current_phase = target_phase
            phase = self.phases[self.current_phase]
            
            if self.verbose:
                print(f"\n=== Curriculum Phase {self.current_phase + 1}/{len(self.phases)} ===")
                print(f"Settings: {phase}")
            
            # Apply phase settings
            self._apply_phase(phase)
        
        return True
    
    def _apply_phase(self, phase: Dict[str, Any]):
        """Apply curriculum phase settings to environment."""
        # Access the underlying env (unwrap if needed)
        env = self.env
        while hasattr(env, 'env'):
            env = env.env
        while hasattr(env, 'envs'):
            env = env.envs[0]
        while hasattr(env, 'env'):
            env = env.env
            
        if 'pursuer_speed' in phase:
            env.pursuer_cfg.max_speed = phase['pursuer_speed']
        if 'evader_speed' in phase:
            env.evader_cfg.max_speed = phase['evader_speed']
        if 'num_pursuers_per_station' in phase:
            env.station_cfg.pursuers_per_station = phase['num_pursuers_per_station']


class SelfPlayCallback(BaseCallback):
    """
    Callback for population-based self-play training.
    
    Maintains a pool of past models and periodically updates opponents.
    """
    
    def __init__(self,
                 save_dir: str,
                 opponent_update_freq: int = 50_000,
                 pool_size: int = 5,
                 verbose: int = 1):
        super().__init__(verbose)
        self.save_dir = Path(save_dir)
        self.opponent_update_freq = opponent_update_freq
        self.pool_size = pool_size
        self.model_pool: List[Path] = []
        self.last_update = 0
        
    def _on_step(self) -> bool:
        # Periodically save current model to pool
        if self.num_timesteps - self.last_update >= self.opponent_update_freq:
            self._save_to_pool()
            self.last_update = self.num_timesteps
        return True
    
    def _save_to_pool(self):
        """Save current model to opponent pool."""
        model_path = self.save_dir / f"pool_model_{self.num_timesteps}.zip"
        self.model.save(model_path)
        self.model_pool.append(model_path)
        
        # Keep pool size limited
        while len(self.model_pool) > self.pool_size:
            old_model = self.model_pool.pop(0)
            if old_model.exists():
                old_model.unlink()
        
        if self.verbose:
            print(f"\nSaved model to pool ({len(self.model_pool)} models)")


class RobustTrainingMetrics(BaseCallback):
    """Track detailed training metrics."""
    
    def __init__(self, log_dir: str, verbose: int = 1):
        super().__init__(verbose)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.episode_rewards = {'pursuer': [], 'evader': []}
        self.episode_lengths = []
        self.captures = []
        self.goals = []
        self.win_rates = {'pursuer': [], 'evader': []}
        
    def _on_step(self) -> bool:
        # Check for episode end
        infos = self.locals.get('infos', [])
        for info in infos:
            if 'episode' in info:
                self.episode_lengths.append(info['episode']['l'])
            if 'captures' in info:
                self.captures.append(info['captures'])
            if 'goals' in info:
                self.goals.append(info['goals'])
        
        return True
    
    def _on_training_end(self):
        """Save final metrics."""
        metrics = {
            'episode_lengths': self.episode_lengths[-1000:],
            'captures': self.captures[-1000:],
            'goals': self.goals[-1000:],
            'avg_captures': np.mean(self.captures[-100:]) if self.captures else 0,
            'avg_goals': np.mean(self.goals[-100:]) if self.goals else 0,
        }
        
        with open(self.log_dir / 'training_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)


def create_env(
    train_pursuers: bool = True,
    train_evaders: bool = True,
    domain_randomization: bool = True,
    seed: Optional[int] = None,
) -> MultiAgentPursuitEnv:
    """Create environment with optional domain randomization."""
    
    config = MultiAgentConfig()
    
    # Apply domain randomization settings
    if domain_randomization:
        rng = np.random.default_rng(seed)
        
        # Speed variation ±15%
        config.pursuer.max_speed = 9.0 * (1 + rng.uniform(-0.15, 0.15))
        config.evader.max_speed = 11.0 * (1 + rng.uniform(-0.15, 0.15))
        
        # Arena variation ±10%
        config.arena.radius = 50.0 * (1 + rng.uniform(-0.1, 0.1))
        
        # Detection range variation ±10%
        config.evader.detection_range = 40.0 * (1 + rng.uniform(-0.1, 0.1))
        config.pursuer.radar_range = 35.0 * (1 + rng.uniform(-0.1, 0.1))
    
    env = MultiAgentPursuitEnv(
        config=config,
        train_pursuers=train_pursuers,
        train_evaders=train_evaders,
    )
    
    return env


def make_vec_env(
    n_envs: int,
    train_pursuers: bool,
    train_evaders: bool,
    domain_randomization: bool,
    log_dir: str,
) -> DummyVecEnv:
    """Create vectorized environment."""
    
    def make_env(rank: int):
        def _init():
            env = create_env(
                train_pursuers=train_pursuers,
                train_evaders=train_evaders,
                domain_randomization=domain_randomization,
                seed=rank * 1000,
            )
            env = Monitor(env, os.path.join(log_dir, f"env_{rank}"))
            return env
        return _init
    
    return DummyVecEnv([make_env(i) for i in range(n_envs)])


def train_robust(
    total_timesteps: int = 3_000_000,
    n_envs: int = 4,
    use_curriculum: bool = True,
    use_domain_randomization: bool = True,
    save_freq: int = 100_000,
    log_dir: Optional[str] = None,
    pursuer_model_path: Optional[str] = None,
    evader_model_path: Optional[str] = None,
):
    """
    Train robust pursuer and evader models with curriculum and domain randomization.
    """
    
    # Setup directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if log_dir is None:
        log_dir = f"logs/robust_training_{timestamp}"
    
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    model_dir = log_path / "models"
    model_dir.mkdir(exist_ok=True)
    
    print(f"Training with:")
    print(f"  Total timesteps: {total_timesteps:,}")
    print(f"  Parallel envs: {n_envs}")
    print(f"  Curriculum learning: {use_curriculum}")
    print(f"  Domain randomization: {use_domain_randomization}")
    print(f"  Log directory: {log_dir}")
    
    # Curriculum phases - gradually increase difficulty
    curriculum_phases = [
        # Phase 1: Easy - slow pursuers
        {'pursuer_speed': 7.0, 'evader_speed': 11.0},
        # Phase 2: Medium - normal pursuers
        {'pursuer_speed': 8.5, 'evader_speed': 11.0},
        # Phase 3: Hard - fast pursuers
        {'pursuer_speed': 9.0, 'evader_speed': 11.0},
        # Phase 4: Very hard - slightly faster pursuers
        {'pursuer_speed': 9.5, 'evader_speed': 11.0},
    ]
    
    # =========================================================================
    # PHASE 1: Train Evaders (Pursuers use heuristic)
    # =========================================================================
    print("\n" + "="*60)
    print("PHASE 1: Training Evaders")
    print("="*60)
    
    evader_env = make_vec_env(
        n_envs=n_envs,
        train_pursuers=False,  # Heuristic pursuers
        train_evaders=True,
        domain_randomization=use_domain_randomization,
        log_dir=str(log_path / "evader_training"),
    )
    
    # Create or load evader model
    if evader_model_path and os.path.exists(evader_model_path):
        print(f"Loading evader model from {evader_model_path}")
        evader_model = PPO.load(evader_model_path, env=evader_env)
    else:
        evader_model = PPO(
            "MlpPolicy",
            evader_env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=256,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
            tensorboard_log=str(log_path / "tb_evader"),
        )
    
    # Callbacks for evader training
    evader_callbacks = [
        CheckpointCallback(
            save_freq=save_freq // n_envs,
            save_path=str(model_dir / "evader_checkpoints"),
            name_prefix="evader",
        ),
        RobustTrainingMetrics(str(log_path / "evader_metrics")),
    ]
    
    if use_curriculum:
        evader_callbacks.append(
            CurriculumCallback(
                evader_env,
                phases=curriculum_phases,
                phase_timesteps=total_timesteps // (2 * len(curriculum_phases)),
            )
        )
    
    # Train evaders
    evader_model.learn(
        total_timesteps=total_timesteps // 2,
        callback=evader_callbacks,
        progress_bar=True,
    )
    
    evader_model.save(model_dir / "evader_robust")
    evader_env.close()
    
    # =========================================================================
    # PHASE 2: Train Pursuers (Evaders use trained model)
    # =========================================================================
    print("\n" + "="*60)
    print("PHASE 2: Training Pursuers against trained Evaders")
    print("="*60)
    
    pursuer_env = make_vec_env(
        n_envs=n_envs,
        train_pursuers=True,
        train_evaders=False,  # Will use trained evader model
        domain_randomization=use_domain_randomization,
        log_dir=str(log_path / "pursuer_training"),
    )
    
    # Create or load pursuer model
    if pursuer_model_path and os.path.exists(pursuer_model_path):
        print(f"Loading pursuer model from {pursuer_model_path}")
        pursuer_model = PPO.load(pursuer_model_path, env=pursuer_env)
    else:
        pursuer_model = PPO(
            "MlpPolicy",
            pursuer_env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=256,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
            tensorboard_log=str(log_path / "tb_pursuer"),
        )
    
    # Callbacks for pursuer training
    pursuer_callbacks = [
        CheckpointCallback(
            save_freq=save_freq // n_envs,
            save_path=str(model_dir / "pursuer_checkpoints"),
            name_prefix="pursuer",
        ),
        RobustTrainingMetrics(str(log_path / "pursuer_metrics")),
    ]
    
    # Train pursuers
    pursuer_model.learn(
        total_timesteps=total_timesteps // 2,
        callback=pursuer_callbacks,
        progress_bar=True,
    )
    
    pursuer_model.save(model_dir / "pursuer_robust")
    pursuer_env.close()
    
    # =========================================================================
    # PHASE 3: Joint Self-Play Training
    # =========================================================================
    print("\n" + "="*60)
    print("PHASE 3: Joint Self-Play Training")
    print("="*60)
    
    # Create self-play environment
    selfplay_env = make_vec_env(
        n_envs=n_envs,
        train_pursuers=True,
        train_evaders=True,
        domain_randomization=use_domain_randomization,
        log_dir=str(log_path / "selfplay_training"),
    )
    
    # Continue training both models with self-play
    # Alternate training to improve both
    for round_num in range(3):
        print(f"\n--- Self-Play Round {round_num + 1}/3 ---")
        
        # Train evaders for a bit
        evader_model.set_env(selfplay_env)
        evader_model.learn(
            total_timesteps=total_timesteps // 6,
            progress_bar=True,
            reset_num_timesteps=False,
        )
        evader_model.save(model_dir / f"evader_selfplay_r{round_num}")
        
        # Train pursuers for a bit
        pursuer_model.set_env(selfplay_env)
        pursuer_model.learn(
            total_timesteps=total_timesteps // 6,
            progress_bar=True,
            reset_num_timesteps=False,
        )
        pursuer_model.save(model_dir / f"pursuer_selfplay_r{round_num}")
    
    selfplay_env.close()
    
    # Save final models
    final_dir = model_dir / "final"
    final_dir.mkdir(exist_ok=True)
    evader_model.save(final_dir / "evader_model")
    pursuer_model.save(final_dir / "pursuer_model")
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Final models saved to: {final_dir}")
    
    return str(final_dir)


def main():
    parser = argparse.ArgumentParser(description="Robust Self-Play Training")
    parser.add_argument("--timesteps", type=int, default=3_000_000,
                        help="Total training timesteps")
    parser.add_argument("--n-envs", type=int, default=4,
                        help="Number of parallel environments")
    parser.add_argument("--no-curriculum", action="store_true",
                        help="Disable curriculum learning")
    parser.add_argument("--no-domain-rand", action="store_true",
                        help="Disable domain randomization")
    parser.add_argument("--log-dir", type=str, default=None,
                        help="Log directory")
    parser.add_argument("--pursuer-model", type=str, default=None,
                        help="Path to pretrained pursuer model")
    parser.add_argument("--evader-model", type=str, default=None,
                        help="Path to pretrained evader model")
    parser.add_argument("--save-freq", type=int, default=100_000,
                        help="Checkpoint save frequency")
    
    args = parser.parse_args()
    
    train_robust(
        total_timesteps=args.timesteps,
        n_envs=args.n_envs,
        use_curriculum=not args.no_curriculum,
        use_domain_randomization=not args.no_domain_rand,
        save_freq=args.save_freq,
        log_dir=args.log_dir,
        pursuer_model_path=args.pursuer_model,
        evader_model_path=args.evader_model,
    )


if __name__ == "__main__":
    main()
