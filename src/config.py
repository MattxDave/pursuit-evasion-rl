"""
Configuration Management
========================
Centralized configuration for training, environment, and rewards.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any
import json
import yaml
from pathlib import Path


@dataclass
class EnvironmentConfig:
    """Environment parameters."""
    # Arena
    radius: float = 25.0
    capture_radius: float = 0.5
    
    # Agent speeds
    pursuer_max_speed: float = 10.0
    evader_min_speed: float = 0.6
    evader_max_speed: float = 5.0
    
    # Throttle mapping (action -> speed fraction)
    throttle_min: float = 0.5   # Minimum speed = 50% of max
    throttle_max: float = 1.0   # Maximum speed = 100% of max
    
    # Battery system
    battery_capacity: float = 100.0      # Full battery
    battery_drain_rate: float = 0.3      # Drain per step at full speed (quadratic scaling)
    battery_min_speed_factor: float = 0.4  # Min speed when battery depleted (40% of max)
    
    # Heading control
    heading_nudge_deg: float = 30.0  # Max RL heading adjustment (±degrees)
    turn_rate_deg: float = 0.0       # Turn rate limit (deg/s), 0 = unlimited
    ignore_battery_limits: bool = True  # If True, pursuer ignores battery speed limits
    
    # Time
    dt: float = 0.1
    max_steps: int = 600  # 60 seconds
    
    # Spawn settings
    min_separation: float = 2.0
    
    def __post_init__(self):
        """Validate configuration."""
        assert self.radius > 0, "Radius must be positive"
        assert self.capture_radius > 0, "Capture radius must be positive"
        assert self.pursuer_max_speed > 0, "Pursuer speed must be positive"
        assert 0 <= self.throttle_min <= self.throttle_max <= 1.0


@dataclass
class EvaderNoiseConfig:
    """Noise model for evader heading jitter."""
    heading_noise_std: float = 0.0  # radians, 0 disables noise
    noise_prob: float = 0.0         # probability per step to apply noise


@dataclass
class RewardConfig:
    """Reward shaping parameters."""
    # Terminal rewards
    capture: float = 15.0
    evader_escaped: float = -10.0
    evader_reached_goal: float = -12.0
    timeout: float = -5.0
    
    # Shaping rewards (per step)
    closing_distance: float = 1.0      # Reward for reducing distance
    time_penalty: float = -0.01        # Small penalty per step
    
    # Optional penalties
    low_speed_penalty: float = 0.0     # Penalty for going slow when should chase
    
    # Battery rewards
    battery_efficiency_bonus: float = 0.02   # Bonus for energy-efficient chasing
    low_battery_penalty: float = -0.1        # Penalty when battery < 20%


@dataclass  
class CurriculumConfig:
    """Curriculum learning settings."""
    enabled: bool = True
    
    # Easy range (slow evaders)
    easy_speed_min: float = 0.6
    easy_speed_max: float = 2.0
    
    # Hard range (fast evaders)  
    hard_speed_min: float = 3.0
    hard_speed_max: float = 5.0
    
    # Progression
    initial_hard_prob: float = 0.2   # Start with 20% hard
    final_hard_prob: float = 0.9     # End with 90% hard
    warmup_episodes: int = 50_000    # Episodes to reach final difficulty


@dataclass
class TrainingConfig:
    """PPO training hyperparameters."""
    # Core
    total_timesteps: int = 3_000_000
    n_envs: int = 8
    
    # PPO specific
    n_steps: int = 2048
    batch_size: int = 256
    n_epochs: int = 10
    gamma: float = 0.995
    gae_lambda: float = 0.98
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    learning_rate: float = 3e-4
    max_grad_norm: float = 0.5
    
    # Network architecture
    policy_layers: tuple = (128, 128)
    value_layers: tuple = (128, 128)
    
    # Checkpointing
    eval_freq: int = 10_000
    eval_episodes: int = 20
    save_freq: int = 50_000
    
    # Misc
    seed: int = 42
    device: str = "auto"


@dataclass
class Config:
    """Master configuration combining all settings."""
    env: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    rewards: RewardConfig = field(default_factory=RewardConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    evader_noise: EvaderNoiseConfig = field(default_factory=EvaderNoiseConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    # Experiment tracking
    experiment_name: str = "pursuit_hybrid"
    log_dir: str = "logs"
    model_dir: str = "models"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def save(self, path: str):
        """Save configuration to file."""
        path = Path(path)
        data = self.to_dict()
        
        if path.suffix == ".json":
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
        elif path.suffix in (".yaml", ".yml"):
            with open(path, "w") as f:
                yaml.dump(data, f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported format: {path.suffix}")
    
    @classmethod
    def load(cls, path: str) -> "Config":
        """Load configuration from file."""
        path = Path(path)
        
        if path.suffix == ".json":
            with open(path) as f:
                data = json.load(f)
        elif path.suffix in (".yaml", ".yml"):
            with open(path) as f:
                data = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported format: {path.suffix}")
        
        return cls(
            env=EnvironmentConfig(**data.get("env", {})),
            rewards=RewardConfig(**data.get("rewards", {})),
            curriculum=CurriculumConfig(**data.get("curriculum", {})),
            training=TrainingConfig(**data.get("training", {})),
            experiment_name=data.get("experiment_name", "pursuit_hybrid"),
            log_dir=data.get("log_dir", "logs"),
            model_dir=data.get("model_dir", "models"),
        )


# ============== PRESET CONFIGURATIONS ==============

def get_default_config() -> Config:
    """Standard training configuration."""
    return Config()


def get_fast_config() -> Config:
    """Quick debug/test configuration."""
    cfg = Config()
    cfg.training.total_timesteps = 100_000
    cfg.training.n_envs = 4
    cfg.training.eval_freq = 5_000
    cfg.training.save_freq = 25_000
    cfg.experiment_name = "debug_run"
    return cfg


def get_full_config() -> Config:
    """Full training for best performance."""
    cfg = Config()
    cfg.training.total_timesteps = 3_000_000
    cfg.training.n_envs = 16
    cfg.training.policy_layers = (256, 256)
    cfg.training.value_layers = (256, 256)
    cfg.curriculum.warmup_episodes = 100_000
    cfg.experiment_name = "full_training"
    return cfg


PRESETS = {
    "default": get_default_config,
    "fast": get_fast_config,
    "full": get_full_config,
}


def get_config(preset: str = "default") -> Config:
    """Get configuration by preset name."""
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset: {preset}. Available: {list(PRESETS.keys())}")
    return PRESETS[preset]()
