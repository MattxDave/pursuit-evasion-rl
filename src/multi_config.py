"""
Multi-Agent Configuration
=========================
Configuration for multi-pursuer, multi-evader pursuit-evasion with stations.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List, Tuple
import json
import yaml
from pathlib import Path


@dataclass
class ArenaConfig:
    """Arena/playfield parameters."""
    radius: float = 45.0                    # Tighter arena -> more action
    capture_radius: float = 2.5             # Generous capture reach for pursuers
    station_capture_radius: float = 3.0     # Distance for evader to capture station
    dt: float = 0.1                         # Time step
    max_steps: int = 1500                   # 2.5 minutes max episode


@dataclass
class StationConfig:
    """Station/base configuration."""
    num_stations: int = 2                   # Number of stations to guard
    station_radius: float = 5.0             # Physical size of station (evader goal)
    pursuers_per_station: int = 2           # Pursuers guarding each station
    min_station_separation: float = 25.0    # Min distance between stations
    spawn_radius_from_center: float = 0.5   # Station spawn range (fraction of arena)
    capture_time: float = 2.0              # Time evader must stay at station to capture (2s = 20 steps)


@dataclass
class PursuerConfig:
    """Pursuer agent parameters."""
    max_speed: float = 12.0                 # Faster than evaders to allow interception
    turn_rate_deg: float = 200.0            # More agile for tighter turns
    
    # Radar system
    radar_range: float = 40.0               # Wider detection for earlier reaction
    radar_fov_deg: float = 360.0            # Field of view (360 = omnidirectional)
    
    # Communication
    comm_range: float = 60.0                # Longer comms so all pursuers can coordinate
    
    # Battery
    battery_capacity: float = 100.0
    battery_drain_rate: float = 0.10        # Less drain — pursuers need endurance
    battery_recharge_rate: float = 0.8      # Faster recharge at station
    
    # Behavior thresholds
    return_to_base_battery: float = 15.0    # Lower threshold — stay in fight longer
    intercept_priority_distance: float = 40.0  # React to threats further out

    # Guard patrol behavior
    guard_patrol_radius: float = 3.5        # Patrol radius around station (meters)
    guard_patrol_speed: float = 0.4         # Patrol speed as fraction of max
    max_time_away_from_station: float = 8.0 # Seconds a partner can be away before forcing guard


@dataclass
class EvaderConfig:
    """Smart evader agent parameters."""
    num_evaders: int = 3                    # Number of evaders in flock
    max_speed: float = 11.0                 # SIGNIFICANTLY faster than pursuers (9.0)!
    turn_rate_deg: float = 500.0            # EXTREMELY agile - can out-turn pursuers
    
    # Flock formation
    flock_separation: float = 3.0           # Preferred distance between evaders
    flock_cohesion: float = 0.5             # Cohesion strength
    flock_alignment: float = 0.3            # Alignment strength
    
    # Sensing
    detection_range: float = 40.0           # Early warning of pursuers
    comm_range: float = 30.0                # Better communication range
    
    # Mission
    goal_attraction: float = 0.7            # Weight toward goal vs evasion
    
    # Decoy behavior
    decoy_sacrifice_threshold: float = 0.3  # Probability to become decoy when needed


@dataclass
class FlockConfig:
    """Flock coordination parameters."""
    formation_type: str = "wedge"           # wedge, line, diamond
    formation_spacing: float = 4.0          # Distance between evaders in formation
    leader_rotation: bool = True            # Rotate leader role
    
    # Communication protocol
    threat_broadcast_delay: float = 0.2     # Delay before threat broadcast (seconds)
    decoy_assignment_method: str = "nearest"  # nearest, random, designated


@dataclass
class PursuerRewardConfig:
    """Reward shaping for pursuer RL policy."""
    # Terminal
    capture_evader: float = 30.0            # Strong reward for captures
    all_evaders_escaped: float = -20.0
    evader_reached_goal: float = -25.0      # Harsh penalty for losing a station
    timeout: float = -3.0                   # Lighter timeout penalty (surviving is ok)
    
    # Shaping
    closing_distance: float = 1.0           # Stronger chase incentive
    radar_detection_bonus: float = 0.2
    return_to_station_bonus: float = 2.0
    coordination_bonus: float = 0.5         # Reward for good team coordination
    
    # Penalties
    time_penalty: float = -0.005
    leave_station_unguarded: float = -2.0   # Harsher penalty for leaving station open
    low_battery_penalty: float = -0.1


@dataclass
class EvaderRewardConfig:
    """Reward shaping for evader RL policy."""
    # Terminal
    reach_goal: float = 30.0                # Higher reward for capturing station (was 25)
    captured: float = -15.0                 # Less harsh penalty (was -20)
    timeout: float = 8.0                    # Surviving is good (was 5)
    
    # Shaping  
    progress_to_goal: float = 1.0           # Stronger goal attraction (was 0.5)
    evasion_success: float = 0.8            # Reward for escaping pursuer (was 0.3)
    decoy_sacrifice: float = 8.0            # Reward when sacrifice helps team (was 5)
    
    # Penalties
    time_penalty: float = -0.01             # Less harsh time penalty (was -0.02)
    flock_separation_penalty: float = -0.05 # Less harsh separation penalty (was -0.1)
    detected_penalty: float = -0.02         # Less harsh detection penalty (was -0.05)


@dataclass
class TrainingConfig:
    """Training hyperparameters for multi-agent setup."""
    # Timesteps
    total_timesteps: int = 5_000_000        # More training needed
    n_envs: int = 8
    
    # PPO
    n_steps: int = 2048
    batch_size: int = 512                   # Larger batch for multi-agent
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.02                  # More exploration
    learning_rate: float = 3e-4
    
    # Network
    policy_layers: Tuple[int, ...] = (256, 256, 128)
    value_layers: Tuple[int, ...] = (256, 256, 128)
    
    # Checkpointing
    eval_freq: int = 20_000
    eval_episodes: int = 20
    save_freq: int = 100_000
    
    # Training mode
    train_pursuers: bool = True
    train_evaders: bool = True
    alternating_training: bool = True       # Alternate training pursuer/evader
    alternating_steps: int = 100_000        # Steps before switching


@dataclass
class MultiAgentConfig:
    """Master configuration for multi-agent pursuit-evasion."""
    arena: ArenaConfig = field(default_factory=ArenaConfig)
    station: StationConfig = field(default_factory=StationConfig)
    pursuer: PursuerConfig = field(default_factory=PursuerConfig)
    evader: EvaderConfig = field(default_factory=EvaderConfig)
    flock: FlockConfig = field(default_factory=FlockConfig)
    pursuer_rewards: PursuerRewardConfig = field(default_factory=PursuerRewardConfig)
    evader_rewards: EvaderRewardConfig = field(default_factory=EvaderRewardConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    # Experiment tracking
    experiment_name: str = "multi_agent_pursuit"
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
    def load(cls, path: str) -> "MultiAgentConfig":
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
            arena=ArenaConfig(**data.get("arena", {})),
            station=StationConfig(**data.get("station", {})),
            pursuer=PursuerConfig(**data.get("pursuer", {})),
            evader=EvaderConfig(**data.get("evader", {})),
            flock=FlockConfig(**data.get("flock", {})),
            pursuer_rewards=PursuerRewardConfig(**data.get("pursuer_rewards", {})),
            evader_rewards=EvaderRewardConfig(**data.get("evader_rewards", {})),
            training=TrainingConfig(**data.get("training", {})),
            experiment_name=data.get("experiment_name", "multi_agent_pursuit"),
            log_dir=data.get("log_dir", "logs"),
            model_dir=data.get("model_dir", "models"),
        )


def get_default_multi_config() -> MultiAgentConfig:
    """Standard multi-agent configuration."""
    return MultiAgentConfig()


def get_small_multi_config() -> MultiAgentConfig:
    """Smaller scenario for testing."""
    cfg = MultiAgentConfig()
    cfg.arena.radius = 30.0
    cfg.station.num_stations = 1
    cfg.evader.num_evaders = 2
    cfg.training.total_timesteps = 500_000
    return cfg
