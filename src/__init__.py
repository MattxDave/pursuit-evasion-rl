"""
Pursuit-Evasion Hybrid RL - Source Package
==========================================
Includes both single-agent and multi-agent pursuit-evasion environments.
"""
# Single-agent (original)
from .environment import PursuitEnv
from .agents import Pursuer, Evader
from .geometry import lead_intercept, pure_pursuit, normalize_angle
from .rewards import RewardCalculator
from .config import Config

# Multi-agent (new)
from .multi_config import MultiAgentConfig
from .multi_agents import (
    Station, 
    SmartPursuer, 
    SmartEvader, 
    FlockController,
    PursuerState,
    EvaderRole,
)
from .multi_environment import (
    MultiAgentPursuitEnv,
    PursuerTrainingEnv,
    EvaderTrainingEnv,
)

__all__ = [
    # Single-agent
    "PursuitEnv",
    "Pursuer", 
    "Evader",
    "lead_intercept",
    "pure_pursuit", 
    "normalize_angle",
    "RewardCalculator",
    "Config",
    # Multi-agent
    "MultiAgentConfig",
    "Station",
    "SmartPursuer",
    "SmartEvader",
    "FlockController",
    "PursuerState",
    "EvaderRole",
    "MultiAgentPursuitEnv",
    "PursuerTrainingEnv",
    "EvaderTrainingEnv",
]
