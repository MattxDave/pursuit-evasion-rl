"""
Reward Calculation
==================
Modular reward computation for the pursuit-evasion task.
"""

import numpy as np
from typing import Dict, Tuple
from dataclasses import dataclass

from .config import RewardConfig


@dataclass
class StepInfo:
    """Information about a simulation step for reward computation."""
    prev_distance: float
    curr_distance: float
    pursuer_speed: float
    pursuer_max_speed: float
    captured: bool = False
    evader_escaped: bool = False
    evader_reached_goal: bool = False
    timeout: bool = False
    # Battery info
    battery_percent: float = 100.0
    requested_speed: float = 0.0


class RewardCalculator:
    """Computes rewards based on configuration."""
    
    def __init__(self, config: RewardConfig):
        self.config = config
    
    def compute(self, info: StepInfo) -> Tuple[float, Dict[str, float]]:
        """
        Compute total reward and breakdown.
        
        Args:
            info: Step information
        
        Returns:
            (total_reward, reward_breakdown)
        """
        breakdown = {}
        total = 0.0
        
        # Terminal rewards
        if info.captured:
            breakdown["capture"] = self.config.capture
            total += self.config.capture
            return total, breakdown
        
        if info.evader_escaped:
            breakdown["escaped"] = self.config.evader_escaped
            total += self.config.evader_escaped
            return total, breakdown
        
        if info.evader_reached_goal:
            breakdown["goal_reached"] = self.config.evader_reached_goal
            total += self.config.evader_reached_goal
            return total, breakdown
        
        if info.timeout:
            breakdown["timeout"] = self.config.timeout
            total += self.config.timeout
            return total, breakdown
        
        # Shaping rewards (non-terminal steps)
        
        # Distance closing reward
        distance_delta = info.prev_distance - info.curr_distance
        closing_reward = self.config.closing_distance * distance_delta
        breakdown["closing"] = closing_reward
        total += closing_reward
        
        # Time penalty
        breakdown["time"] = self.config.time_penalty
        total += self.config.time_penalty
        
        # Low speed penalty (optional)
        if self.config.low_speed_penalty != 0 and info.pursuer_max_speed > 0:
            speed_ratio = info.pursuer_speed / info.pursuer_max_speed
            if speed_ratio < 0.5:  # Penalty for going less than half speed
                penalty = self.config.low_speed_penalty * (0.5 - speed_ratio)
                breakdown["low_speed"] = penalty
                total += penalty
        
        # Battery efficiency bonus (reward for closing distance without wasting battery)
        if self.config.battery_efficiency_bonus != 0 and distance_delta > 0:
            # Efficiency = how much distance closed per unit of speed used
            efficiency = distance_delta / max(0.1, info.pursuer_speed / info.pursuer_max_speed)
            bonus = self.config.battery_efficiency_bonus * efficiency
            breakdown["battery_efficiency"] = bonus
            total += bonus
        
        # Low battery penalty
        if self.config.low_battery_penalty != 0 and info.battery_percent < 20.0:
            penalty = self.config.low_battery_penalty * (1.0 - info.battery_percent / 20.0)
            breakdown["low_battery"] = penalty
            total += penalty
        
        return total, breakdown


def compute_simple_reward(
    prev_dist: float,
    curr_dist: float,
    captured: bool,
    escaped: bool,
    timeout: bool,
) -> float:
    """Simple reward function for quick testing."""
    if captured:
        return 10.0
    if escaped:
        return -10.0
    if timeout:
        return -5.0
    
    # Distance closing
    return (prev_dist - curr_dist) * 1.0
