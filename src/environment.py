"""
Pursuit-Evasion Gymnasium Environment
=====================================
Clean, modular environment for training hybrid pursuit agents.
"""

import math
import numpy as np
from typing import Dict, Any, Tuple, Optional

import gymnasium as gym
from gymnasium import spaces

from .config import Config, EnvironmentConfig, RewardConfig, CurriculumConfig
from .agents import Pursuer, Evader
from .rewards import RewardCalculator, StepInfo
from .geometry import random_point_in_circle, unit_vector


class PursuitEnv(gym.Env):
    """
    Hybrid Pursuit-Evasion Environment.
    
    The pursuer uses classical lead-intercept for heading computation,
    while RL controls:
    - Throttle (speed as fraction of max)
    - Heading nudge (small angular adjustment)
    
    Features battery system where higher speeds drain battery faster.
    
    Observation Space (9D):
        [rel_x, rel_y, evader_vx, evader_vy, dist_to_goal, goal_dir_x, goal_dir_y, speed_ratio, battery_percent]
    
    Action Space (2D):
        [throttle, heading_nudge]
        - throttle: [0, 1] -> mapped to [throttle_min, throttle_max] * max_speed
        - heading_nudge: [-1, 1] -> mapped to [-nudge_max, +nudge_max] radians
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(
        self,
        config: Optional[Config] = None,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        
        # Load config
        self.config = config or Config()
        self.env_cfg = self.config.env
        self.reward_cfg = self.config.rewards
        self.curriculum_cfg = self.config.curriculum
        self.evader_noise_cfg = self.config.evader_noise
        self.ignore_battery_limits = self.env_cfg.ignore_battery_limits

        # Random state
        self.rng = np.random.default_rng()
        
        # Create agents
        self.pursuer = Pursuer(
            max_speed=self.env_cfg.pursuer_max_speed,
            turn_rate_limit=math.radians(self.env_cfg.turn_rate_deg),
            battery_capacity=self.env_cfg.battery_capacity,
            battery_drain_rate=self.env_cfg.battery_drain_rate,
            battery_min_speed_factor=self.env_cfg.battery_min_speed_factor,
            ignore_battery_limits=self.ignore_battery_limits,
        )
        self.evader = Evader(
            heading_noise_std=self.evader_noise_cfg.heading_noise_std,
            noise_prob=self.evader_noise_cfg.noise_prob,
            rng=self.rng,
        )
        
        # Reward calculator
        self.reward_calc = RewardCalculator(self.reward_cfg)
        
        # Curriculum state
        self.episode_count = 0
        
        # Convert heading nudge to radians
        self.max_nudge_rad = math.radians(self.env_cfg.heading_nudge_deg)
        
        # ===== Define spaces =====
        
        # Observation bounds
        R = self.env_cfg.radius
        max_speed = max(self.env_cfg.evader_max_speed, self.env_cfg.pursuer_max_speed)
        
        obs_high = np.array([
            2*R, 2*R,           # relative position (can be up to 2R apart)
            max_speed, max_speed,  # evader velocity
            2*R,                # distance to goal
            1.0, 1.0,           # goal direction (unit vector)
            1.0,                # speed ratio
            1.0,                # battery percent (normalized 0-1)
        ], dtype=np.float32)
        
        self.observation_space = spaces.Box(
            low=-obs_high,
            high=obs_high,
            dtype=np.float32,
        )
        
        # Action space: [throttle, heading_nudge]
        self.action_space = spaces.Box(
            low=np.array([0.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        
        # Rendering
        self.render_mode = render_mode
        self._fig = None
        self._ax = None
        
        # Episode state
        self.step_count = 0
        self.prev_distance = 0.0
    
    def _sample_evader_speed(self) -> float:
        """Sample evader speed using curriculum."""
        if not self.curriculum_cfg.enabled:
            return float(self.rng.uniform(
                self.env_cfg.evader_min_speed,
                self.env_cfg.evader_max_speed
            ))
        
        # Compute curriculum progress
        progress = min(1.0, self.episode_count / max(1, self.curriculum_cfg.warmup_episodes))
        hard_prob = (
            self.curriculum_cfg.initial_hard_prob + 
            progress * (self.curriculum_cfg.final_hard_prob - self.curriculum_cfg.initial_hard_prob)
        )
        
        if float(self.rng.random()) < hard_prob:
            # Hard (fast) evader
            return float(self.rng.uniform(
                self.curriculum_cfg.hard_speed_min,
                self.curriculum_cfg.hard_speed_max
            ))
        else:
            # Easy (slow) evader
            return float(self.rng.uniform(
                self.curriculum_cfg.easy_speed_min,
                self.curriculum_cfg.easy_speed_max
            ))
    
    def _spawn_agents(self):
        """Spawn agents with valid initial positions."""
        R = self.env_cfg.radius
        min_sep = self.env_cfg.min_separation
        
        for _ in range(100):  # Max attempts
            # Random positions
            p_pos = random_point_in_circle(R * 0.9, self.rng)
            e_pos = random_point_in_circle(R * 0.9, self.rng)
            e_goal = random_point_in_circle(R * 0.95, self.rng)
            
            # Check constraints
            pe_dist = np.linalg.norm(e_pos - p_pos)
            eg_dist = np.linalg.norm(e_goal - e_pos)
            
            if pe_dist >= min_sep and eg_dist >= min_sep:
                return p_pos, e_pos, e_goal
        
        # Fallback: just use random positions
        return (
            random_point_in_circle(R * 0.9, self.rng),
            random_point_in_circle(R * 0.9, self.rng),
            random_point_in_circle(R * 0.95, self.rng),
        )
    
    def _get_observation(self) -> np.ndarray:
        """Compute observation vector."""
        # Relative position
        rel_pos = self.evader.position - self.pursuer.position
        
        # Evader velocity
        e_vel = self.evader.velocity
        
        # Distance to goal
        dist_to_goal = self.evader.distance_to_goal()
        
        # Goal direction (unit vector from evader to goal)
        goal_dir = unit_vector(self.evader.goal - self.evader.position)
        
        # Speed ratio (evader speed / pursuer max speed)
        speed_ratio = self.evader.speed / self.env_cfg.pursuer_max_speed
        
        # Battery level (normalized 0-1)
        battery_norm = self.pursuer.battery / self.pursuer.battery_capacity
        
        return np.array([
            rel_pos[0], rel_pos[1],
            e_vel[0], e_vel[1],
            dist_to_goal,
            goal_dir[0], goal_dir[1],
            speed_ratio,
            battery_norm,
        ], dtype=np.float32)
    
    def _check_termination(self) -> Tuple[bool, bool, Dict[str, Any]]:
        """
        Check for episode termination.
        
        Returns:
            (terminated, truncated, info)
        """
        info = {}
        
        # Distance between agents
        distance = np.linalg.norm(self.evader.position - self.pursuer.position)
        
        # Capture check
        if distance <= self.env_cfg.capture_radius:
            info["outcome"] = "capture"
            info["capture_time"] = self.step_count * self.env_cfg.dt
            return True, False, info
        
        # Evader escaped (left arena)
        if np.linalg.norm(self.evader.position) > self.env_cfg.radius:
            info["outcome"] = "escaped"
            return True, False, info
        
        # Evader reached goal
        if self.evader.distance_to_goal() <= self.env_cfg.capture_radius:
            info["outcome"] = "goal_reached"
            return True, False, info
        
        # Timeout
        if self.step_count >= self.env_cfg.max_steps:
            info["outcome"] = "timeout"
            return False, True, info
        
        return False, False, info
    
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment for new episode."""
        super().reset(seed=seed)
        
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        
        # Spawn agents
        p_pos, e_pos, e_goal = self._spawn_agents()
        
        # Sample evader speed
        evader_speed = self._sample_evader_speed()
        
        # Reset agents
        initial_heading = unit_vector(e_pos - p_pos)  # Face evader
        self.pursuer.reset(p_pos, initial_heading)
        self.evader.reset(
            e_pos,
            e_goal,
            evader_speed,
            rng=self.rng,
            heading_noise_std=self.evader_noise_cfg.heading_noise_std,
            noise_prob=self.evader_noise_cfg.noise_prob,
        )
        
        # Reset episode state
        self.step_count = 0
        self.prev_distance = float(np.linalg.norm(self.evader.position - self.pursuer.position))
        self.episode_count += 1
        
        obs = self._get_observation()
        info = {
            "evader_speed": evader_speed,
            "initial_distance": self.prev_distance,
        }
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one environment step."""
        self.step_count += 1
        
        # Parse action
        action = np.asarray(action, dtype=np.float32).flatten()
        raw_throttle = float(np.clip(action[0], 0.0, 1.0))
        raw_nudge = float(np.clip(action[1], -1.0, 1.0)) if len(action) > 1 else 0.0
        
        # Map throttle to actual range
        throttle = (
            self.env_cfg.throttle_min + 
            raw_throttle * (self.env_cfg.throttle_max - self.env_cfg.throttle_min)
        )
        
        # Map nudge to radians
        heading_nudge = raw_nudge * self.max_nudge_rad
        
        # Step agents
        self.pursuer.step(
            self.evader.position,
            self.evader.velocity,
            throttle,
            heading_nudge,
            self.env_cfg.dt,
        )
        self.evader.step(self.env_cfg.dt)
        
        # Compute distance
        curr_distance = float(np.linalg.norm(self.evader.position - self.pursuer.position))
        
        # Check termination
        terminated, truncated, info = self._check_termination()
        
        # Compute reward
        step_info = StepInfo(
            prev_distance=self.prev_distance,
            curr_distance=curr_distance,
            pursuer_speed=self.pursuer.actual_speed,
            pursuer_max_speed=self.env_cfg.pursuer_max_speed,
            captured=info.get("outcome") == "capture",
            evader_escaped=info.get("outcome") == "escaped",
            evader_reached_goal=info.get("outcome") == "goal_reached",
            timeout=info.get("outcome") == "timeout",
            battery_percent=self.pursuer.battery_percent,
            requested_speed=self.pursuer.requested_speed,
        )
        
        reward, reward_breakdown = self.reward_calc.compute(step_info)
        info["reward_breakdown"] = reward_breakdown
        info["distance"] = curr_distance
        info["battery_percent"] = self.pursuer.battery_percent
        info["pursuer_speed"] = self.pursuer.actual_speed
        info["evader_speed"] = self.evader.speed
        
        # Update state
        self.prev_distance = curr_distance
        
        # Get observation
        obs = self._get_observation()
        
        # Render if needed
        if self.render_mode == "human":
            self.render()
        
        return obs, reward, terminated, truncated, info
    
    def render(self):
        """Render current state."""
        if self.render_mode is None:
            return None
        
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        
        if self._fig is None:
            plt.ion()
            self._fig, self._ax = plt.subplots(figsize=(8, 8))
        
        self._ax.clear()
        
        R = self.env_cfg.radius
        self._ax.set_xlim(-R - 2, R + 2)
        self._ax.set_ylim(-R - 2, R + 2)
        self._ax.set_aspect("equal")
        self._ax.grid(True, alpha=0.3)
        
        # Arena boundary
        arena = patches.Circle((0, 0), R, fill=False, linewidth=2, color="navy", linestyle="--")
        self._ax.add_patch(arena)
        
        # Goal
        self._ax.plot(*self.evader.goal, marker="*", markersize=15, color="gold", markeredgecolor="orange")
        
        # Agents
        self._ax.plot(*self.pursuer.position, "o", markersize=12, color="blue", label="Pursuer")
        self._ax.plot(*self.evader.position, "o", markersize=10, color="red", label="Evader")
        
        # Info
        dist = np.linalg.norm(self.evader.position - self.pursuer.position)
        self._ax.set_title(f"Step: {self.step_count} | Distance: {dist:.2f}m | Evader Speed: {self.evader.speed:.1f}m/s")
        self._ax.legend(loc="upper right")
        
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()
        
        if self.render_mode == "rgb_array":
            # Convert to RGB array
            self._fig.canvas.draw()
            data = np.frombuffer(self._fig.canvas.tostring_rgb(), dtype=np.uint8)
            data = data.reshape(self._fig.canvas.get_width_height()[::-1] + (3,))
            return data
        
        return None
    
    def close(self):
        """Clean up resources."""
        if self._fig is not None:
            import matplotlib.pyplot as plt
            plt.close(self._fig)
            self._fig = None
            self._ax = None
