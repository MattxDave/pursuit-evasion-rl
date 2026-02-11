"""
Agent Definitions
=================
Pursuer and Evader agent classes.
"""

import math
import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass, field

from .geometry import (
    lead_intercept, 
    pure_pursuit, 
    rotate_vector, 
    unit_vector,
    normalize_angle,
)


@dataclass
class Pursuer:
    """
    Hybrid pursuer agent.
    
    Uses classical heading computation (lead intercept) with RL-controlled:
    - Speed/throttle
    - Small heading adjustments (nudge)
    
    Features battery system where higher speeds drain battery faster (quadratic).
    """
    position: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    heading: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0], dtype=np.float32))
    max_speed: float = 3.0
    
    # Battery system
    battery_capacity: float = 100.0
    battery: float = 100.0  # Current battery level
    battery_drain_rate: float = 0.3  # Drain per step at full speed
    battery_min_speed_factor: float = 0.4  # Min speed when depleted
    ignore_battery_limits: bool = False
    
    # Tracking
    actual_speed: float = 0.0  # Actual speed after battery limiting
    requested_speed: float = 0.0  # Speed before battery limiting
    
    # Optional dynamics constraints
    turn_rate_limit: float = 0.0  # rad/s, 0 = unlimited
    
    def compute_classical_heading(
        self,
        target_pos: np.ndarray,
        target_vel: np.ndarray,
        speed: float,
    ) -> Tuple[np.ndarray, float, bool]:
        """
        Compute classical intercept heading.
        
        Returns:
            (heading, intercept_time, is_feasible)
        """
        heading, t_intercept, feasible = lead_intercept(
            self.position, target_pos, target_vel, speed
        )
        
        if not feasible or not np.isfinite(t_intercept):
            # Fall back to pure pursuit
            heading = pure_pursuit(self.position, target_pos)
            return heading, float("inf"), False
        
        return heading, t_intercept, True
    
    def apply_heading_nudge(
        self, 
        base_heading: np.ndarray,
        nudge_angle: float,
    ) -> np.ndarray:
        """Apply small angular adjustment to heading."""
        return rotate_vector(base_heading, nudge_angle)
    
    def apply_turn_rate_limit(
        self,
        desired_heading: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """Apply turn rate constraint to heading change."""
        if self.turn_rate_limit <= 0:
            self.heading = desired_heading
            return desired_heading
        
        # Compute angular difference
        current_angle = math.atan2(self.heading[1], self.heading[0])
        desired_angle = math.atan2(desired_heading[1], desired_heading[0])
        
        delta = normalize_angle(desired_angle - current_angle)
        max_delta = self.turn_rate_limit * dt
        
        # Clamp turn rate
        delta = np.clip(delta, -max_delta, max_delta)
        
        new_angle = current_angle + delta
        self.heading = np.array([math.cos(new_angle), math.sin(new_angle)], dtype=np.float32)
        
        return self.heading
    
    def step(
        self,
        target_pos: np.ndarray,
        target_vel: np.ndarray,
        throttle: float,
        heading_nudge: float,
        dt: float,
    ) -> np.ndarray:
        """
        Execute one simulation step.
        
        Args:
            target_pos: Evader position
            target_vel: Evader velocity  
            throttle: Speed fraction [0, 1]
            heading_nudge: Heading adjustment (radians)
            dt: Time step
        
        Returns:
            New position
        """
        # Compute requested speed
        self.requested_speed = throttle * self.max_speed

        if self.ignore_battery_limits:
            # Run at requested speed; do not cap or drain
            self.actual_speed = min(self.requested_speed, self.max_speed)
        else:
            # Battery affects max achievable speed
            battery_factor = max(
                self.battery_min_speed_factor,
                self.battery / self.battery_capacity
            )
            effective_max_speed = self.max_speed * battery_factor
            self.actual_speed = min(self.requested_speed, effective_max_speed)

            # Drain battery (quadratic - faster = much more drain)
            speed_ratio = self.actual_speed / self.max_speed
            drain = self.battery_drain_rate * (speed_ratio ** 2)
            self.battery = max(0.0, self.battery - drain)
        
        # Compute classical heading
        base_heading, _, _ = self.compute_classical_heading(target_pos, target_vel, self.actual_speed)
        
        # Apply RL nudge
        desired_heading = self.apply_heading_nudge(base_heading, heading_nudge)
        
        # Apply dynamics (turn rate limit)
        actual_heading = self.apply_turn_rate_limit(desired_heading, dt)
        
        # Update position
        velocity = self.actual_speed * actual_heading
        self.position = self.position + velocity * dt
        
        return self.position
    
    def reset(self, position: np.ndarray, heading: Optional[np.ndarray] = None):
        """Reset agent state."""
        self.position = position.astype(np.float32)
        if heading is not None:
            self.heading = unit_vector(heading)
        else:
            self.heading = np.array([1.0, 0.0], dtype=np.float32)
        
        # Reset battery
        self.battery = self.battery_capacity
        self.actual_speed = 0.0
        self.requested_speed = 0.0
    
    @property
    def battery_percent(self) -> float:
        """Battery level as percentage."""
        return (self.battery / self.battery_capacity) * 100.0


@dataclass
class Evader:
    """
    Simple evader agent moving toward goal at constant velocity.
    """
    position: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    goal: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    heading_noise_std: float = 0.0
    noise_prob: float = 0.0
    rng: Optional[np.random.Generator] = field(default_factory=np.random.default_rng)
    
    @property
    def speed(self) -> float:
        """Current speed."""
        return float(np.linalg.norm(self.velocity))
    
    @property
    def heading(self) -> np.ndarray:
        """Current heading direction."""
        return unit_vector(self.velocity)
    
    def step(self, dt: float) -> np.ndarray:
        """Move evader for one time step, optionally adding heading noise."""
        if self.heading_noise_std > 0.0 and self.rng is not None:
            if float(self.rng.random()) < self.noise_prob:
                # Preserve speed while jittering heading
                speed = self.speed
                heading = unit_vector(self.velocity)
                jitter = float(self.rng.normal(0.0, self.heading_noise_std))
                new_heading = rotate_vector(heading, jitter)
                self.velocity = (speed * new_heading).astype(np.float32)
        self.position = self.position + self.velocity * dt
        return self.position
    
    def reset(
        self, 
        position: np.ndarray, 
        goal: np.ndarray, 
        speed: float,
        direction_noise: float = 0.0,
        rng: Optional[np.random.Generator] = None,
        heading_noise_std: Optional[float] = None,
        noise_prob: Optional[float] = None,
    ):
        """
        Reset evader state.
        
        Args:
            position: Starting position
            goal: Goal position
            speed: Movement speed
            direction_noise: Random noise to add to direction (radians std)
            rng: Random number generator
        """
        self.position = position.astype(np.float32)
        self.goal = goal.astype(np.float32)
        
        # Direction toward goal
        direction = unit_vector(goal - position)
        
        # Add noise if specified
        if direction_noise > 0 and rng is not None:
            noise_angle = float(rng.normal(0, direction_noise))
            direction = rotate_vector(direction, noise_angle)
        
        self.velocity = (speed * direction).astype(np.float32)
        self.rng = rng or self.rng
        if heading_noise_std is not None:
            self.heading_noise_std = heading_noise_std
        if noise_prob is not None:
            self.noise_prob = noise_prob
    
    def distance_to_goal(self) -> float:
        """Distance remaining to goal."""
        return float(np.linalg.norm(self.goal - self.position))
