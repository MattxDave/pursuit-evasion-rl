"""
Geometry & Classical Control Algorithms
=======================================
Pure pursuit, lead intercept, and utility functions.
"""

import math
import numpy as np
from typing import Tuple

EPS = 1e-8


def normalize_angle(angle: float) -> float:
    """Normalize angle to [-π, π]."""
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle


def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    """Signed angle from v1 to v2."""
    return math.atan2(
        v1[0] * v2[1] - v1[1] * v2[0],
        v1[0] * v2[0] + v1[1] * v2[1]
    )


def rotate_vector(v: np.ndarray, angle: float) -> np.ndarray:
    """Rotate 2D vector by angle (radians)."""
    c, s = math.cos(angle), math.sin(angle)
    return np.array([
        c * v[0] - s * v[1],
        s * v[0] + c * v[1]
    ], dtype=np.float32)


def unit_vector(v: np.ndarray) -> np.ndarray:
    """Return unit vector in direction of v."""
    norm = np.linalg.norm(v)
    if norm < EPS:
        return np.array([1.0, 0.0], dtype=np.float32)
    return (v / norm).astype(np.float32)


def pure_pursuit(pursuer_pos: np.ndarray, target_pos: np.ndarray) -> np.ndarray:
    """
    Pure pursuit: head directly toward target.
    
    Args:
        pursuer_pos: Current pursuer position [x, y]
        target_pos: Target position [x, y]
    
    Returns:
        Unit heading vector toward target
    """
    direction = target_pos - pursuer_pos
    return unit_vector(direction)


def lead_intercept(
    pursuer_pos: np.ndarray,
    target_pos: np.ndarray, 
    target_vel: np.ndarray,
    pursuer_speed: float,
) -> Tuple[np.ndarray, float, bool]:
    """
    Lead intercept: compute heading to intercept moving target.
    
    Solves the intercept equation:
        |target_pos + target_vel * t - pursuer_pos| = pursuer_speed * t
    
    This is a quadratic in t: at² + bt + c = 0 where:
        a = |v_t|² - v_p²
        b = 2 * r · v_t  
        c = |r|²
        r = target_pos - pursuer_pos
    
    Args:
        pursuer_pos: Pursuer position [x, y]
        target_pos: Target position [x, y]
        target_vel: Target velocity [vx, vy]
        pursuer_speed: Pursuer speed (scalar)
    
    Returns:
        (heading, intercept_time, feasible)
        - heading: Unit vector for optimal intercept
        - intercept_time: Time to intercept (inf if impossible)  
        - feasible: True if intercept is geometrically possible
    """
    r = target_pos - pursuer_pos
    
    # Handle edge cases
    if pursuer_speed < EPS:
        return pure_pursuit(pursuer_pos, target_pos), float("inf"), False
    
    target_speed_sq = float(np.dot(target_vel, target_vel))
    pursuer_speed_sq = pursuer_speed ** 2
    
    a = target_speed_sq - pursuer_speed_sq
    b = 2.0 * float(np.dot(r, target_vel))
    c = float(np.dot(r, r))
    
    # Special case: equal speeds (linear equation)
    if abs(a) < EPS:
        if abs(b) < EPS:
            # Target stationary or moving away at same speed
            return pure_pursuit(pursuer_pos, target_pos), float("inf"), False
        t = -c / b
        if t < 0:
            return pure_pursuit(pursuer_pos, target_pos), float("inf"), False
        intercept_point = target_pos + target_vel * t
        heading = unit_vector(intercept_point - pursuer_pos)
        return heading, t, True
    
    # Quadratic formula
    discriminant = b * b - 4.0 * a * c
    
    if discriminant < 0:
        # No real solution - intercept impossible
        return pure_pursuit(pursuer_pos, target_pos), float("inf"), False
    
    sqrt_disc = math.sqrt(discriminant)
    t1 = (-b - sqrt_disc) / (2.0 * a)
    t2 = (-b + sqrt_disc) / (2.0 * a)
    
    # Take smallest positive time
    candidates = [t for t in (t1, t2) if t > EPS]
    if not candidates:
        return pure_pursuit(pursuer_pos, target_pos), float("inf"), False
    
    t_intercept = min(candidates)
    
    # Compute intercept point and heading
    intercept_point = target_pos + target_vel * t_intercept
    heading = unit_vector(intercept_point - pursuer_pos)
    
    return heading, t_intercept, True


def compute_closing_rate(
    pursuer_pos: np.ndarray,
    pursuer_vel: np.ndarray,
    target_pos: np.ndarray,
    target_vel: np.ndarray,
) -> float:
    """
    Compute rate at which distance is changing.
    
    Negative = closing (good for pursuer)
    Positive = opening (bad for pursuer)
    """
    r = target_pos - pursuer_pos
    r_norm = np.linalg.norm(r)
    if r_norm < EPS:
        return 0.0
    
    r_hat = r / r_norm
    relative_vel = target_vel - pursuer_vel
    
    return float(np.dot(relative_vel, r_hat))


def random_point_in_circle(radius: float, rng: np.random.Generator = None) -> np.ndarray:
    """Generate random point uniformly distributed in circle."""
    if rng is None:
        rng = np.random.default_rng()
    
    # Use sqrt for uniform distribution
    r = radius * math.sqrt(float(rng.random()))
    theta = 2.0 * math.pi * float(rng.random())
    
    return np.array([r * math.cos(theta), r * math.sin(theta)], dtype=np.float32)


def random_point_on_circle(radius: float, rng: np.random.Generator = None) -> np.ndarray:
    """Generate random point on circle boundary."""
    if rng is None:
        rng = np.random.default_rng()
    
    theta = 2.0 * math.pi * float(rng.random())
    return np.array([radius * math.cos(theta), radius * math.sin(theta)], dtype=np.float32)
