"""
Multi-Agent Pursuit-Evasion Environment
=======================================
Gymnasium environment for multi-pursuer, multi-evader scenarios with stations.
"""

import math
import numpy as np
from collections import deque
from typing import Dict, Any, Tuple, Optional, List

from matplotlib.patches import Wedge

import gymnasium as gym
from gymnasium import spaces

from .multi_config import MultiAgentConfig
from .multi_agents import (
    Station, spawn_stations,
    SmartPursuer, PursuerState,
    SmartEvader, EvaderRole,
    FlockController,
)
from .geometry import random_point_in_circle, unit_vector


class MultiAgentPursuitEnv(gym.Env):
    """
    Multi-Agent Pursuit-Evasion Environment.
    
    Features:
    - Multiple stations with pursuer guards
    - Multiple pursuers with radar and communication
    - Multiple evaders in flock formation with decoy behavior
    - Both pursuers and evaders have RL-controlled actions
    
    This environment supports:
    1. Training pursuers (evaders use classical/heuristic policy)
    2. Training evaders (pursuers use classical/heuristic policy)
    3. Self-play (both use RL policies)
    
    Observation Space (per agent):
        Pursuers: [own_state, nearby_evaders, nearby_pursuers, station_info]
        Evaders: [own_state, nearby_pursuers, nearby_evaders, goal_info]
    
    Action Space (per agent):
        [throttle, heading_adjustment]
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(
        self,
        config: Optional[MultiAgentConfig] = None,
        render_mode: Optional[str] = None,
        train_pursuers: bool = True,
        train_evaders: bool = True,
    ):
        super().__init__()
        
        self.config = config or MultiAgentConfig()
        self.render_mode = render_mode
        self.train_pursuers = train_pursuers
        self.train_evaders = train_evaders
        
        # Extract config sections
        self.arena_cfg = self.config.arena
        self.station_cfg = self.config.station
        self.pursuer_cfg = self.config.pursuer
        self.evader_cfg = self.config.evader
        self.flock_cfg = self.config.flock
        
        # Random state
        self.rng = np.random.default_rng()
        
        # Calculate total agents
        self.num_pursuers = self.station_cfg.num_stations * self.station_cfg.pursuers_per_station
        self.num_evaders = self.evader_cfg.num_evaders
        
        # ===== Define spaces =====
        
        # Pursuer observation: 
        # [px, py, heading_x, heading_y, speed, battery] + 
        # [nearest 3 evaders: rel_x, rel_y, vx, vy] * 3 +
        # [partner: rel_x, rel_y, state] +
        # [station: rel_x, rel_y]
        # Total: 6 + 12 + 3 + 2 = 23
        pursuer_obs_dim = 23
        
        # Evader observation (enhanced for smarter RL):
        # [px, py, heading_x, heading_y, speed] = 5
        # [goal: rel_x, rel_y, dist] = 3
        # [nearest 3 pursuers: rel_x, rel_y, vx, vy, state, dist] * 3 = 18
        # [flock mates: rel_x, rel_y, role] * 2 = 6
        # [flock center: rel_x, rel_y] = 2
        # [role_one_hot: 3] = 3
        # [threat_level, nearest_pursuer_dist, nearest_station_dist] = 3
        # Total: 5 + 3 + 18 + 6 + 2 + 3 + 3 = 40
        evader_obs_dim = 40
        
        R = self.arena_cfg.radius
        max_speed = max(self.pursuer_cfg.max_speed, self.evader_cfg.max_speed)
        
        # Pursuer spaces
        self.pursuer_observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(pursuer_obs_dim,),
            dtype=np.float32,
        )
        
        self.pursuer_action_space = spaces.Box(
            low=np.array([0.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        
        # Evader spaces
        self.evader_observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(evader_obs_dim,),
            dtype=np.float32,
        )
        
        # Evader action: [throttle, heading_adj, decoy_signal]
        # decoy_signal > 0.5 means volunteer to become decoy if threatened
        self.evader_action_space = spaces.Box(
            low=np.array([0.0, -1.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        
        # Combined observation/action spaces for gym interface
        # We'll use Dict spaces for multi-agent
        self.observation_space = spaces.Dict({
            'pursuers': spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.num_pursuers, pursuer_obs_dim),
                dtype=np.float32
            ),
            'evaders': spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.num_evaders, evader_obs_dim),
                dtype=np.float32
            ),
        })
        
        self.action_space = spaces.Dict({
            'pursuers': spaces.Box(
                low=np.tile([0.0, -1.0], (self.num_pursuers, 1)),
                high=np.tile([1.0, 1.0], (self.num_pursuers, 1)),
                dtype=np.float32
            ),
            'evaders': spaces.Box(
                low=np.tile([0.0, -1.0, 0.0], (self.num_evaders, 1)),
                high=np.tile([1.0, 1.0, 1.0], (self.num_evaders, 1)),
                dtype=np.float32
            ),
        })
        
        # Agents and environment objects
        self.stations: List[Station] = []
        self.pursuers: List[SmartPursuer] = []
        self.evaders: List[SmartEvader] = []
        self.flock: Optional[FlockController] = None
        
        # Episode state
        self.step_count = 0
        self.current_time = 0.0
        self.evader_goal: Optional[np.ndarray] = None
        
        # Rendering
        self._fig = None
        self._ax = None
    
    def _create_pursuers(self):
        """Create pursuer agents assigned to stations."""
        self.pursuers = []
        
        pursuer_id = 0
        for station in self.stations:
            station_pursuers = []
            
            for i in range(self.station_cfg.pursuers_per_station):
                # Spawn near station
                offset = np.array([
                    self.station_cfg.station_radius * (1 if i == 0 else -1),
                    0.0
                ], dtype=np.float32)
                
                pursuer = SmartPursuer(
                    pursuer_id=pursuer_id,
                    position=station.position + offset,
                    max_speed=self.pursuer_cfg.max_speed,
                    turn_rate_limit=math.radians(self.pursuer_cfg.turn_rate_deg),
                    radar_range=self.pursuer_cfg.radar_range,
                    radar_fov=math.radians(self.pursuer_cfg.radar_fov_deg),
                    comm_range=self.pursuer_cfg.comm_range,
                    battery_capacity=self.pursuer_cfg.battery_capacity,
                    battery_drain_rate=self.pursuer_cfg.battery_drain_rate,
                    battery_recharge_rate=self.pursuer_cfg.battery_recharge_rate,
                    return_threshold=self.pursuer_cfg.return_to_base_battery,
                )
                
                pursuer.assigned_station = station.station_id
                station.assign_pursuer(pursuer_id)
                station_pursuers.append(pursuer)
                self.pursuers.append(pursuer)
                pursuer_id += 1
            
            # Assign partners
            if len(station_pursuers) >= 2:
                station_pursuers[0].partner_id = station_pursuers[1].pursuer_id
                station_pursuers[1].partner_id = station_pursuers[0].pursuer_id
    
    def _create_evaders(self):
        """Create evader flock. Goal = Capture stations."""
        self.evaders = []
        
        # Spawn evaders on opposite side of arena from stations
        station_center = np.mean([s.position for s in self.stations], axis=0)
        spawn_direction = -unit_vector(station_center) if np.linalg.norm(station_center) > 0.1 else np.array([1.0, 0.0])
        
        # Spawn near edge of arena, opposite from stations - gives evaders running room
        spawn_center = spawn_direction * self.arena_cfg.radius * 0.8  # Far from stations
        
        for i in range(self.num_evaders):
            # Spread evaders in tighter formation
            offset = np.array([
                self.flock_cfg.formation_spacing * 0.5 * (i % 2 - 0.5),
                -self.flock_cfg.formation_spacing * 0.5 * (i // 2)
            ], dtype=np.float32)
            
            spawn_pos = (spawn_center + offset).astype(np.float32)
            # Ensure spawn is inside arena
            dist = np.linalg.norm(spawn_pos)
            if dist > self.arena_cfg.radius * 0.95:
                spawn_pos = spawn_pos * (self.arena_cfg.radius * 0.9 / dist)
            
            # GOAL = Nearest station (evaders try to capture stations)
            nearest_station = min(self.stations, key=lambda s: np.linalg.norm(s.position - spawn_pos))
            goal_pos = nearest_station.position.copy()
            
            evader = SmartEvader(
                evader_id=i,
                position=spawn_pos,
                goal=goal_pos,
                max_speed=self.evader_cfg.max_speed,
                turn_rate_limit=math.radians(self.evader_cfg.turn_rate_deg),
                detection_range=self.evader_cfg.detection_range,
                comm_range=self.evader_cfg.comm_range,
            )
            
            # ALL evaders are equal attackers - no fixed leader
            evader.role = EvaderRole.LEADER  # All are attackers
            evader.flock_position_idx = i
            self.evaders.append(evader)
        
        # Create flock controller (still useful for coordination)
        self.flock = FlockController(
            evaders=self.evaders,
            formation_type=self.flock_cfg.formation_type,
            formation_spacing=self.flock_cfg.formation_spacing,
        )
    
    def _get_pursuer_observation(self, pursuer: SmartPursuer) -> np.ndarray:
        """Compute observation for a single pursuer."""
        obs = []
        
        # Own state (6)
        obs.extend(pursuer.position / self.arena_cfg.radius)  # Normalized position
        obs.extend(pursuer.heading)
        obs.append(pursuer.actual_speed / pursuer.max_speed)
        obs.append(pursuer.battery / pursuer.battery_capacity)
        
        # Nearest 3 evaders (12)
        active_evaders = [e for e in self.evaders if not e.is_captured]
        evader_dists = [(e, np.linalg.norm(e.position - pursuer.position)) for e in active_evaders]
        evader_dists.sort(key=lambda x: x[1])
        
        for i in range(3):
            if i < len(evader_dists):
                evader = evader_dists[i][0]
                rel_pos = (evader.position - pursuer.position) / self.arena_cfg.radius
                rel_vel = evader.velocity / self.evader_cfg.max_speed
                obs.extend(rel_pos)
                obs.extend(rel_vel)
            else:
                obs.extend([0.0, 0.0, 0.0, 0.0])  # No evader
        
        # Partner info (3)
        if pursuer.partner_id is not None:
            partner = self.pursuers[pursuer.partner_id]
            rel_pos = (partner.position - pursuer.position) / self.arena_cfg.radius
            obs.extend(rel_pos)
            obs.append(partner.state.value / 4.0)  # Normalized state
        else:
            obs.extend([0.0, 0.0, 0.0])
        
        # Station info (2)
        if pursuer.assigned_station is not None:
            station = self.stations[pursuer.assigned_station]
            rel_pos = (station.position - pursuer.position) / self.arena_cfg.radius
            obs.extend(rel_pos)
        else:
            obs.extend([0.0, 0.0])
        
        return np.array(obs, dtype=np.float32)
    
    def _get_evader_observation(self, evader: SmartEvader) -> np.ndarray:
        """Compute enhanced observation for a single evader (40-dim)."""
        obs = []
        
        # Own state (5)
        obs.extend(evader.position / self.arena_cfg.radius)
        obs.extend(evader.heading)
        obs.append(evader.actual_speed / evader.max_speed)
        
        # Goal info (3)
        rel_goal = (evader.goal - evader.position) / self.arena_cfg.radius
        obs.extend(rel_goal)
        obs.append(evader.distance_to_goal() / self.arena_cfg.radius)
        
        # Nearest 3 pursuers with full info (18)
        pursuer_dists = [(p, np.linalg.norm(p.position - evader.position)) for p in self.pursuers]
        pursuer_dists.sort(key=lambda x: x[1])
        
        for i in range(3):
            if i < len(pursuer_dists):
                p, dist = pursuer_dists[i]
                rel_pos = (p.position - evader.position) / self.arena_cfg.radius
                rel_vel = p.velocity / self.pursuer_cfg.max_speed
                state_norm = p.state.value / 4.0  # 0-1 normalized state
                dist_norm = dist / self.arena_cfg.radius
                obs.extend(rel_pos)       # 2
                obs.extend(rel_vel)       # 2
                obs.append(state_norm)    # 1 (is pursuer intercepting?)
                obs.append(dist_norm)     # 1
            else:
                obs.extend([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])  # No pursuer, max dist
        
        # Flock mates relative info (6) - nearest 2 teammates
        flock_mates = [e for e in self.evaders if e.evader_id != evader.evader_id and not e.is_captured]
        flock_mates.sort(key=lambda e: np.linalg.norm(e.position - evader.position))
        
        for i in range(2):
            if i < len(flock_mates):
                mate = flock_mates[i]
                rel_pos = (mate.position - evader.position) / self.arena_cfg.radius
                role_val = {EvaderRole.LEADER: 0.0, EvaderRole.FOLLOWER: 0.5, EvaderRole.DECOY: 1.0}.get(mate.role, 0.5)
                obs.extend(rel_pos)   # 2
                obs.append(role_val)  # 1
            else:
                obs.extend([0.0, 0.0, 0.5])  # No mate
        
        # Flock center (2)
        if self.flock:
            flock_center = self.flock.get_flock_center()
            rel_center = (flock_center - evader.position) / self.arena_cfg.radius
            obs.extend(rel_center)
        else:
            obs.extend([0.0, 0.0])
        
        # Role one-hot (3)
        role_vec = [0.0, 0.0, 0.0]
        if evader.role == EvaderRole.LEADER:
            role_vec[0] = 1.0
        elif evader.role == EvaderRole.FOLLOWER:
            role_vec[1] = 1.0
        elif evader.role == EvaderRole.DECOY:
            role_vec[2] = 1.0
        obs.extend(role_vec)
        
        # Tactical info (3) - encode chase status into threat_level
        # Threat level: how many pursuers in detection range and how close
        pursuers_in_range = sum(1 for p, d in pursuer_dists if d < self.pursuer_cfg.radar_range)
        threat_level = min(1.0, pursuers_in_range / 2.0)
        # Boost threat if being chased (encodes is_being_chased into existing field)
        if evader.is_being_chased:
            threat_level = max(threat_level, 0.8)  # High threat when locked on
        
        nearest_pursuer_dist = pursuer_dists[0][1] / self.arena_cfg.radius if pursuer_dists else 1.0
        
        # Nearest station distance (for tactical awareness)
        station_dists = [np.linalg.norm(s.position - evader.position) for s in self.stations if s.is_active]
        nearest_station_dist = min(station_dists) / self.arena_cfg.radius if station_dists else 1.0
        
        obs.append(threat_level)
        obs.append(nearest_pursuer_dist)
        obs.append(nearest_station_dist)
        
        return np.array(obs, dtype=np.float32)
    
    def _get_observations(self) -> Dict[str, np.ndarray]:
        """Get observations for all agents."""
        pursuer_obs = np.array([
            self._get_pursuer_observation(p) for p in self.pursuers
        ], dtype=np.float32)
        
        evader_obs = np.array([
            self._get_evader_observation(e) for e in self.evaders
        ], dtype=np.float32)
        
        return {
            'pursuers': pursuer_obs,
            'evaders': evader_obs,
        }
    
    def _compute_heuristic_pursuer_action(self, pursuer: SmartPursuer) -> np.ndarray:
        """Compute heuristic action for pursuer (when not training pursuers).
        
        Simple: ALWAYS full speed chase. The step() function handles targeting.
        """
        # Full speed, no heading nudge - let step() compute intercept heading
        return np.array([1.0, 0.0], dtype=np.float32)
    
    def _compute_heuristic_evader_action(self, evader: SmartEvader) -> np.ndarray:
        """Compute heuristic action for evader (when not training evaders).
        
        GOAL = Capture stations. Evader heads toward nearest active station,
        while evading pursuers if they get too close.
        """
        # Find nearest ACTIVE station (not yet captured)
        active_stations = [s for s in self.stations if s.is_active]
        if not active_stations:
            # All stations captured - just flee
            return np.array([1.0, 0.0], dtype=np.float32)
        
        nearest_station = min(active_stations, 
                            key=lambda s: np.linalg.norm(s.position - evader.position))
        
        # Update evader's goal to nearest active station
        evader.goal = nearest_station.position.copy()
        
        # Check for nearby pursuers
        min_pursuer_dist = float('inf')
        closest_pursuer = None
        for p in self.pursuers:
            dist = np.linalg.norm(p.position - evader.position)
            if dist < min_pursuer_dist:
                min_pursuer_dist = dist
                closest_pursuer = p
        
        # If pursuer is close, add strong evasion
        if min_pursuer_dist < 12:
            # Evade perpendicular to pursuer direction
            nudge = 0.5 * (1 if self.rng.random() > 0.5 else -1)
            speed = 1.0  # Full speed when evading
        elif min_pursuer_dist < 20:
            # Moderate evasion
            nudge = 0.2 * (1 if self.rng.random() > 0.5 else -1)
            speed = 0.9
        else:
            # No threat - head straight for station
            nudge = 0.0
            speed = 0.8
        
        # Heuristic doesn't use decoy signal
        return np.array([speed, nudge, 0.0], dtype=np.float32)
    
    def _update_pursuer_states(self):
        """Update pursuer state machines based on situation."""
        for pursuer in self.pursuers:
            action_type, target_id = pursuer.decide_action(
                self.stations, self.evaders, self.pursuers, self.current_time
            )
            
            if action_type == 'guard':
                pursuer.state = PursuerState.GUARDING
                pursuer.target_evader_id = None
            elif action_type == 'intercept':
                pursuer.state = PursuerState.INTERCEPTING
                pursuer.target_evader_id = target_id
                
                # MARK THE EVADER AS BEING CHASED
                target_evader = next((e for e in self.evaders if e.evader_id == target_id), None)
                if target_evader:
                    target_evader.is_being_chased = True
                
                # Broadcast threat to partner
                if target_id in pursuer.radar_contacts:
                    pursuer.broadcast_threat(
                        pursuer.radar_contacts[target_id],
                        self.pursuers
                    )
            elif action_type == 'return':
                pursuer.state = PursuerState.RETURNING
                pursuer.target_evader_id = None
            elif action_type == 'recharge':
                pursuer.state = PursuerState.RECHARGING
                pursuer.target_evader_id = None
    
    def _check_captures(self) -> List[int]:
        """Check for captures and return list of captured evader IDs."""
        captured = []
        
        for evader in self.evaders:
            if evader.is_captured:
                continue
            
            for pursuer in self.pursuers:
                dist = np.linalg.norm(evader.position - pursuer.position)
                if dist <= self.arena_cfg.capture_radius:
                    evader.is_captured = True
                    evader.role = EvaderRole.CAPTURED
                    pursuer.intercept_count += 1
                    captured.append(evader.evader_id)
                    
                    # Pursuer should return to base or find next target
                    pursuer.target_evader_id = None
                    break
        
        return captured
    
    def _check_station_captures(self) -> List[int]:
        """Check if evaders are capturing stations. Returns list of captured station IDs."""
        captured_stations = []
        
        for station in self.stations:
            if not station.is_active:
                continue
                
            # Check if any evader is in the station's capture zone
            evader_in_zone = None
            for evader in self.evaders:
                if evader.is_captured:
                    continue
                if station.is_in_capture_zone(evader.position):
                    evader_in_zone = evader.evader_id
                    break
            
            # Update station capture progress
            station.update_capture(
                evader_present=(evader_in_zone is not None),
                evader_id=evader_in_zone,
                dt=self.arena_cfg.dt,
                capture_time=self.station_cfg.capture_time
            )
            
            if not station.is_active:
                captured_stations.append(station.station_id)
                # Mark the capturing evader as successful
                for evader in self.evaders:
                    if evader.evader_id == station.capturing_evader_id:
                        evader.is_at_goal = True
                        evader.role = EvaderRole.ESCAPED
        
        return captured_stations
    
    def _check_goals(self) -> List[int]:
        """Legacy method - now handled by _check_station_captures."""
        return []
    
    def _check_boundaries(self):
        """Keep agents within arena."""
        R = self.arena_cfg.radius
        
        for pursuer in self.pursuers:
            dist = np.linalg.norm(pursuer.position)
            if dist > R:
                pursuer.position = pursuer.position * (R / dist)
        
        for evader in self.evaders:
            if evader.is_captured:
                continue
            dist = np.linalg.norm(evader.position)
            if dist > R:
                evader.position = evader.position * (R / dist)
    
    def _compute_rewards(
        self,
        captured: List[int],
        reached_goal: List[int],
        terminated: bool,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute rewards for all agents with station capture/defense mechanics."""
        pursuer_rewards = np.zeros(self.num_pursuers, dtype=np.float32)
        evader_rewards = np.zeros(self.num_evaders, dtype=np.float32)
        
        p_cfg = self.config.pursuer_rewards
        e_cfg = self.config.evader_rewards
        
        # ===== EVADER CAPTURE REWARDS =====
        for eid in captured:
            # Reward all pursuers for team capture
            pursuer_rewards += p_cfg.capture_evader / self.num_pursuers
            # Penalize captured evader
            evader_rewards[eid] = e_cfg.captured
        
        # ===== STATION CAPTURE REWARDS =====
        for sid in reached_goal:  # reached_goal now contains station IDs that were captured
            # Big reward for evaders capturing station
            for i, evader in enumerate(self.evaders):
                if not evader.is_captured:
                    evader_rewards[i] += e_cfg.reach_goal / max(1, self.num_evaders - len(captured))
            # Big penalty for pursuers losing station
            pursuer_rewards -= p_cfg.evader_reached_goal / self.num_pursuers
        
        # ===== PURSUER SHAPING REWARDS =====
        for i, pursuer in enumerate(self.pursuers):
            # Time penalty
            pursuer_rewards[i] += p_cfg.time_penalty
            
            # Get pursuer's assigned station
            my_station = self.stations[pursuer.assigned_station] if pursuer.assigned_station is not None else None
            
            # Intercept shaping - reward closing distance to target
            if pursuer.state == PursuerState.INTERCEPTING and pursuer.target_evader_id is not None:
                target = next((e for e in self.evaders if e.evader_id == pursuer.target_evader_id), None)
                if target and not target.is_captured:
                    dist = np.linalg.norm(target.position - pursuer.position)
                    pursuer_rewards[i] += p_cfg.closing_distance * (1.0 / (dist + 1.0))
            
            # Station defense shaping - penalize if evader is capturing our station
            if my_station and my_station.is_active and my_station.capture_progress > 0:
                # Station is being captured! Strong penalty
                pursuer_rewards[i] -= 0.5 * my_station.capture_progress
            
            # Reward for staying near assigned station when guarding
            if pursuer.state == PursuerState.GUARDING and my_station:
                dist_to_station = np.linalg.norm(pursuer.position - my_station.position)
                if dist_to_station < 10:
                    pursuer_rewards[i] += 0.01  # Small reward for good positioning
            
            # Battery penalty
            if pursuer.battery_percent < 20:
                pursuer_rewards[i] += p_cfg.low_battery_penalty
        
        # ===== EVADER SHAPING REWARDS =====
        for i, evader in enumerate(self.evaders):
            if evader.is_captured:
                continue
            
            # Time penalty
            evader_rewards[i] += e_cfg.time_penalty
            
            # Progress toward nearest active station
            active_stations = [s for s in self.stations if s.is_active]
            if active_stations:
                nearest = min(active_stations, key=lambda s: np.linalg.norm(s.position - evader.position))
                dist_to_station = np.linalg.norm(nearest.position - evader.position)
                
                # Reward for getting closer to station
                progress = 1.0 - (dist_to_station / (2 * self.arena_cfg.radius))
                evader_rewards[i] += e_cfg.progress_to_goal * progress * 0.1
                
                # Bonus for being in capture zone
                if dist_to_station < self.station_cfg.station_radius:
                    evader_rewards[i] += 0.2  # Reward for staying in capture zone
                    
                    # Extra reward for capture progress
                    evader_rewards[i] += 0.3 * nearest.capture_progress
            
            # Detection penalty (but less severe - evaders may need to take risks)
            if evader.detected_pursuers:
                evader_rewards[i] += e_cfg.detected_penalty * len(evader.detected_pursuers) * 0.5
            
            # ===== SMART EVADER REWARDS =====
            # Evasion success: reward for being detected but maintaining distance
            pursuer_dists = [np.linalg.norm(p.position - evader.position) for p in self.pursuers]
            min_pursuer_dist = min(pursuer_dists) if pursuer_dists else 100.0
            
            # Reward for successful evasion maneuvers (close call but not caught)
            if 5 < min_pursuer_dist < 15:
                evader_rewards[i] += e_cfg.evasion_success * (15 - min_pursuer_dist) / 10.0
            
            # Penalty for getting cornered (close to edge AND close to pursuer)
            dist_from_center = np.linalg.norm(evader.position)
            if dist_from_center > self.arena_cfg.radius * 0.85 and min_pursuer_dist < 20:
                evader_rewards[i] -= 0.1  # Cornered penalty
            
            # Decoy sacrifice reward: if I'm a decoy and other evaders are progressing
            if evader.role == EvaderRole.DECOY:
                # Check if other evaders are making progress (closer to stations)
                other_evaders = [e for e in self.evaders if e.evader_id != evader.evader_id and not e.is_captured]
                if other_evaders and active_stations:
                    team_progress = sum(
                        1.0 - np.linalg.norm(e.position - nearest.position) / self.arena_cfg.radius
                        for e in other_evaders
                    ) / len(other_evaders)
                    # Reward decoy for enabling team progress
                    evader_rewards[i] += e_cfg.decoy_sacrifice * team_progress * 0.1
            
            # Flock separation penalty: don't stray too far from flock (unless decoy)
            if evader.role != EvaderRole.DECOY and self.flock:
                flock_center = self.flock.get_flock_center()
                dist_to_flock = np.linalg.norm(evader.position - flock_center)
                if dist_to_flock > 15:
                    evader_rewards[i] += e_cfg.flock_separation_penalty * (dist_to_flock - 15) / 20.0
        
        return pursuer_rewards, evader_rewards
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset the environment."""
        super().reset(seed=seed)
        
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        
        # Reset counters
        self.step_count = 0
        self.current_time = 0.0
        
        # Create stations
        self.stations = spawn_stations(
            num_stations=self.station_cfg.num_stations,
            arena_radius=self.arena_cfg.radius,
            station_radius=self.station_cfg.station_radius,
            min_separation=self.station_cfg.min_station_separation,
            spawn_fraction=self.station_cfg.spawn_radius_from_center,
            rng=self.rng,
        )
        
        # Reset station capture states
        for station in self.stations:
            station.reset()
        
        # GOAL = Stations themselves (evaders try to capture stations)
        # evader_goal is just for compatibility - actual goal is nearest station
        station_center = np.mean([s.position for s in self.stations], axis=0)
        self.evader_goal = station_center.astype(np.float32)
        
        # Create agents
        self._create_pursuers()
        self._create_evaders()
        
        obs = self._get_observations()
        
        info = {
            'num_stations': len(self.stations),
            'num_pursuers': self.num_pursuers,
            'num_evaders': self.num_evaders,
            'evader_goal': self.evader_goal.copy(),
            'station_positions': [s.position.copy() for s in self.stations],
        }
        
        return obs, info
    
    def step(
        self,
        action: Dict[str, np.ndarray],
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, float], bool, bool, Dict[str, Any]]:
        """
        Execute one environment step.
        
        Args:
            action: Dict with 'pursuers' and 'evaders' action arrays
        
        Returns:
            (observations, rewards, terminated, truncated, info)
        """
        self.step_count += 1
        self.current_time += self.arena_cfg.dt
        
        # Get actions (use heuristics if not training that agent type)
        pursuer_actions = action.get('pursuers', None)
        evader_actions = action.get('evaders', None)
        
        # Reset chase flags before pursuer decisions (will be set by decide_action)
        for evader in self.evaders:
            evader.is_being_chased = False
        
        # Update pursuer state machines (this sets is_being_chased on targeted evaders)
        self._update_pursuer_states()
        
        # Scan for threats (pursuers detect evaders)
        for pursuer in self.pursuers:
            pursuer.scan_radar(self.evaders, self.current_time)
        
        # Evaders scan for pursuers
        for evader in self.evaders:
            if not evader.is_captured:
                evader.scan_for_pursuers(self.pursuers)
        
        # Evaders share threat info with flock and process messages
        for evader in self.evaders:
            if not evader.is_captured:
                evader.share_threat_info(self.evaders)
        for evader in self.evaders:
            if not evader.is_captured:
                evader.process_flock_messages()
        
        # Execute pursuer actions
        for i, pursuer in enumerate(self.pursuers):
            if self.train_pursuers and pursuer_actions is not None:
                act = pursuer_actions[i]
            else:
                act = self._compute_heuristic_pursuer_action(pursuer)
            
            # Get target and station for step
            target = None
            if pursuer.target_evader_id is not None:
                target = next((e for e in self.evaders if e.evader_id == pursuer.target_evader_id), None)
            
            station = self.stations[pursuer.assigned_station] if pursuer.assigned_station is not None else None
            
            pursuer.step(act, target, station, self.arena_cfg.dt)
        
        # Execute evader actions
        for i, evader in enumerate(self.evaders):
            if evader.is_captured:
                continue
            
            if self.train_evaders and evader_actions is not None:
                act = evader_actions[i]
            else:
                act = self._compute_heuristic_evader_action(evader)
            
            evader.step(act, self.evaders, self.arena_cfg.dt)
        
        # Update flock
        if self.flock:
            self.flock.update(self.arena_cfg.dt)
        
        # Check evader captures (by pursuers)
        captured = self._check_captures()
        
        # Check station captures (by evaders)
        stations_captured = self._check_station_captures()
        reached_goal = stations_captured  # For reward compatibility
        
        # Keep agents in bounds
        self._check_boundaries()
        
        # Check termination
        all_evaders_captured = all(e.is_captured for e in self.evaders)
        any_station_captured = any(not s.is_active for s in self.stations)  # Evader reached a station!
        any_evader_reached_goal = any(e.is_at_goal for e in self.evaders)
        
        # Terminate if:
        # - All evaders captured (pursuers win completely)
        # - ANY evader reaches a station (evaders score a goal!)
        # Episode ends on first goal or when all evaders are caught
        evader_scored = any_station_captured or any_evader_reached_goal
        terminated = all_evaders_captured or evader_scored
        truncated = self.step_count >= self.arena_cfg.max_steps
        
        # Compute rewards
        pursuer_rewards, evader_rewards = self._compute_rewards(captured, reached_goal, terminated)
        
        # Combined reward dict
        rewards = {
            'pursuers': pursuer_rewards,
            'evaders': evader_rewards,
        }
        
        obs = self._get_observations()
        
        info = {
            'step': self.step_count,
            'time': self.current_time,
            'captured': captured,
            'reached_goal': reached_goal,
            'num_active_evaders': self.flock.num_active if self.flock else 0,
            'num_captured': self.flock.num_captured if self.flock else 0,
            'num_at_goal': self.flock.num_at_goal if self.flock else 0,
            'pursuer_states': [p.state.name for p in self.pursuers],
            'evader_roles': [e.role.name for e in self.evaders],
        }
        
        # Return single reward value for compatibility (sum of all)
        total_reward = float(pursuer_rewards.sum() + evader_rewards.sum())
        
        return obs, rewards, terminated, truncated, info
    
    def render(self):
        """Render the environment with clean, professional visualization."""
        if self.render_mode is None:
            return None
        
        import matplotlib.pyplot as plt
        import matplotlib.patheffects as pe
        
        # Clean, professional color palette
        C = {
            'bg': '#1a1a2e',
            'arena_fill': '#16213e',
            'arena_edge': '#4fc3f7',
            'grid': '#2a2a4a',
            'station': '#42a5f5',
            'station_glow': '#90caf9',
            'pursuer': '#e91e63',
            'pursuer_guard': '#4fc3f7',
            'evader': '#66bb6a',
            'evader_decoy': '#ffca28',
            'trail_p': '#e91e6355',
            'trail_e': '#66bb6a55',
            'text': '#e0e0e0',
            'text_dim': '#9e9e9e',
            'hud_bg': '#0d0d1a',
        }
        
        if self._fig is None:
            plt.style.use('dark_background')
            self._fig, self._ax = plt.subplots(figsize=(10, 10))
            self._fig.patch.set_facecolor(C['bg'])
            self._trail_pursuers = [deque(maxlen=60) for _ in range(self.num_pursuers)]
            self._trail_evaders = [deque(maxlen=60) for _ in range(self.num_evaders)]
        
        self._ax.clear()
        self._ax.set_facecolor(C['bg'])
        
        R = self.arena_cfg.radius
        
        # === ARENA ===
        # Simple clean arena circle
        arena = plt.Circle((0, 0), R, facecolor=C['arena_fill'], edgecolor=C['arena_edge'], 
                           linewidth=3, alpha=0.9, zorder=1)
        self._ax.add_patch(arena)
        
        # Subtle center cross
        self._ax.plot([-R*0.1, R*0.1], [0, 0], color=C['grid'], lw=1, alpha=0.5, zorder=2)
        self._ax.plot([0, 0], [-R*0.1, R*0.1], color=C['grid'], lw=1, alpha=0.5, zorder=2)
        
        # === STATIONS ===
        for station in self.stations:
            # Station zone
            alpha = 0.7 if station.is_active else 0.3
            station_circle = plt.Circle(
                station.position, station.radius,
                facecolor=C['station'], edgecolor='white',
                linewidth=2, alpha=alpha, zorder=3
            )
            self._ax.add_patch(station_circle)
            
            # Capture progress ring
            if station.capture_progress > 0.0:
                progress_ring = plt.Circle(
                    station.position, station.radius + 2,
                    fill=False, edgecolor=C['evader'], linewidth=4,
                    alpha=station.capture_progress, zorder=4
                )
                self._ax.add_patch(progress_ring)
            
            # Station label (inside)
            self._ax.text(
                station.position[0], station.position[1],
                f"S{station.station_id}", ha='center', va='center',
                fontsize=11, fontweight='bold', color='white', zorder=5
            )
        
        # === TRAILS (draw first, behind agents) ===
        for i, pursuer in enumerate(self.pursuers):
            self._trail_pursuers[i].append(pursuer.position.copy())
            if len(self._trail_pursuers[i]) > 2:
                trail = np.array(self._trail_pursuers[i])
                self._ax.plot(trail[:, 0], trail[:, 1], '-', color=C['pursuer'], 
                             linewidth=2, alpha=0.3, zorder=6)
        
        for j, evader in enumerate(self.evaders):
            if not evader.is_captured:
                self._trail_evaders[j].append(evader.position.copy())
                if len(self._trail_evaders[j]) > 2:
                    trail = np.array(self._trail_evaders[j])
                    self._ax.plot(trail[:, 0], trail[:, 1], '-', color=C['evader'],
                                 linewidth=2, alpha=0.3, zorder=6)
        
        # === PURSUERS ===
        for i, pursuer in enumerate(self.pursuers):
            # Color based on state
            if pursuer.state == PursuerState.INTERCEPTING:
                color = C['pursuer']
            else:
                color = C['pursuer_guard']
            
            # Agent body (larger, cleaner)
            self._ax.scatter(
                pursuer.position[0], pursuer.position[1],
                s=200, c=color, edgecolors='white', linewidths=2,
                marker='o', zorder=10
            )
            
            # Direction indicator
            arrow_end = pursuer.position + pursuer.heading * 4
            self._ax.annotate(
                '', xy=arrow_end, xytext=pursuer.position,
                arrowprops=dict(arrowstyle='->', color='white', lw=2),
                zorder=11
            )
            
            # ID label (offset to avoid overlap)
            self._ax.text(
                pursuer.position[0] + 3, pursuer.position[1] + 3,
                f"P{i}", fontsize=9, fontweight='bold', color=C['text'],
                path_effects=[pe.withStroke(linewidth=2, foreground=C['bg'])],
                zorder=12
            )
        
        # === EVADERS ===
        for j, evader in enumerate(self.evaders):
            if evader.is_captured:
                continue
            
            # Color based on role
            if evader.role == EvaderRole.DECOY:
                color = C['evader_decoy']
                marker = 'D'  # Diamond for decoy
            else:
                color = C['evader']
                marker = '^'  # Triangle for regular evaders
            
            # Agent body
            self._ax.scatter(
                evader.position[0], evader.position[1],
                s=180, c=color, edgecolors='white', linewidths=2,
                marker=marker, zorder=10
            )
            
            # Direction indicator  
            arrow_end = evader.position + evader.heading * 3.5
            self._ax.annotate(
                '', xy=arrow_end, xytext=evader.position,
                arrowprops=dict(arrowstyle='->', color='white', lw=1.5),
                zorder=11
            )
            
            # ID label
            self._ax.text(
                evader.position[0] - 3, evader.position[1] - 4,
                f"E{j}", fontsize=9, fontweight='bold', color=C['text'],
                path_effects=[pe.withStroke(linewidth=2, foreground=C['bg'])],
                zorder=12
            )
        
        # === HUD (Top-left info panel) ===
        active = self.flock.num_active if self.flock else 0
        captured = self.flock.num_captured if self.flock else 0
        stations_lost = sum(not s.is_active for s in self.stations)
        
        hud_text = (
            f"Step: {self.step_count}  |  Time: {self.current_time:.1f}s\n"
            f"Evaders: {active} active, {captured} captured\n"
            f"Stations: {len(self.stations) - stations_lost}/{len(self.stations)} defended"
        )
        
        self._ax.text(
            -R * 0.95, R * 0.95, hud_text,
            fontsize=11, color=C['text'], va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.5', facecolor=C['hud_bg'], 
                     edgecolor=C['arena_edge'], alpha=0.9),
            zorder=20
        )
        
        # === LEGEND (Bottom-right, simplified) ===
        from matplotlib.lines import Line2D
        legend_items = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor=C['pursuer_guard'], 
                   markersize=10, label='Pursuer (Guard)', linestyle='None'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=C['pursuer'], 
                   markersize=10, label='Pursuer (Chase)', linestyle='None'),
            Line2D([0], [0], marker='^', color='w', markerfacecolor=C['evader'], 
                   markersize=10, label='Evader', linestyle='None'),
            Line2D([0], [0], marker='D', color='w', markerfacecolor=C['evader_decoy'], 
                   markersize=9, label='Decoy', linestyle='None'),
        ]
        
        leg = self._ax.legend(
            handles=legend_items, loc='lower right',
            fontsize=9, framealpha=0.9,
            facecolor=C['hud_bg'], edgecolor=C['arena_edge'],
        )
        leg.get_frame().set_linewidth(1.5)
        
        # === LAYOUT ===
        self._ax.set_xlim(-R * 1.15, R * 1.15)
        self._ax.set_ylim(-R * 1.15, R * 1.15)
        self._ax.set_aspect('equal')
        self._ax.axis('off')
        
        # Title
        self._ax.set_title(
            'Multi-Agent Pursuit-Evasion',
            fontsize=16, fontweight='bold', color=C['text'], pad=15
        )
        
        self._fig.tight_layout()
        
        if self.render_mode == "human":
            plt.pause(0.001)
        elif self.render_mode == "rgb_array":
            self._fig.canvas.draw()
            return np.frombuffer(self._fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(
                self._fig.canvas.get_width_height()[::-1] + (3,)
            )
    
    def close(self):
        """Clean up resources."""
        if self._fig is not None:
            import matplotlib.pyplot as plt
            plt.close(self._fig)
            self._fig = None
            self._ax = None


# Wrapper for single-agent training (trains one side at a time)
class PursuerTrainingEnv(gym.Env):
    """Wrapper that exposes only pursuer observations/actions for training."""
    
    def __init__(self, config: Optional[MultiAgentConfig] = None, **kwargs):
        self.multi_env = MultiAgentPursuitEnv(
            config=config, train_pursuers=True, train_evaders=False, **kwargs
        )
        
        num_pursuers = self.multi_env.num_pursuers
        obs_dim = self.multi_env.pursuer_observation_space.shape[0]
        
        # Flatten for single-agent interface
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(num_pursuers * obs_dim,),
            dtype=np.float32
        )
        
        self.action_space = spaces.Box(
            low=np.tile([0.0, -1.0], num_pursuers),
            high=np.tile([1.0, 1.0], num_pursuers),
            dtype=np.float32
        )
        
        self.num_pursuers = num_pursuers
    
    def reset(self, **kwargs):
        obs, info = self.multi_env.reset(**kwargs)
        return obs['pursuers'].flatten(), info
    
    def step(self, action):
        action_reshaped = action.reshape(self.num_pursuers, 2)
        obs, rewards, term, trunc, info = self.multi_env.step({
            'pursuers': action_reshaped,
            'evaders': None,
        })
        reward = float(rewards['pursuers'].sum())
        return obs['pursuers'].flatten(), reward, term, trunc, info
    
    def render(self):
        return self.multi_env.render()
    
    def close(self):
        self.multi_env.close()


class EvaderTrainingEnv(gym.Env):
    """Wrapper that exposes only evader observations/actions for training."""
    
    def __init__(self, config: Optional[MultiAgentConfig] = None, **kwargs):
        self.multi_env = MultiAgentPursuitEnv(
            config=config, train_pursuers=False, train_evaders=True, **kwargs
        )
        
        num_evaders = self.multi_env.num_evaders
        obs_dim = self.multi_env.evader_observation_space.shape[0]
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(num_evaders * obs_dim,),
            dtype=np.float32
        )
        
        # Evader action: [throttle, heading_adj, decoy_signal] per evader
        self.action_space = spaces.Box(
            low=np.tile([0.0, -1.0, 0.0], num_evaders),
            high=np.tile([1.0, 1.0, 1.0], num_evaders),
            dtype=np.float32
        )
        
        self.num_evaders = num_evaders
    
    def reset(self, **kwargs):
        obs, info = self.multi_env.reset(**kwargs)
        return obs['evaders'].flatten(), info
    
    def step(self, action):
        action_reshaped = action.reshape(self.num_evaders, 3)
        obs, rewards, term, trunc, info = self.multi_env.step({
            'pursuers': None,
            'evaders': action_reshaped,
        })
        reward = float(rewards['evaders'].sum())
        return obs['evaders'].flatten(), reward, term, trunc, info
    
    def render(self):
        return self.multi_env.render()
    
    def close(self):
        self.multi_env.close()
