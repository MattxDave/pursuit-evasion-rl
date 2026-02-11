"""
Multi-Agent Entities
====================
Station, Smart Pursuer, and Smart Evader classes for multi-agent pursuit-evasion.
"""

import math
import numpy as np
from typing import Optional, Tuple, List, Dict, Any, Set
from dataclasses import dataclass, field
from enum import Enum, auto

from .geometry import (
    lead_intercept,
    pure_pursuit,
    rotate_vector,
    unit_vector,
    normalize_angle,
    random_point_in_circle,
)


# =============================================================================
# STATION
# =============================================================================

@dataclass
class Station:
    """
    A station/base that pursuers guard and evaders try to capture.
    
    Stations are fixed positions on the playfield. Pursuers are assigned to
    guard stations and only engage when evaders enter their radar range.
    Evaders win by reaching and capturing stations (staying for capture_time).
    """
    station_id: int
    position: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    radius: float = 5.0  # Physical size / capture zone
    
    # Assigned pursuers
    assigned_pursuers: List[int] = field(default_factory=list)
    
    # Capture status
    is_active: bool = True           # False = captured by evaders
    capture_progress: float = 0.0    # 0-1, progress toward capture
    capturing_evader_id: Optional[int] = None  # Which evader is capturing
    
    def distance_to(self, pos: np.ndarray) -> float:
        """Distance from a position to station center."""
        return float(np.linalg.norm(pos - self.position))
    
    def is_at_station(self, pos: np.ndarray, tolerance: float = 1.0) -> bool:
        """Check if position is at/near station."""
        return self.distance_to(pos) <= self.radius + tolerance
    
    def is_in_capture_zone(self, pos: np.ndarray) -> bool:
        """Check if position is within capture zone."""
        return self.distance_to(pos) <= self.radius
    
    def update_capture(self, evader_present: bool, evader_id: Optional[int], dt: float, capture_time: float):
        """Update capture progress."""
        if not self.is_active:
            return
            
        if evader_present and evader_id is not None:
            if self.capturing_evader_id is None or self.capturing_evader_id == evader_id:
                self.capturing_evader_id = evader_id
                self.capture_progress += dt / capture_time
                if self.capture_progress >= 1.0:
                    self.is_active = False
                    self.capture_progress = 1.0
        else:
            # Reset progress if no evader present
            self.capture_progress = max(0.0, self.capture_progress - dt / capture_time * 0.5)
            if self.capture_progress <= 0:
                self.capturing_evader_id = None
    
    def reset(self):
        """Reset station to initial state."""
        self.is_active = True
        self.capture_progress = 0.0
        self.capturing_evader_id = None
    
    def assign_pursuer(self, pursuer_id: int):
        """Assign a pursuer to guard this station."""
        if pursuer_id not in self.assigned_pursuers:
            self.assigned_pursuers.append(pursuer_id)
    
    def remove_pursuer(self, pursuer_id: int):
        """Remove pursuer from station assignment."""
        if pursuer_id in self.assigned_pursuers:
            self.assigned_pursuers.remove(pursuer_id)


def spawn_stations(
    num_stations: int,
    arena_radius: float,
    station_radius: float,
    min_separation: float,
    spawn_fraction: float,
    rng: np.random.Generator,
) -> List[Station]:
    """
    Spawn stations at random positions within the arena.
    
    Args:
        num_stations: Number of stations to create
        arena_radius: Radius of the arena
        station_radius: Physical radius of each station
        min_separation: Minimum distance between stations
        spawn_fraction: Fraction of arena radius for spawn zone
        rng: Random number generator
    
    Returns:
        List of Station objects
    """
    stations = []
    spawn_radius = arena_radius * spawn_fraction
    
    for i in range(num_stations):
        for _ in range(100):  # Max attempts
            pos = random_point_in_circle(spawn_radius, rng)
            
            # Check separation from other stations
            valid = True
            for other in stations:
                if np.linalg.norm(pos - other.position) < min_separation:
                    valid = False
                    break
            
            if valid:
                stations.append(Station(
                    station_id=i,
                    position=pos.astype(np.float32),
                    radius=station_radius,
                ))
                break
        else:
            # Fallback: place anyway
            pos = random_point_in_circle(spawn_radius, rng)
            stations.append(Station(
                station_id=i,
                position=pos.astype(np.float32),
                radius=station_radius,
            ))
    
    return stations


# =============================================================================
# PURSUER STATE MACHINE
# =============================================================================

class PursuerState(Enum):
    """State machine states for pursuer behavior."""
    GUARDING = auto()       # At station, monitoring radar
    INTERCEPTING = auto()   # Chasing detected evader
    RETURNING = auto()      # Returning to station
    RECHARGING = auto()     # At station, recharging battery


@dataclass
class RadarContact:
    """Information about a detected evader."""
    evader_id: int
    position: np.ndarray
    velocity: np.ndarray
    distance: float
    time_detected: float
    
    def age(self, current_time: float) -> float:
        """How old is this contact."""
        return current_time - self.time_detected


@dataclass
class SmartPursuer:
    """
    Smart pursuer with radar, communication, and state machine.
    
    Features:
    - Radar detection system
    - Inter-pursuer communication
    - State machine: GUARDING -> INTERCEPTING -> RETURNING
    - Battery management with recharging at station
    - Coordination with partner pursuer
    """
    pursuer_id: int
    position: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    heading: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0], dtype=np.float32))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    
    # Movement
    max_speed: float = 10.0
    turn_rate_limit: float = math.pi  # rad/s
    actual_speed: float = 0.0
    
    # Radar
    radar_range: float = 15.0
    radar_fov: float = 2 * math.pi  # radians (360 degrees)
    radar_contacts: Dict[int, RadarContact] = field(default_factory=dict)
    
    # Communication
    comm_range: float = 30.0
    received_messages: List[Dict[str, Any]] = field(default_factory=list)
    
    # Battery
    battery: float = 100.0
    battery_capacity: float = 100.0
    battery_drain_rate: float = 0.2
    battery_recharge_rate: float = 0.5
    return_threshold: float = 20.0  # Return when below this %

    # Guard patrol behavior
    guard_patrol_radius: float = 3.5
    guard_patrol_speed: float = 0.4
    max_time_away_from_station: float = 6.0
    
    # State machine
    state: PursuerState = PursuerState.GUARDING
    assigned_station: Optional[int] = None
    target_evader_id: Optional[int] = None
    target_locked: bool = False  # Once locked, keep chasing same target
    partner_id: Optional[int] = None
    
    # Tracking
    intercept_count: int = 0
    time_away_from_station: float = 0.0
    
    def scan_radar(
        self,
        evaders: List['SmartEvader'],
        current_time: float,
    ) -> List[RadarContact]:
        """
        Scan for evaders within radar range.
        
        Returns list of new contacts detected this scan.
        """
        new_contacts = []
        
        for evader in evaders:
            if evader.is_captured:
                continue
                
            # Distance check
            dist = np.linalg.norm(evader.position - self.position)
            if dist > self.radar_range:
                continue
            
            # FOV check (if not omnidirectional)
            if self.radar_fov < 2 * math.pi:
                to_evader = unit_vector(evader.position - self.position)
                angle = math.acos(np.clip(np.dot(self.heading, to_evader), -1, 1))
                if angle > self.radar_fov / 2:
                    continue
            
            # Create contact
            contact = RadarContact(
                evader_id=evader.evader_id,
                position=evader.position.copy(),
                velocity=evader.velocity.copy(),
                distance=dist,
                time_detected=current_time,
            )
            
            # Track if new
            if evader.evader_id not in self.radar_contacts:
                new_contacts.append(contact)
            
            self.radar_contacts[evader.evader_id] = contact
        
        # Remove stale contacts
        stale_ids = [
            eid for eid, contact in self.radar_contacts.items()
            if contact.age(current_time) > 2.0  # 2 second timeout
        ]
        for eid in stale_ids:
            del self.radar_contacts[eid]
        
        return new_contacts
    
    def broadcast_threat(
        self,
        contact: RadarContact,
        pursuers: List['SmartPursuer'],
    ) -> List[Dict[str, Any]]:
        """
        Broadcast detected threat to nearby pursuers.
        
        Returns messages sent.
        """
        messages = []
        
        for other in pursuers:
            if other.pursuer_id == self.pursuer_id:
                continue
            
            dist = np.linalg.norm(other.position - self.position)
            if dist <= self.comm_range:
                msg = {
                    'type': 'threat_detected',
                    'sender': self.pursuer_id,
                    'evader_id': contact.evader_id,
                    'position': contact.position.copy(),
                    'velocity': contact.velocity.copy(),
                    'distance': contact.distance,
                }
                other.received_messages.append(msg)
                messages.append(msg)
        
        return messages
    
    def decide_action(
        self,
        stations: List[Station],
        evaders: List['SmartEvader'],
        pursuers: List['SmartPursuer'],
        current_time: float,
    ) -> Tuple[str, Optional[int]]:
        """
        Decide what action to take based on current state and information.
        
        STATION DEFENSE strategy:
        - Each pursuer defends its OWN station regardless of global threat ranking
        - Both pursuers at a station may intercept when multiple evaders are close
        - One pursuer always stays near the station unless overwhelming threat
        - Pre-emptive move toward evaders heading toward our station
        - HUNT MODE: when pursuers outnumber evaders, all converge to hunt
        
        Returns:
            (action_type, target_id)
        """
        # Low battery - must return
        if self.battery < self.return_threshold and self.state != PursuerState.RECHARGING:
            self.target_locked = False
            return 'return', None
        
        self.received_messages.clear()
        self.scan_radar(evaders, current_time)
        
        # Get my station
        my_station = None
        if self.assigned_station is not None and self.assigned_station < len(stations):
            my_station = stations[self.assigned_station]
        
        if my_station is None:
            return 'guard', None
        
        # Get active evaders and pursuers
        active_evaders = [e for e in evaders if not e.is_captured and not e.is_at_goal]
        active_pursuers = [p for p in pursuers if p.battery > 5.0]
        
        if not active_evaders:
            if my_station.is_at_station(self.position, tolerance=5.0):
                return 'guard', None
            return 'return', None
        
        # Find partner
        partner = None
        if self.partner_id is not None and self.partner_id < len(pursuers):
            partner = pursuers[self.partner_id]
        
        # === HUNT MODE ===
        # When pursuers significantly outnumber evaders, ALL pursuers converge
        # to eliminate the remaining evaders (stations are safe with few threats)
        num_active_evaders = len(active_evaders)
        num_active_pursuers = len(active_pursuers)
        
        if num_active_evaders <= 1 and num_active_pursuers >= 3:
            # Hunt mode: converge all pursuers on the last evader(s)
            target = min(active_evaders, key=lambda e: np.linalg.norm(e.position - self.position))
            self.target_evader_id = target.evader_id
            return 'intercept', target.evader_id
        
        if num_active_evaders <= 2 and num_active_pursuers >= num_active_evaders * 2:
            # Numerical superiority — each pursuer picks the closest evader
            target = min(active_evaders, key=lambda e: np.linalg.norm(e.position - self.position))
            self.target_evader_id = target.evader_id
            return 'intercept', target.evader_id
        
        # === THREAT ASSESSMENT FOR MY STATION ===
        # Rank evaders by how threatening they are to MY station
        # Uses both distance and heading (evaders heading toward us are more dangerous)
        def threat_score(e):
            to_station = my_station.position - e.position
            dist = np.linalg.norm(to_station)
            if dist < 0.1:
                return 999.0  # Already at station = max threat
            heading_alignment = np.dot(unit_vector(e.velocity), unit_vector(to_station)) if np.linalg.norm(e.velocity) > 0.1 else 0.0
            # Higher score = bigger threat. Close + heading toward us = very dangerous
            return max(0.0, heading_alignment + 0.5) / (dist + 1.0) * 100.0
        
        evader_threats = [(e, threat_score(e)) for e in active_evaders]
        evader_threats.sort(key=lambda x: x[1], reverse=True)
        
        # Evaders within our defense zone (radar range or heading toward us)
        defense_zone_evaders = [
            (e, score) for e, score in evader_threats
            if np.linalg.norm(e.position - my_station.position) < self.radar_range * 1.2
            or score > 2.0  # heading toward us even if far
        ]
        
        num_threats = len(defense_zone_evaders)
        
        # URGENT: station is actively being captured — go intercept no matter what
        if my_station.capture_progress > 0.1:
            # Station under active attack!
            attacker = None
            for e in active_evaders:
                if my_station.is_in_capture_zone(e.position):
                    attacker = e
                    break
            if attacker:
                self.target_evader_id = attacker.evader_id
                return 'intercept', attacker.evader_id
        
        if num_threats == 0:
            # No threats nearby — guard station
            if my_station.is_at_station(self.position, tolerance=5.0):
                return 'guard', None
            return 'return', None
        
        # We have at least one threat near our station
        best_target = defense_zone_evaders[0][0]
        
        if num_threats >= 2 and partner is not None:
            # Multiple evaders approaching — BOTH pursuers can intercept
            # each takes the closest threat to themselves
            my_dist_to_first = np.linalg.norm(self.position - defense_zone_evaders[0][0].position)
            my_dist_to_second = np.linalg.norm(self.position - defense_zone_evaders[1][0].position)
            partner_dist_to_first = np.linalg.norm(partner.position - defense_zone_evaders[0][0].position)
            partner_dist_to_second = np.linalg.norm(partner.position - defense_zone_evaders[1][0].position)
            
            # Assign each pursuer to the evader they're closest to (Hungarian-lite)
            if my_dist_to_first + partner_dist_to_second <= my_dist_to_second + partner_dist_to_first:
                best_target = defense_zone_evaders[0][0]
            else:
                best_target = defense_zone_evaders[1][0]
            
            self.target_evader_id = best_target.evader_id
            return 'intercept', best_target.evader_id
        
        # Single threat — coordinate with partner (one intercepts, one guards)
        if partner is not None:
            partner_is_intercepting = partner.state == PursuerState.INTERCEPTING
            partner_chasing_same = partner.target_evader_id == best_target.evader_id
            
            if partner_is_intercepting and partner_chasing_same:
                # Partner already has this target — I guard
                if my_station.is_at_station(self.position, tolerance=5.0):
                    return 'guard', None
                return 'return', None
            
            if partner_is_intercepting and not partner_chasing_same:
                # Partner is chasing something else — I take this target
                self.target_evader_id = best_target.evader_id
                return 'intercept', best_target.evader_id
            
            # Neither of us is intercepting yet — closer one goes
            my_dist = np.linalg.norm(self.position - best_target.position)
            partner_dist = np.linalg.norm(partner.position - best_target.position)
            
            if partner_dist < my_dist:
                # Partner is closer — I stay guarding
                if my_station.is_at_station(self.position, tolerance=5.0):
                    return 'guard', None
                return 'return', None
        
        # I'm the one intercepting
        self.target_evader_id = best_target.evader_id
        return 'intercept', best_target.evader_id
    
    def compute_intercept_heading(
        self,
        target_pos: np.ndarray,
        target_vel: np.ndarray,
        speed: float,
    ) -> np.ndarray:
        """Compute classical intercept heading toward target."""
        heading, _, feasible = lead_intercept(
            self.position, target_pos, target_vel, speed
        )
        
        if not feasible:
            heading = pure_pursuit(self.position, target_pos)
        
        return heading
    
    def step(
        self,
        action: np.ndarray,
        target: Optional['SmartEvader'],
        station: Optional[Station],
        dt: float,
    ) -> np.ndarray:
        """
        Execute one step.
        
        Args:
            action: [throttle, heading_nudge] from RL policy
            target: Target evader (if intercepting)
            station: Assigned station (if returning/guarding)
            dt: Time step
        
        Returns:
            New position
        """
        throttle = float(np.clip(action[0], 0, 1))
        nudge = float(action[1]) * 0.5  # ±0.5 rad nudge
        
        # Compute base speed based on urgency
        if target is not None:
            # Chasing target - go full speed!
            requested_speed = self.max_speed
        elif self.state == PursuerState.RETURNING:
            # Returning to station urgently - sprint back
            requested_speed = self.max_speed * 0.85
        else:
            # Guarding patrol speed when not chasing
            requested_speed = max(self.guard_patrol_speed, throttle) * self.max_speed
        
        # Battery limiting
        battery_factor = max(0.4, self.battery / self.battery_capacity)
        self.actual_speed = min(requested_speed, self.max_speed * battery_factor)
        
        # Compute desired heading based on target
        # PRIORITY: If we have a target, ALWAYS chase it regardless of state
        if target is not None and not target.is_captured:
            base_heading = self.compute_intercept_heading(
                target.position, target.velocity, self.actual_speed
            )
            # Force state to intercepting
            self.state = PursuerState.INTERCEPTING
        elif self.state == PursuerState.RETURNING and station is not None:
            base_heading = pure_pursuit(self.position, station.position)
        else:
            # Guarding - patrol a small circle around the station
            if station is not None and self.state == PursuerState.GUARDING:
                to_station = station.position - self.position
                dist = float(np.linalg.norm(to_station))
                if dist > 0.1:
                    radial_dir = to_station / dist
                else:
                    radial_dir = self.heading
                tangential = np.array([-radial_dir[1], radial_dir[0]], dtype=np.float32)
                radial_error = dist - self.guard_patrol_radius
                correction = -0.5 * radial_error * radial_dir
                patrol_vec = tangential + correction
                if np.linalg.norm(patrol_vec) > 0.1:
                    base_heading = unit_vector(patrol_vec)
                else:
                    base_heading = self.heading
            else:
                # Default: maintain heading
                base_heading = self.heading
        
        # Apply nudge
        desired_heading = rotate_vector(base_heading, nudge)
        
        # Apply turn rate limit
        current_angle = math.atan2(self.heading[1], self.heading[0])
        desired_angle = math.atan2(desired_heading[1], desired_heading[0])
        delta = normalize_angle(desired_angle - current_angle)
        max_delta = self.turn_rate_limit * dt
        delta = np.clip(delta, -max_delta, max_delta)
        
        new_angle = current_angle + delta
        self.heading = np.array([math.cos(new_angle), math.sin(new_angle)], dtype=np.float32)
        
        # Update velocity and position
        self.velocity = self.actual_speed * self.heading
        self.position = self.position + self.velocity * dt
        
        # Battery drain
        if self.state != PursuerState.RECHARGING:
            speed_ratio = self.actual_speed / self.max_speed
            drain = self.battery_drain_rate * (speed_ratio ** 2)
            self.battery = max(0.0, self.battery - drain)
        else:
            # Recharge at station
            self.battery = min(self.battery_capacity, self.battery + self.battery_recharge_rate)

        # Track time away from station for coordination
        if station is not None:
            dist_to_station = np.linalg.norm(self.position - station.position)
            if dist_to_station > self.guard_patrol_radius + 1.0:
                self.time_away_from_station += dt
            else:
                self.time_away_from_station = 0.0
        
        return self.position
    
    def reset(
        self,
        position: np.ndarray,
        station_id: int,
        partner_id: Optional[int] = None,
    ):
        """Reset pursuer state."""
        self.position = position.astype(np.float32)
        self.heading = np.array([1.0, 0.0], dtype=np.float32)
        self.velocity = np.zeros(2, dtype=np.float32)
        self.battery = self.battery_capacity
        self.state = PursuerState.GUARDING
        self.assigned_station = station_id
        self.partner_id = partner_id
        self.target_evader_id = None
        self.radar_contacts.clear()
        self.received_messages.clear()
        self.intercept_count = 0
        self.time_away_from_station = 0.0
        self.actual_speed = 0.0

    @property
    def battery_percent(self) -> float:
        return (self.battery / self.battery_capacity) * 100.0


# =============================================================================
# EVADER STATE MACHINE
# =============================================================================

class EvaderRole(Enum):
    """Role within the flock."""
    LEADER = auto()         # Leading the formation
    FOLLOWER = auto()       # Following in formation
    DECOY = auto()          # Drawing pursuers away
    ESCAPED = auto()        # Successfully evaded
    CAPTURED = auto()       # Was captured


@dataclass
class SmartEvader:
    """
    Smart evader with RL-controlled behavior.
    
    FLOCK TACTICS:
    - Leader heads toward the goal (station)
    - Followers maintain formation while moving toward the goal
    - Flock shares threat info and applies local evasion when needed
    """
    evader_id: int
    position: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    heading: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0], dtype=np.float32))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    goal: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    
    # Movement
    max_speed: float = 6.0
    turn_rate_limit: float = 1.5 * math.pi  # rad/s (more agile)
    actual_speed: float = 0.0
    
    # Sensing
    detection_range: float = 20.0
    detected_pursuers: Dict[int, np.ndarray] = field(default_factory=dict)  # id -> position
    
    # Communication
    comm_range: float = 15.0
    received_messages: List[Dict[str, Any]] = field(default_factory=list)
    
    # Role and state - all evaders are equal attackers
    role: EvaderRole = EvaderRole.FOLLOWER  # Will be set to LEADER (attacker mode)
    is_captured: bool = False
    is_at_goal: bool = False
    is_being_chased: bool = False  # True if a pursuer is locked onto us
    
    # Flock
    flock_position_idx: int = 0  # Position in formation
    
    def scan_for_pursuers(
        self,
        pursuers: List[SmartPursuer],
    ) -> List[int]:
        """Detect nearby pursuers."""
        detected = []
        
        for pursuer in pursuers:
            dist = np.linalg.norm(pursuer.position - self.position)
            if dist <= self.detection_range:
                self.detected_pursuers[pursuer.pursuer_id] = pursuer.position.copy()
                detected.append(pursuer.pursuer_id)
        
        return detected
    
    def broadcast_to_flock(
        self,
        message_type: str,
        data: Dict[str, Any],
        flock: List['SmartEvader'],
    ):
        """Broadcast message to flock members in range."""
        for other in flock:
            if other.evader_id == self.evader_id:
                continue
            if other.is_captured:
                continue
                
            dist = np.linalg.norm(other.position - self.position)
            if dist <= self.comm_range:
                msg = {
                    'type': message_type,
                    'sender': self.evader_id,
                    **data
                }
                other.received_messages.append(msg)
    
    def share_threat_info(self, flock: List['SmartEvader']):
        """Share detected pursuer positions with nearby flock members."""
        if not self.detected_pursuers:
            return
        
        # Broadcast threat info to flock
        self.broadcast_to_flock(
            message_type='threat',
            data={
                'pursuers': {pid: pos.tolist() for pid, pos in self.detected_pursuers.items()},
                'sender_pos': self.position.tolist(),
            },
            flock=flock,
        )
    
    def process_flock_messages(self):
        """Process messages from flock mates to update threat awareness."""
        for msg in self.received_messages:
            if msg['type'] == 'threat':
                # Add shared threat info to our detected pursuers
                for pid, pos in msg['pursuers'].items():
                    if int(pid) not in self.detected_pursuers:
                        self.detected_pursuers[int(pid)] = np.array(pos, dtype=np.float32)
        
        # Clear processed messages
        self.received_messages.clear()
    
    def compute_flock_forces(
        self,
        flock: List['SmartEvader'],
        separation_dist: float = 3.0,
        cohesion_weight: float = 0.7,  # Increased from 0.5
        alignment_weight: float = 0.5,  # Increased from 0.3
    ) -> np.ndarray:
        """
        Compute flocking behavior forces (separation, cohesion, alignment).
        The flock should move TOGETHER toward stations.
        """
        separation = np.zeros(2, dtype=np.float32)
        cohesion = np.zeros(2, dtype=np.float32)
        alignment = np.zeros(2, dtype=np.float32)
        
        neighbors = [e for e in flock if e.evader_id != self.evader_id and not e.is_captured]
        
        if not neighbors:
            # No neighbors = just head to goal
            return self.compute_goal_force()
        
        # Find the leader for reference
        leader = next((e for e in neighbors if e.role == EvaderRole.LEADER), None)
        
        for other in neighbors:
            diff = self.position - other.position
            dist = np.linalg.norm(diff)
            
            # Separation (avoid too close)
            if dist < separation_dist and dist > 0.1:
                separation += diff / (dist ** 2)
            
            # Cohesion (move toward center, with bias toward leader)
            if other.role == EvaderRole.LEADER:
                cohesion += 2.0 * other.position  # Extra weight for leader
            else:
                cohesion += other.position
            
            # Alignment (match velocity - head same direction)
            alignment += other.velocity
        
        # Normalize separation
        if np.linalg.norm(separation) > 0:
            separation = unit_vector(separation)
        
        # Cohesion: if leader exists, add extra pull toward leader
        weight_sum = len(neighbors) + (1.0 if leader else 0.0)  # Leader counted twice
        cohesion = cohesion / weight_sum - self.position
        if np.linalg.norm(cohesion) > 0:
            cohesion = unit_vector(cohesion)
        
        if np.linalg.norm(alignment) > 0:
            alignment = unit_vector(alignment)
        
        return separation + cohesion_weight * cohesion + alignment_weight * alignment
    
    def compute_evasion_force(self) -> np.ndarray:
        """Compute force to evade detected pursuers."""
        if not self.detected_pursuers:
            return np.zeros(2, dtype=np.float32)
        
        evasion = np.zeros(2, dtype=np.float32)
        
        for pid, p_pos in self.detected_pursuers.items():
            diff = self.position - p_pos
            dist = np.linalg.norm(diff)
            if dist > 0.1:
                # Stronger evasion for closer pursuers
                weight = 1.0 / (dist ** 2)
                evasion += weight * unit_vector(diff)
        
        if np.linalg.norm(evasion) > 0:
            evasion = unit_vector(evasion)
        
        return evasion
    
    def compute_goal_force(self) -> np.ndarray:
        """Compute force toward goal."""
        to_goal = self.goal - self.position
        dist = np.linalg.norm(to_goal)
        if dist > 0.1:
            return unit_vector(to_goal)
        return np.zeros(2, dtype=np.float32)
    
    def _find_best_opening(self, flock: List['SmartEvader']) -> np.ndarray:
        """
        Find the direction with the best opening to the goal.
        
        Analyzes pursuer positions and finds a path that:
        1. Gets closer to the goal
        2. Avoids pursuers as much as possible
        """
        if not self.detected_pursuers:
            # No pursuers detected - go straight to goal
            return self.compute_goal_force()
        
        goal_dir = self.compute_goal_force()
        
        # Sample directions around goal direction to find best opening
        best_dir = goal_dir
        best_score = -999
        
        for angle_offset in np.linspace(-math.pi/2, math.pi/2, 9):  # -90 to +90 degrees
            # Test direction
            test_angle = math.atan2(goal_dir[1], goal_dir[0]) + angle_offset
            test_dir = np.array([math.cos(test_angle), math.sin(test_angle)], dtype=np.float32)
            
            # Score = how good is this direction?
            # Positive: toward goal, away from pursuers
            # Negative: away from goal, toward pursuers
            
            # Goal alignment (1.0 = toward goal, -1.0 = away)
            goal_alignment = np.dot(test_dir, goal_dir)
            
            # Pursuer clearance (higher = more clearance from pursuers)
            min_pursuer_clearance = 999
            for p_pos in self.detected_pursuers.values():
                to_pursuer = p_pos - self.position
                dist = np.linalg.norm(to_pursuer)
                if dist > 0.1:
                    pursuer_dir = to_pursuer / dist
                    # How aligned is this test direction with the pursuer direction?
                    alignment = np.dot(test_dir, pursuer_dir)
                    # Clearance is negative alignment (we want to go AWAY from pursuers)
                    clearance = -alignment * (20 / max(dist, 1))  # Weight by distance
                    min_pursuer_clearance = min(min_pursuer_clearance, clearance)
            
            # Combined score: goal is important, but avoiding pursuers is critical
            score = 0.4 * goal_alignment + 0.6 * min_pursuer_clearance
            
            if score > best_score:
                best_score = score
                best_dir = test_dir
        
        return best_dir
    
    def step(
        self,
        action: np.ndarray,
        flock: List['SmartEvader'],
        dt: float,
        formation_target: Optional[np.ndarray] = None,
        separation_dist: float = 3.0,
        cohesion_weight: float = 0.7,
        alignment_weight: float = 0.5,
    ) -> np.ndarray:
        """
        Execute one step with RL action.
        
        FLOCK TACTICS: Stay in formation while progressing toward the goal.
        If a pursuer gets close, apply local evasion while keeping formation.
        
        Args:
            action: [throttle, heading_adjustment, scatter_signal] from RL policy
            flock: List of all evaders
            dt: Time step
        
        Returns:
            New position
        """
        if self.is_captured:
            return self.position
        
        throttle = float(np.clip(action[0], 0, 1))
        heading_adj = float(action[1]) * 0.5  # Reduced RL influence, more classical control
        scatter_signal = float(action[2]) if len(action) > 2 else 0.0

        # Use throttle while keeping a minimum speed to avoid stalling
        self.actual_speed = max(0.6, throttle) * self.max_speed

        # Compute forces
        goal_force = self.compute_goal_force()
        evasion_force = self.compute_evasion_force()
        flock_force = self.compute_flock_forces(
            flock,
            separation_dist=separation_dist,
            cohesion_weight=cohesion_weight,
            alignment_weight=alignment_weight,
        )

        best_opening = None
        if scatter_signal > 0.5:
            best_opening = self._find_best_opening(flock)

        formation_force = np.zeros(2, dtype=np.float32)
        if formation_target is not None:
            to_formation = formation_target - self.position
            if np.linalg.norm(to_formation) > 0.1:
                formation_force = unit_vector(to_formation)

        # Adjust weights when a pursuer is very close
        min_pursuer_dist = None
        if self.detected_pursuers:
            min_pursuer_dist = min(
                np.linalg.norm(p_pos - self.position)
                for p_pos in self.detected_pursuers.values()
            )

        if min_pursuer_dist is not None and min_pursuer_dist < 8.0:
            # Emergency scatter when a pursuer is very close
            if best_opening is None:
                best_opening = self._find_best_opening(flock)
            base_direction = 0.70 * evasion_force + 0.30 * best_opening
        elif min_pursuer_dist is not None and min_pursuer_dist < 10.0:
            # Strong evasion, but keep formation intent
            if formation_target is not None:
                base_direction = 0.55 * evasion_force + 0.30 * formation_force + 0.15 * goal_force
            else:
                base_direction = 0.60 * evasion_force + 0.25 * goal_force + 0.15 * flock_force
        elif scatter_signal > 0.5 and best_opening is not None:
            # Voluntary scatter to create openings while still moving forward
            base_direction = 0.55 * best_opening + 0.30 * goal_force + 0.15 * evasion_force
        else:
            # Normal flocking behavior
            if formation_target is not None:
                base_direction = 0.60 * formation_force + 0.25 * goal_force + 0.15 * flock_force
            else:
                base_direction = 0.60 * goal_force + 0.25 * flock_force + 0.15 * evasion_force
        
        if np.linalg.norm(base_direction) < 0.1:
            base_direction = self.heading
        else:
            base_direction = unit_vector(base_direction)
        
        # Apply SMALL RL heading adjustment (classical behavior dominates)
        desired_angle = math.atan2(base_direction[1], base_direction[0]) + heading_adj
        
        # Apply turn rate limit
        current_angle = math.atan2(self.heading[1], self.heading[0])
        delta = normalize_angle(desired_angle - current_angle)
        max_delta = self.turn_rate_limit * dt
        delta = np.clip(delta, -max_delta, max_delta)
        
        new_angle = current_angle + delta
        self.heading = np.array([math.cos(new_angle), math.sin(new_angle)], dtype=np.float32)
        
        # Update velocity and position
        self.velocity = self.actual_speed * self.heading
        self.position = self.position + self.velocity * dt
        
        # Clear detected pursuers (re-scan each step)
        self.detected_pursuers.clear()
        
        return self.position
    
    def reset(
        self,
        position: np.ndarray,
        goal: np.ndarray,
        role: EvaderRole = EvaderRole.FOLLOWER,
        flock_idx: int = 0,
    ):
        """Reset evader state."""
        self.position = position.astype(np.float32)
        self.goal = goal.astype(np.float32)
        
        # Initial heading toward goal
        to_goal = goal - position
        if np.linalg.norm(to_goal) > 0.1:
            self.heading = unit_vector(to_goal)
        else:
            self.heading = np.array([1.0, 0.0], dtype=np.float32)
        
        self.velocity = np.zeros(2, dtype=np.float32)
        self.role = role
        self.flock_position_idx = flock_idx
        self.is_captured = False
        self.is_at_goal = False
        self.detected_pursuers.clear()
        self.received_messages.clear()
        self.actual_speed = 0.0
    
    @property
    def speed(self) -> float:
        return float(np.linalg.norm(self.velocity))
    
    def distance_to_goal(self) -> float:
        return float(np.linalg.norm(self.goal - self.position))


# =============================================================================
# FLOCK CONTROLLER
# =============================================================================

class FlockController:
    """
    Manages the evader flock coordination.
    
    Handles:
    - Formation maintenance
    - Decoy assignment
    - Threat communication
    - Mission success tracking
    """
    
    def __init__(
        self,
        evaders: List[SmartEvader],
        formation_type: str = "wedge",
        formation_spacing: float = 4.0,
    ):
        self.evaders = evaders
        self.formation_type = formation_type
        self.formation_spacing = formation_spacing
        
        # Assign initial roles
        if evaders:
            evaders[0].role = EvaderRole.LEADER
            for e in evaders[1:]:
                e.role = EvaderRole.FOLLOWER
    
    def get_formation_positions(
        self,
        leader_pos: np.ndarray,
        leader_heading: np.ndarray,
    ) -> List[np.ndarray]:
        """
        Compute formation positions relative to leader.
        """
        positions = [leader_pos.copy()]
        
        if self.formation_type == "wedge":
            # V formation behind leader
            angle_offset = math.pi * 0.75  # 135 degrees back
            for i, evader in enumerate(self.evaders[1:], 1):
                side = 1 if i % 2 == 1 else -1
                rank = (i + 1) // 2
                
                offset_angle = math.atan2(leader_heading[1], leader_heading[0]) + side * angle_offset
                offset = self.formation_spacing * rank * np.array([
                    math.cos(offset_angle),
                    math.sin(offset_angle)
                ])
                positions.append(leader_pos + offset)
        
        elif self.formation_type == "line":
            # Single file behind leader
            back_dir = -leader_heading
            for i in range(1, len(self.evaders)):
                positions.append(leader_pos + back_dir * self.formation_spacing * i)
        
        elif self.formation_type == "diamond":
            # Diamond formation
            if len(self.evaders) >= 2:
                positions.append(leader_pos - leader_heading * self.formation_spacing)
            if len(self.evaders) >= 3:
                perp = np.array([-leader_heading[1], leader_heading[0]])
                positions.append(leader_pos - leader_heading * self.formation_spacing * 0.5 + perp * self.formation_spacing * 0.5)
            if len(self.evaders) >= 4:
                positions.append(leader_pos - leader_heading * self.formation_spacing * 0.5 - perp * self.formation_spacing * 0.5)
        
        return positions
    
    def assign_decoy(
        self,
        threat_direction: np.ndarray,
        method: str = "nearest",
    ) -> Optional[int]:
        """
        Assign a decoy to draw pursuers away.
        
        Returns evader_id of assigned decoy, or None.
        """
        available = [e for e in self.evaders if e.role == EvaderRole.FOLLOWER and not e.is_captured]
        
        if not available:
            return None
        
        if method == "nearest":
            # Assign evader nearest to threat direction
            best = min(available, key=lambda e: np.dot(
                e.position - self.evaders[0].position,
                threat_direction
            ))
        elif method == "random":
            best = np.random.choice(available)
        else:
            # Designated: use last follower
            best = available[-1]
        
        best.role = EvaderRole.DECOY
        return best.evader_id
    
    def update(self, dt: float):
        """Update flock state each step."""
        # Check for captured/escaped evaders
        active = [e for e in self.evaders if not e.is_captured and not e.is_at_goal]
        
        if not active:
            return
        
        # Rotate leader if current leader is captured
        leader = next((e for e in self.evaders if e.role == EvaderRole.LEADER), None)
        if leader is None or leader.is_captured:
            # Promote first available follower
            for e in self.evaders:
                if not e.is_captured and e.role == EvaderRole.FOLLOWER:
                    e.role = EvaderRole.LEADER
                    break
    
    def get_flock_center(self) -> np.ndarray:
        """Get center of active flock."""
        active = [e for e in self.evaders if not e.is_captured]
        if not active:
            return np.zeros(2, dtype=np.float32)
        return np.mean([e.position for e in active], axis=0).astype(np.float32)
    
    @property
    def num_active(self) -> int:
        return sum(1 for e in self.evaders if not e.is_captured and not e.is_at_goal)
    
    @property
    def num_at_goal(self) -> int:
        return sum(1 for e in self.evaders if e.is_at_goal)
    
    @property
    def num_captured(self) -> int:
        return sum(1 for e in self.evaders if e.is_captured)
