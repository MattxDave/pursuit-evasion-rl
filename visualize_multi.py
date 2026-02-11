"""
Multi-Agent Pursuit-Evasion Visualization
==========================================
Real-time visualization and trajectory plotting for trained agents.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
import argparse
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass
from stable_baselines3 import PPO

from src.multi_config import get_default_multi_config, MultiAgentConfig
from src.multi_environment import MultiAgentPursuitEnv
from src.multi_agents import PursuerState, EvaderRole


@dataclass
class TrajectoryData:
    """Stores trajectory data for visualization."""
    pursuer_positions: List[List[np.ndarray]]  # [pursuer_idx][timestep] -> position
    evader_positions: List[List[np.ndarray]]   # [evader_idx][timestep] -> position
    pursuer_states: List[List[PursuerState]]   # [pursuer_idx][timestep] -> state
    evader_roles: List[List[EvaderRole]]       # [evader_idx][timestep] -> role
    station_positions: List[np.ndarray]        # [station_idx] -> position
    captures: List[Tuple[int, int, int]]       # [(timestep, pursuer_idx, evader_idx), ...]
    escapes: List[Tuple[int, int]]             # [(timestep, evader_idx), ...]
    

class MultiAgentVisualizer:
    """Real-time and post-hoc visualization for multi-agent pursuit-evasion."""
    
    # Color schemes
    PURSUER_COLORS = ['#e41a1c', '#ff7f00', '#984ea3', '#a65628']  # Reds/oranges
    EVADER_COLORS = ['#377eb8', '#4daf4a', '#00ced1']  # Blues/greens
    STATION_COLOR = '#666666'
    ARENA_COLOR = '#f0f0f0'
    
    # State colors for pursuers
    STATE_COLORS = {
        PursuerState.GUARDING: '#4daf4a',      # Green
        PursuerState.INTERCEPTING: '#e41a1c',  # Red
        PursuerState.RETURNING: '#ff7f00',     # Orange
        PursuerState.RECHARGING: '#984ea3',    # Purple
    }
    
    # Role colors for evaders
    ROLE_COLORS = {
        EvaderRole.LEADER: '#377eb8',    # Blue
        EvaderRole.FOLLOWER: '#4daf4a',  # Green
        EvaderRole.DECOY: '#ff7f00',     # Orange
        EvaderRole.ESCAPED: '#00ff00',   # Bright green
        EvaderRole.CAPTURED: '#808080',  # Gray
    }
    
    def __init__(self, config: MultiAgentConfig):
        self.config = config
        self.fig = None
        self.ax = None
        
    def setup_figure(self, figsize: Tuple[int, int] = (12, 10)) -> Tuple[plt.Figure, plt.Axes]:
        """Create and setup the figure and axes."""
        self.fig, self.ax = plt.subplots(1, 1, figsize=figsize)
        self.ax.set_aspect('equal')
        self.ax.set_xlim(-self.config.arena.radius * 1.1, self.config.arena.radius * 1.1)
        self.ax.set_ylim(-self.config.arena.radius * 1.1, self.config.arena.radius * 1.1)
        self.ax.set_xlabel('X Position (m)', fontsize=12)
        self.ax.set_ylabel('Y Position (m)', fontsize=12)
        self.ax.grid(True, alpha=0.3)
        
        # Draw arena boundary
        arena_circle = plt.Circle((0, 0), self.config.arena.radius, 
                                   fill=False, color='black', linewidth=2, linestyle='--')
        self.ax.add_patch(arena_circle)
        
        return self.fig, self.ax
    
    def draw_stations(self, ax: plt.Axes, stations: List[np.ndarray]):
        """Draw station positions with guard radius."""
        for i, pos in enumerate(stations):
            # Station marker
            ax.plot(pos[0], pos[1], 's', markersize=15, color=self.STATION_COLOR,
                   markeredgecolor='black', markeredgewidth=2, zorder=5)
            ax.annotate(f'S{i+1}', (pos[0], pos[1]), textcoords="offset points",
                       xytext=(0, 12), ha='center', fontsize=10, fontweight='bold')
            
            # Station radius circle
            station_circle = plt.Circle(pos, self.config.station.station_radius,
                                      fill=True, color=self.STATION_COLOR, alpha=0.2)
            ax.add_patch(station_circle)
            station_border = plt.Circle(pos, self.config.station.station_radius,
                                      fill=False, color=self.STATION_COLOR, 
                                      linestyle='-', linewidth=2)
            ax.add_patch(station_border)
    
    def draw_pursuers(self, ax: plt.Axes, positions: List[np.ndarray], 
                      states: Optional[List[PursuerState]] = None,
                      show_radar: bool = True):
        """Draw pursuers with optional radar range and state coloring."""
        for i, pos in enumerate(positions):
            color = self.PURSUER_COLORS[i % len(self.PURSUER_COLORS)]
            if states:
                color = self.STATE_COLORS.get(states[i], color)
            
            # Pursuer marker (triangle pointing in direction)
            ax.plot(pos[0], pos[1], '^', markersize=12, color=color,
                   markeredgecolor='black', markeredgewidth=1.5, zorder=10)
            ax.annotate(f'P{i+1}', (pos[0], pos[1]), textcoords="offset points",
                       xytext=(8, 8), ha='left', fontsize=9)
            
            # Radar range
            if show_radar:
                radar_circle = plt.Circle(pos, self.config.pursuer.radar_range,
                                         fill=False, color=color, alpha=0.3, linestyle='-')
                ax.add_patch(radar_circle)
    
    def draw_evaders(self, ax: plt.Axes, positions: List[np.ndarray],
                     roles: Optional[List[EvaderRole]] = None,
                     show_detection: bool = False):
        """Draw evaders with role coloring."""
        for i, pos in enumerate(positions):
            color = self.EVADER_COLORS[i % len(self.EVADER_COLORS)]
            marker = 'o'
            size = 10
            
            if roles:
                role = roles[i]
                color = self.ROLE_COLORS.get(role, color)
                if role == EvaderRole.DECOY:
                    marker = 'D'  # Diamond for decoy
                elif role == EvaderRole.LEADER:
                    marker = '*'  # Star for leader
                    size = 14
                elif role == EvaderRole.CAPTURED:
                    marker = 'x'
                    size = 12
            
            ax.plot(pos[0], pos[1], marker, markersize=size, color=color,
                   markeredgecolor='black', markeredgewidth=1, zorder=10)
            ax.annotate(f'E{i+1}', (pos[0], pos[1]), textcoords="offset points",
                       xytext=(8, -8), ha='left', fontsize=9)
            
            # Detection range
            if show_detection and roles and roles[i] not in [EvaderRole.CAPTURED, EvaderRole.ESCAPED]:
                det_circle = plt.Circle(pos, self.config.evader.detection_range,
                                       fill=False, color=color, alpha=0.2, linestyle='--')
                ax.add_patch(det_circle)
    
    def draw_trajectories(self, ax: plt.Axes, trajectory: TrajectoryData,
                          fade: bool = True, line_width: float = 1.5):
        """Draw trajectory lines with optional fading."""
        # Pursuer trajectories
        for i, positions in enumerate(trajectory.pursuer_positions):
            if len(positions) < 2:
                continue
            color = self.PURSUER_COLORS[i % len(self.PURSUER_COLORS)]
            pts = np.array(positions)
            
            if fade:
                # Create segments with varying alpha
                segments = np.stack([pts[:-1], pts[1:]], axis=1)
                alphas = np.linspace(0.2, 1.0, len(segments))
                colors = [(color[0], color[1], color[2], a) if isinstance(color, tuple) 
                         else (*plt.cm.colors.to_rgb(color), a) for a in alphas]
                lc = LineCollection(segments, colors=colors, linewidth=line_width)
                ax.add_collection(lc)
            else:
                ax.plot(pts[:, 0], pts[:, 1], '-', color=color, 
                       linewidth=line_width, alpha=0.7)
        
        # Evader trajectories
        for i, positions in enumerate(trajectory.evader_positions):
            if len(positions) < 2:
                continue
            color = self.EVADER_COLORS[i % len(self.EVADER_COLORS)]
            pts = np.array(positions)
            
            if fade:
                segments = np.stack([pts[:-1], pts[1:]], axis=1)
                alphas = np.linspace(0.2, 1.0, len(segments))
                colors = [(*plt.cm.colors.to_rgb(color), a) for a in alphas]
                lc = LineCollection(segments, colors=colors, linewidth=line_width, linestyle='--')
                ax.add_collection(lc)
            else:
                ax.plot(pts[:, 0], pts[:, 1], '--', color=color,
                       linewidth=line_width, alpha=0.7)
    
    def draw_capture_markers(self, ax: plt.Axes, trajectory: TrajectoryData):
        """Mark capture and escape events."""
        for timestep, p_idx, e_idx in trajectory.captures:
            if timestep < len(trajectory.evader_positions[e_idx]):
                pos = trajectory.evader_positions[e_idx][timestep]
                ax.plot(pos[0], pos[1], 'X', markersize=15, color='red',
                       markeredgecolor='black', markeredgewidth=2, zorder=15)
        
        for timestep, e_idx in trajectory.escapes:
            if timestep < len(trajectory.evader_positions[e_idx]):
                pos = trajectory.evader_positions[e_idx][timestep]
                ax.plot(pos[0], pos[1], '*', markersize=20, color='lime',
                       markeredgecolor='black', markeredgewidth=2, zorder=15)
    
    def create_legend(self, ax: plt.Axes, show_states: bool = True, show_roles: bool = True):
        """Create a legend for the visualization."""
        legend_elements = []
        
        # Agent types
        legend_elements.append(plt.Line2D([0], [0], marker='^', color='w', 
                                          markerfacecolor='red', markersize=10,
                                          label='Pursuer'))
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                          markerfacecolor='blue', markersize=10,
                                          label='Evader'))
        legend_elements.append(plt.Line2D([0], [0], marker='s', color='w',
                                          markerfacecolor=self.STATION_COLOR, markersize=10,
                                          label='Station'))
        
        if show_states:
            legend_elements.append(plt.Line2D([0], [0], marker='^', color='w',
                                              markerfacecolor=self.STATE_COLORS[PursuerState.GUARDING],
                                              markersize=8, label='Guarding'))
            legend_elements.append(plt.Line2D([0], [0], marker='^', color='w',
                                              markerfacecolor=self.STATE_COLORS[PursuerState.INTERCEPTING],
                                              markersize=8, label='Intercepting'))
        
        if show_roles:
            legend_elements.append(plt.Line2D([0], [0], marker='*', color='w',
                                              markerfacecolor=self.ROLE_COLORS[EvaderRole.LEADER],
                                              markersize=12, label='Leader'))
            legend_elements.append(plt.Line2D([0], [0], marker='D', color='w',
                                              markerfacecolor=self.ROLE_COLORS[EvaderRole.DECOY],
                                              markersize=8, label='Decoy'))
        
        # Events
        legend_elements.append(plt.Line2D([0], [0], marker='X', color='w',
                                          markerfacecolor='red', markersize=12,
                                          label='Capture'))
        legend_elements.append(plt.Line2D([0], [0], marker='*', color='w',
                                          markerfacecolor='lime', markersize=14,
                                          label='Escape'))
        
        ax.legend(handles=legend_elements, loc='upper right', fontsize=9)


class LiveVisualizer(MultiAgentVisualizer):
    """Real-time visualization during simulation."""
    
    def __init__(self, config: MultiAgentConfig, update_interval: float = 0.05):
        super().__init__(config)
        self.update_interval = update_interval
        self.trajectory = None
        self.env = None
        
    def run_episode(self, env: MultiAgentPursuitEnv, 
                    pursuer_model: Optional[PPO] = None,
                    evader_model: Optional[PPO] = None,
                    save_path: Optional[str] = None):
        """Run and visualize a single episode."""
        self.env = env
        self.setup_figure()
        
        # Initialize trajectory storage
        n_pursuers = len(env.pursuers)
        n_evaders = len(env.evaders)
        self.trajectory = TrajectoryData(
            pursuer_positions=[[] for _ in range(n_pursuers)],
            evader_positions=[[] for _ in range(n_evaders)],
            pursuer_states=[[] for _ in range(n_pursuers)],
            evader_roles=[[] for _ in range(n_evaders)],
            station_positions=[s.position.copy() for s in env.stations],
            captures=[],
            escapes=[]
        )
        
        # Reset environment
        obs, info = env.reset()
        done = False
        step = 0
        
        # Animation function
        def animate(frame):
            nonlocal obs, done, step
            
            if done:
                return []
            
            self.ax.clear()
            self.ax.set_xlim(-self.config.arena.radius * 1.1, self.config.arena.radius * 1.1)
            self.ax.set_ylim(-self.config.arena.radius * 1.1, self.config.arena.radius * 1.1)
            self.ax.set_xlabel('X Position (m)', fontsize=12)
            self.ax.set_ylabel('Y Position (m)', fontsize=12)
            self.ax.set_title(f'Multi-Agent Pursuit-Evasion - Step {step}', fontsize=14)
            self.ax.grid(True, alpha=0.3)
            
            # Arena boundary
            arena_circle = plt.Circle((0, 0), self.config.arena.radius,
                                       fill=False, color='black', linewidth=2, linestyle='--')
            self.ax.add_patch(arena_circle)
            
            # Record current positions
            for i, p in enumerate(env.pursuers):
                self.trajectory.pursuer_positions[i].append(p.position.copy())
                self.trajectory.pursuer_states[i].append(p.state)
            for i, e in enumerate(env.evaders):
                self.trajectory.evader_positions[i].append(e.position.copy())
                self.trajectory.evader_roles[i].append(e.role)
            
            # Draw elements
            self.draw_stations(self.ax, self.trajectory.station_positions)
            self.draw_trajectories(self.ax, self.trajectory, fade=True, line_width=1.0)
            
            pursuer_positions = [p.position for p in env.pursuers]
            pursuer_states = [p.state for p in env.pursuers]
            evader_positions = [e.position for e in env.evaders]
            evader_roles = [e.role for e in env.evaders]
            
            self.draw_pursuers(self.ax, pursuer_positions, pursuer_states, show_radar=True)
            self.draw_evaders(self.ax, evader_positions, evader_roles, show_detection=True)
            self.create_legend(self.ax)
            
            # Get actions
            actions = {}
            
            # Pursuer actions
            if pursuer_model is not None:
                for i, p in enumerate(env.pursuers):
                    p_obs = env._get_pursuer_obs(i)
                    action, _ = pursuer_model.predict(p_obs, deterministic=True)
                    actions[f'pursuer_{i}'] = action
            else:
                for i in range(n_pursuers):
                    actions[f'pursuer_{i}'] = env.action_space[f'pursuer_{i}'].sample()
            
            # Evader actions
            if evader_model is not None:
                for i, e in enumerate(env.evaders):
                    if e.role not in [EvaderRole.CAPTURED, EvaderRole.ESCAPED]:
                        e_obs = env._get_evader_obs(i)
                        action, _ = evader_model.predict(e_obs, deterministic=True)
                        actions[f'evader_{i}'] = action
                    else:
                        actions[f'evader_{i}'] = np.zeros(2)
            else:
                for i in range(n_evaders):
                    actions[f'evader_{i}'] = env.action_space[f'evader_{i}'].sample()
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(actions)
            done = terminated or truncated
            step += 1
            
            # Record events
            for i, e in enumerate(env.evaders):
                if e.role == EvaderRole.CAPTURED:
                    if not any(c[2] == i for c in self.trajectory.captures):
                        self.trajectory.captures.append((step, -1, i))
                elif e.role == EvaderRole.ESCAPED:
                    if not any(esc[1] == i for esc in self.trajectory.escapes):
                        self.trajectory.escapes.append((step, i))
            
            # Status text
            active_evaders = sum(1 for e in env.evaders 
                                if e.role not in [EvaderRole.CAPTURED, EvaderRole.ESCAPED])
            captured = sum(1 for e in env.evaders if e.role == EvaderRole.CAPTURED)
            escaped = sum(1 for e in env.evaders if e.role == EvaderRole.ESCAPED)
            
            status = f'Active: {active_evaders} | Captured: {captured} | Escaped: {escaped}'
            self.ax.text(0.02, 0.98, status, transform=self.ax.transAxes,
                        fontsize=11, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            return []
        
        # Run animation
        anim = FuncAnimation(self.fig, animate, frames=self.config.arena.max_steps,
                            interval=self.update_interval * 1000, blit=False, repeat=False)
        
        if save_path:
            print(f"Saving animation to {save_path}...")
            anim.save(save_path, writer='pillow', fps=20)
            print("Animation saved!")
        else:
            plt.show()
        
        return self.trajectory


def plot_episode_summary(trajectory: TrajectoryData, config: MultiAgentConfig,
                         title: str = "Episode Summary", save_path: Optional[str] = None):
    """Create a static summary plot of an episode."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    viz = MultiAgentVisualizer(config)
    
    # Left: Full trajectory plot
    ax1 = axes[0]
    ax1.set_aspect('equal')
    ax1.set_xlim(-config.arena.radius * 1.1, config.arena.radius * 1.1)
    ax1.set_ylim(-config.arena.radius * 1.1, config.arena.radius * 1.1)
    ax1.set_xlabel('X Position (m)', fontsize=12)
    ax1.set_ylabel('Y Position (m)', fontsize=12)
    ax1.set_title(f'{title} - Trajectories', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # Arena
    arena_circle = plt.Circle((0, 0), config.arena.radius,
                               fill=False, color='black', linewidth=2, linestyle='--')
    ax1.add_patch(arena_circle)
    
    viz.draw_stations(ax1, trajectory.station_positions)
    viz.draw_trajectories(ax1, trajectory, fade=False, line_width=1.5)
    
    # Final positions
    if trajectory.pursuer_positions[0]:
        final_p_pos = [positions[-1] for positions in trajectory.pursuer_positions]
        final_p_states = [states[-1] if states else PursuerState.GUARDING 
                         for states in trajectory.pursuer_states]
        viz.draw_pursuers(ax1, final_p_pos, final_p_states, show_radar=False)
    
    if trajectory.evader_positions[0]:
        final_e_pos = [positions[-1] for positions in trajectory.evader_positions]
        final_e_roles = [roles[-1] if roles else EvaderRole.FOLLOWER 
                        for roles in trajectory.evader_roles]
        viz.draw_evaders(ax1, final_e_pos, final_e_roles)
    
    viz.draw_capture_markers(ax1, trajectory)
    viz.create_legend(ax1)
    
    # Right: Statistics
    ax2 = axes[1]
    ax2.axis('off')
    
    # Calculate statistics
    n_pursuers = len(trajectory.pursuer_positions)
    n_evaders = len(trajectory.evader_positions)
    episode_length = max(len(positions) for positions in trajectory.pursuer_positions)
    n_captures = len(trajectory.captures)
    n_escapes = len(trajectory.escapes)
    
    # Distance traveled
    def calc_distance(positions):
        if len(positions) < 2:
            return 0
        pts = np.array(positions)
        return np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))
    
    pursuer_distances = [calc_distance(positions) for positions in trajectory.pursuer_positions]
    evader_distances = [calc_distance(positions) for positions in trajectory.evader_positions]
    
    # Build stats text
    stats_text = f"""
    EPISODE STATISTICS
    {'='*40}
    
    Duration: {episode_length} steps
    
    PURSUERS ({n_pursuers} total)
    {'─'*30}
    """
    for i, dist in enumerate(pursuer_distances):
        stats_text += f"  P{i+1}: {dist:.1f}m traveled\n"
    
    stats_text += f"""
    EVADERS ({n_evaders} total)
    {'─'*30}
    """
    for i, dist in enumerate(evader_distances):
        final_role = trajectory.evader_roles[i][-1] if trajectory.evader_roles[i] else "Unknown"
        stats_text += f"  E{i+1}: {dist:.1f}m traveled ({final_role.name})\n"
    
    stats_text += f"""
    OUTCOMES
    {'─'*30}
    Captures: {n_captures}
    Escapes: {n_escapes}
    Pursuer Win: {'Yes' if n_captures == n_evaders else 'No'}
    """
    
    ax2.text(0.1, 0.95, stats_text, transform=ax2.transAxes,
             fontsize=11, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved episode summary to {save_path}")
    else:
        plt.show()
    
    return fig


def run_visualization(config: MultiAgentConfig,
                      pursuer_model_path: Optional[str] = None,
                      evader_model_path: Optional[str] = None,
                      num_episodes: int = 1,
                      save_dir: Optional[str] = None,
                      live: bool = True):
    """Run visualization with optional trained models."""
    
    # Load models
    pursuer_model = None
    evader_model = None
    
    if pursuer_model_path:
        print(f"Loading pursuer model from {pursuer_model_path}")
        pursuer_model = PPO.load(pursuer_model_path)
    
    if evader_model_path:
        print(f"Loading evader model from {evader_model_path}")
        evader_model = PPO.load(evader_model_path)
    
    # Create environment
    env = MultiAgentPursuitEnv(config)
    
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    for ep in range(num_episodes):
        print(f"\n=== Episode {ep + 1}/{num_episodes} ===")
        
        if live:
            # Live visualization
            viz = LiveVisualizer(config)
            save_path = f"{save_dir}/episode_{ep+1}.gif" if save_dir else None
            trajectory = viz.run_episode(env, pursuer_model, evader_model, save_path)
        else:
            # Run episode without live viz, then plot summary
            trajectory = collect_trajectory(env, config, pursuer_model, evader_model)
        
        # Save summary plot
        summary_path = f"{save_dir}/episode_{ep+1}_summary.png" if save_dir else None
        plot_episode_summary(trajectory, config, f"Episode {ep+1}", summary_path)
    
    env.close()


def collect_trajectory(env: MultiAgentPursuitEnv, config: MultiAgentConfig,
                       pursuer_model: Optional[PPO] = None,
                       evader_model: Optional[PPO] = None) -> TrajectoryData:
    """Collect trajectory data without visualization.
    
    Note: Models are trained with flattened observation/action spaces.
    - Pursuer model expects obs shape (n_pursuers * obs_dim,) and outputs (n_pursuers * 2,)
    - Evader model expects obs shape (n_evaders * obs_dim,) and outputs (n_evaders * 2,)
    """
    # Reset first to initialize agents
    obs, info = env.reset()
    
    n_pursuers = len(env.pursuers)
    n_evaders = len(env.evaders)
    
    trajectory = TrajectoryData(
        pursuer_positions=[[] for _ in range(n_pursuers)],
        evader_positions=[[] for _ in range(n_evaders)],
        pursuer_states=[[] for _ in range(n_pursuers)],
        evader_roles=[[] for _ in range(n_evaders)],
        station_positions=[s.position.copy() for s in env.stations],
        captures=[],
        escapes=[]
    )
    
    done = False
    step = 0
    
    while not done:
        # Record positions
        for i, p in enumerate(env.pursuers):
            trajectory.pursuer_positions[i].append(p.position.copy())
            trajectory.pursuer_states[i].append(p.state)
        for i, e in enumerate(env.evaders):
            trajectory.evader_positions[i].append(e.position.copy())
            trajectory.evader_roles[i].append(e.role)
        
        # Build actions dict for multi-env step
        actions = {'pursuers': None, 'evaders': None}
        
        # Extract observations
        pursuer_obs_all = obs['pursuers']  # Shape: (n_pursuers, obs_dim)
        evader_obs_all = obs['evaders']    # Shape: (n_evaders, obs_dim)
        
        if pursuer_model:
            # Model expects flattened obs and outputs flattened actions
            flat_obs = pursuer_obs_all.flatten()
            flat_action, _ = pursuer_model.predict(flat_obs, deterministic=True)
            actions['pursuers'] = flat_action.reshape(n_pursuers, 2)
        else:
            # Use heuristic - None tells env to use built-in heuristic
            actions['pursuers'] = None
        
        if evader_model:
            # Model expects flattened obs and outputs flattened actions
            flat_obs = evader_obs_all.flatten()
            flat_action, _ = evader_model.predict(flat_obs, deterministic=True)
            actions['evaders'] = flat_action.reshape(n_evaders, 2)
        else:
            # Use heuristic - None tells env to use built-in heuristic
            actions['evaders'] = None
        
        obs, reward, terminated, truncated, info = env.step(actions)
        done = terminated or truncated
        step += 1
        
        # Record events
        for i, e in enumerate(env.evaders):
            if e.role == EvaderRole.CAPTURED and not any(c[2] == i for c in trajectory.captures):
                trajectory.captures.append((step, -1, i))
            elif e.role == EvaderRole.ESCAPED and not any(esc[1] == i for esc in trajectory.escapes):
                trajectory.escapes.append((step, i))
    
    return trajectory


def main():
    parser = argparse.ArgumentParser(description='Visualize Multi-Agent Pursuit-Evasion')
    parser.add_argument('--pursuer-model', type=str, default=None,
                        help='Path to trained pursuer model')
    parser.add_argument('--evader-model', type=str, default=None,
                        help='Path to trained evader model')
    parser.add_argument('--episodes', type=int, default=1,
                        help='Number of episodes to visualize')
    parser.add_argument('--save-dir', type=str, default=None,
                        help='Directory to save visualizations')
    parser.add_argument('--no-live', action='store_true',
                        help='Disable live visualization (only save plots)')
    parser.add_argument('--max-steps', type=int, default=500,
                        help='Maximum steps per episode')
    
    args = parser.parse_args()
    
    # Get config
    config = get_default_multi_config()
    config.arena.max_steps = args.max_steps
    
    run_visualization(
        config=config,
        pursuer_model_path=args.pursuer_model,
        evader_model_path=args.evader_model,
        num_episodes=args.episodes,
        save_dir=args.save_dir,
        live=not args.no_live
    )


if __name__ == '__main__':
    main()
