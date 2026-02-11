#!/usr/bin/env python3
"""
Live Multi-Agent Pursuit-Evasion Simulation
============================================
Real-time visualization of the multi-agent pursuit-evasion scenario.

Usage:
    python live_simulation.py                     # Heuristic vs Heuristic
    python live_simulation.py --pursuers rl      # RL Pursuers vs Heuristic
    python live_simulation.py --evaders rl       # Heuristic vs RL Evaders
    python live_simulation.py --pursuers rl --evaders rl  # RL vs RL
    python live_simulation.py --speed 2.0        # 2x speed
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
import os
from pathlib import Path
from typing import Optional, List, Dict

from src.multi_config import get_default_multi_config, MultiAgentConfig
from src.multi_environment import MultiAgentPursuitEnv
from src.multi_agents import PursuerState, EvaderRole


class LiveSimulation:
    """Real-time simulation viewer."""
    
    # Color scheme
    COLORS = {
        'arena': '#1a1a2e',
        'arena_border': '#4a4e69',
        'station': '#ffd166',
        'station_area': '#ffd16633',
        'pursuer': ['#ef476f', '#118ab2', '#06d6a0', '#073b4c'],
        'evader': ['#8338ec', '#fb5607', '#3a86ff'],
        'radar': '#ffffff22',
        'trail': 0.3,
        'capture': '#ff0000',
        'escape': '#00ff00',
        'goal': '#00ff0044',
    }
    
    STATE_COLORS = {
        PursuerState.GUARDING: '#ffd166',
        PursuerState.INTERCEPTING: '#ef476f',
        PursuerState.RETURNING: '#118ab2',
        PursuerState.RECHARGING: '#aaaaaa',
    }
    
    def __init__(self, config: MultiAgentConfig,
                 pursuer_model=None, evader_model=None,
                 speed: float = 1.0, trail_length: int = 50):
        self.config = config
        self.pursuer_model = pursuer_model
        self.evader_model = evader_model
        self.speed = speed
        self.trail_length = trail_length
        
        # Create environment
        self.env = MultiAgentPursuitEnv(config)
        self.obs, _ = self.env.reset()
        
        # Trail history
        self.pursuer_trails: List[List[np.ndarray]] = [[] for _ in range(len(self.env.pursuers))]
        self.evader_trails: List[List[np.ndarray]] = [[] for _ in range(len(self.env.evaders))]
        
        # Statistics
        self.step_count = 0
        self.captures = []  # Evader captures
        self.escapes = []   # Evader escapes
        self.station_captures = []  # Station captures
        self.done = False
        
        # Setup figure
        self._setup_figure()
        
    def _setup_figure(self):
        """Setup matplotlib figure and axes."""
        plt.style.use('dark_background')
        self.fig, self.ax = plt.subplots(1, 1, figsize=(12, 12))
        self.fig.patch.set_facecolor(self.COLORS['arena'])
        self.ax.set_facecolor(self.COLORS['arena'])
        
        # Set limits
        r = self.config.arena.radius * 1.1
        self.ax.set_xlim(-r, r)
        self.ax.set_ylim(-r, r)
        self.ax.set_aspect('equal')
        self.ax.axis('off')
        
        # Title
        self.title = self.ax.set_title('', fontsize=14, color='white', pad=10)
        
        # Draw static elements
        self._draw_arena()
        self._create_station_artists()  # Dynamic stations now
        self._draw_goal()
        
        # Create dynamic elements
        self._create_agent_artists()
        self._create_legend()
        
    def _draw_arena(self):
        """Draw arena boundary."""
        arena = plt.Circle((0, 0), self.config.arena.radius,
                          fill=False, color=self.COLORS['arena_border'],
                          linewidth=3, linestyle='--')
        self.ax.add_patch(arena)
        
        # Grid lines
        for r in [self.config.arena.radius * 0.25, 
                  self.config.arena.radius * 0.5,
                  self.config.arena.radius * 0.75]:
            circle = plt.Circle((0, 0), r, fill=False,
                               color='#ffffff11', linewidth=1)
            self.ax.add_patch(circle)
            
    def _create_station_artists(self):
        """Create dynamic station artists (stations can be captured)."""
        self.station_areas = []
        self.station_markers = []
        self.station_progress = []
        self.station_labels = []
        
        for i, station in enumerate(self.env.stations):
            pos = station.position
            
            # Station capture zone (circle that changes color)
            area = plt.Circle(pos, self.config.station.station_radius,
                            fill=True, color=self.COLORS['station_area'], zorder=2)
            self.ax.add_patch(area)
            self.station_areas.append(area)
            
            # Station marker
            marker, = self.ax.plot(pos[0], pos[1], 's', markersize=18,
                        color=self.COLORS['station'],
                        markeredgecolor='black', markeredgewidth=2, zorder=10)
            self.station_markers.append(marker)
            
            # Capture progress ring
            progress_ring = plt.Circle(pos, self.config.station.station_radius + 1,
                                      fill=False, color='#ff0000', linewidth=3, zorder=9)
            self.ax.add_patch(progress_ring)
            self.station_progress.append(progress_ring)
            
            # Label
            label = self.ax.annotate(f'S{i+1}', pos, textcoords="offset points",
                           xytext=(0, -25), ha='center', fontsize=11,
                           color='white', fontweight='bold')
            self.station_labels.append(label)
                           
    def _draw_goal(self):
        """Goals are now stations - no separate goal zone needed."""
        # Stations ARE the goals - evaders try to capture them
        pass
    
    def _create_agent_artists(self):
        """Create matplotlib artists for agents."""
        # Pursuer artists
        self.pursuer_markers = []
        self.pursuer_radar = []
        self.pursuer_trails_art = []
        
        for i in range(len(self.env.pursuers)):
            color = self.COLORS['pursuer'][i % len(self.COLORS['pursuer'])]
            
            # Marker (triangle)
            marker, = self.ax.plot([], [], '^', markersize=15, color=color,
                                  markeredgecolor='white', markeredgewidth=2, zorder=20)
            self.pursuer_markers.append(marker)
            
            # Radar range
            radar = plt.Circle((0, 0), self.config.pursuer.radar_range,
                              fill=True, color=self.COLORS['radar'], zorder=5)
            self.ax.add_patch(radar)
            self.pursuer_radar.append(radar)
            
            # Trail
            trail, = self.ax.plot([], [], '-', color=color, alpha=0.4, linewidth=2, zorder=3)
            self.pursuer_trails_art.append(trail)
            
        # Evader artists
        self.evader_markers = []
        self.evader_trails_art = []
        
        for i in range(len(self.env.evaders)):
            color = self.COLORS['evader'][i % len(self.COLORS['evader'])]
            
            # Marker (circle)
            marker, = self.ax.plot([], [], 'o', markersize=12, color=color,
                                  markeredgecolor='white', markeredgewidth=2, zorder=20)
            self.evader_markers.append(marker)
            
            # Trail
            trail, = self.ax.plot([], [], '-', color=color, alpha=0.4, linewidth=2, zorder=3)
            self.evader_trails_art.append(trail)
            
        # Status text
        self.status_text = self.ax.text(0.02, 0.98, '', transform=self.ax.transAxes,
                                       fontsize=11, color='white', verticalalignment='top',
                                       fontfamily='monospace',
                                       bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
                                       
    def _create_legend(self):
        """Create legend."""
        legend_elements = [
            Line2D([0], [0], marker='^', color='w', markerfacecolor=self.COLORS['pursuer'][0],
                  markersize=10, label='Pursuers', linestyle='None'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=self.COLORS['evader'][0],
                  markersize=10, label='Evaders', linestyle='None'),
            Line2D([0], [0], marker='s', color='w', markerfacecolor=self.COLORS['station'],
                  markersize=10, label='Stations', linestyle='None'),
            patches.Patch(facecolor=self.COLORS['goal'], label='Goal Zone'),
        ]
        
        self.ax.legend(handles=legend_elements, loc='lower right',
                      facecolor='black', edgecolor='white', fontsize=10)
                      
    def _get_actions(self) -> Dict:
        """Get actions from models or heuristics."""
        n_pursuers = len(self.env.pursuers)
        n_evaders = len(self.env.evaders)
        
        actions = {'pursuers': None, 'evaders': None}
        
        if self.pursuer_model:
            flat_obs = self.obs['pursuers'].flatten()
            flat_action, _ = self.pursuer_model.predict(flat_obs, deterministic=True)
            actions['pursuers'] = flat_action.reshape(n_pursuers, 2)
        
        if self.evader_model:
            flat_obs = self.obs['evaders'].flatten()
            flat_action, _ = self.evader_model.predict(flat_obs, deterministic=True)
            actions['evaders'] = flat_action.reshape(n_evaders, 2)
            
        return actions
        
    def _update_trails(self):
        """Update trail histories."""
        for i, p in enumerate(self.env.pursuers):
            self.pursuer_trails[i].append(p.position.copy())
            if len(self.pursuer_trails[i]) > self.trail_length:
                self.pursuer_trails[i].pop(0)
                
        for i, e in enumerate(self.env.evaders):
            self.evader_trails[i].append(e.position.copy())
            if len(self.evader_trails[i]) > self.trail_length:
                self.evader_trails[i].pop(0)
                
    def _check_events(self):
        """Check for captures, escapes, and station captures."""
        for i, e in enumerate(self.env.evaders):
            if e.role == EvaderRole.CAPTURED and i not in [c[1] for c in self.captures]:
                self.captures.append((self.step_count, i))
            elif e.role == EvaderRole.ESCAPED and i not in [esc[1] for esc in self.escapes]:
                self.escapes.append((self.step_count, i))
        
        # Check station captures
        for i, station in enumerate(self.env.stations):
            if not station.is_active and i not in [sc[1] for sc in self.station_captures]:
                self.station_captures.append((self.step_count, i))
                
    def update(self, frame):
        """Animation update function."""
        if self.done:
            return []
            
        # Step environment
        actions = self._get_actions()
        self.obs, reward, terminated, truncated, info = self.env.step(actions)
        self.step_count += 1
        self.done = terminated or truncated
        
        # Update trails
        self._update_trails()
        self._check_events()
        
        # Update pursuer artists
        for i, p in enumerate(self.env.pursuers):
            self.pursuer_markers[i].set_data([p.position[0]], [p.position[1]])
            
            # Update color based on state
            color = self.STATE_COLORS.get(p.state, self.COLORS['pursuer'][i])
            self.pursuer_markers[i].set_color(color)
            
            # Update radar
            self.pursuer_radar[i].center = p.position
            
            # Update trail
            if self.pursuer_trails[i]:
                trail = np.array(self.pursuer_trails[i])
                self.pursuer_trails_art[i].set_data(trail[:, 0], trail[:, 1])
                
        # Update evader artists
        for i, e in enumerate(self.env.evaders):
            if e.role == EvaderRole.CAPTURED:
                self.evader_markers[i].set_visible(False)
            elif e.role == EvaderRole.ESCAPED:
                self.evader_markers[i].set_color(self.COLORS['escape'])
                self.evader_markers[i].set_data([e.position[0]], [e.position[1]])
            else:
                self.evader_markers[i].set_data([e.position[0]], [e.position[1]])
                
            # Update trail
            if self.evader_trails[i]:
                trail = np.array(self.evader_trails[i])
                self.evader_trails_art[i].set_data(trail[:, 0], trail[:, 1])
        
        # Update station artists (capture progress visualization)
        for i, station in enumerate(self.env.stations):
            # Update station area color based on capture state
            if not station.is_active:
                # Station captured - show as red
                self.station_areas[i].set_facecolor((1.0, 0.2, 0.2, 0.4))
                self.station_areas[i].set_edgecolor('darkred')
                self.station_markers[i].set_color('darkred')
                self.station_progress[i].set_visible(False)
            elif station.capture_progress > 0:
                # Being captured - blend from green to yellow to red
                progress = station.capture_progress / self.config.station.capture_time
                r = min(1.0, progress * 2)
                g = min(1.0, 2.0 - progress * 2) if progress > 0.5 else 1.0
                self.station_areas[i].set_facecolor((r, g, 0.2, 0.4))
                
                # Update progress ring color
                self.station_progress[i].set_edgecolor((1.0, progress, 0.0))
                self.station_progress[i].set_visible(True)
            else:
                # Normal active station
                self.station_areas[i].set_facecolor((0.2, 0.8, 0.2, 0.3))
                self.station_areas[i].set_edgecolor('green')
                self.station_markers[i].set_color(self.COLORS['station'])
                self.station_progress[i].set_visible(False)
                
        # Update title
        p_type = "RL" if self.pursuer_model else "Heuristic"
        e_type = "RL" if self.evader_model else "Heuristic"
        self.title.set_text(f'{p_type} Pursuers vs {e_type} Evaders')
        
        # Update status
        active_evaders = sum(1 for e in self.env.evaders 
                           if e.role not in [EvaderRole.CAPTURED, EvaderRole.ESCAPED])
        active_stations = sum(1 for s in self.env.stations if s.is_active)
        
        status_lines = [
            f"Step: {self.step_count}/{self.config.arena.max_steps}",
            f"Evader Captures: {len(self.captures)}",
            f"Station Captures: {len(self.station_captures)}",
            f"Active Evaders: {active_evaders}",
            f"Active Stations: {active_stations}/{len(self.env.stations)}",
            "",
            "Station Status:",
        ]
        
        for i, station in enumerate(self.env.stations):
            if not station.is_active:
                status_lines.append(f"  S{i+1}: CAPTURED")
            elif station.capture_progress > 0:
                prog = station.capture_progress / self.config.station.capture_time * 100
                status_lines.append(f"  S{i+1}: Capturing {prog:.0f}%")
            else:
                status_lines.append(f"  S{i+1}: Active")
        
        status_lines.append("")
        status_lines.append("Pursuer States:")
        
        for i, p in enumerate(self.env.pursuers):
            state_name = p.state.name if hasattr(p.state, 'name') else str(p.state)
            status_lines.append(f"  P{i+1}: {state_name}")
            
        self.status_text.set_text('\n'.join(status_lines))
        
        # Check if done
        if self.done:
            result = "DRAW"
            active_stations = sum(1 for s in self.env.stations if s.is_active)
            if active_stations == 0:
                result = "EVADERS WIN! (All stations captured)"
            elif len(self.captures) == len(self.env.evaders):
                result = "PURSUERS WIN! (All evaders captured)"
            elif len(self.station_captures) > 0 and len(self.captures) > 0:
                result = f"PARTIAL - {len(self.station_captures)} stations lost"
            self.title.set_text(f'{p_type} vs {e_type} - {result}')
            
        return (self.pursuer_markers + self.evader_markers + 
                self.pursuer_trails_art + self.evader_trails_art +
                self.station_areas + self.station_progress +
                [self.title, self.status_text])
                
    def reset(self):
        """Reset simulation."""
        self.obs, _ = self.env.reset()
        self.step_count = 0
        self.captures = []
        self.escapes = []
        self.station_captures = []
        self.done = False
        self.pursuer_trails = [[] for _ in range(len(self.env.pursuers))]
        self.evader_trails = [[] for _ in range(len(self.env.evaders))]
        
        # Reset evader visibility
        for marker in self.evader_markers:
            marker.set_visible(True)
        
        # Reset station visuals
        for i, station in enumerate(self.env.stations):
            self.station_areas[i].set_facecolor((0.2, 0.8, 0.2, 0.3))
            self.station_areas[i].set_edgecolor('green')
            self.station_markers[i].set_color(self.COLORS['station'])
            self.station_progress[i].set_visible(False)
            
    def run(self):
        """Run the live simulation."""
        interval = int(100 / self.speed)  # milliseconds between frames
        
        self.ani = FuncAnimation(
            self.fig, self.update,
            frames=self.config.arena.max_steps,
            interval=interval,
            blit=False,
            repeat=False
        )
        
        # Add key handler for restart
        def on_key(event):
            if event.key == 'r':
                self.reset()
            elif event.key == 'q':
                plt.close()
                
        self.fig.canvas.mpl_connect('key_press_event', on_key)
        
        plt.tight_layout()
        plt.show()
        
    def close(self):
        """Clean up."""
        self.env.close()
        plt.close()


def find_latest_model(model_type: str) -> Optional[str]:
    """Find the latest trained model of the given type."""
    models_dir = Path('models')
    if not models_dir.exists():
        return None
        
    pattern = f'*_{model_type}_*'
    matching_dirs = sorted(models_dir.glob(pattern))
    
    if not matching_dirs:
        return None
    
    # Find the latest directory that has a valid model
    for latest in reversed(matching_dirs):
        model_path = latest / 'best' / 'best_model'
        if model_path.with_suffix('.zip').exists():
            return str(model_path)
    
    return None


def main():
    parser = argparse.ArgumentParser(description='Live Multi-Agent Pursuit-Evasion Simulation')
    parser.add_argument('--pursuers', type=str, choices=['rl', 'heuristic'], default='heuristic',
                       help='Pursuer control type')
    parser.add_argument('--evaders', type=str, choices=['rl', 'heuristic'], default='heuristic',
                       help='Evader control type')
    parser.add_argument('--pursuer-model', type=str, default=None,
                       help='Path to pursuer model (auto-detected if not specified)')
    parser.add_argument('--evader-model', type=str, default=None,
                       help='Path to evader model (auto-detected if not specified)')
    parser.add_argument('--speed', type=float, default=1.0,
                       help='Simulation speed multiplier')
    parser.add_argument('--max-steps', type=int, default=500,
                       help='Maximum episode steps')
    parser.add_argument('--trail-length', type=int, default=50,
                       help='Trail history length')
    args = parser.parse_args()
    
    # Load config
    config = get_default_multi_config()
    config.arena.max_steps = args.max_steps
    
    # Load models
    pursuer_model = None
    evader_model = None
    
    if args.pursuers == 'rl':
        model_path = args.pursuer_model or find_latest_model('pursuers')
        if model_path:
            from stable_baselines3 import PPO
            pursuer_model = PPO.load(model_path)
            print(f"✓ Loaded pursuer model: {model_path}")
        else:
            print("⚠ No pursuer model found, using heuristic")
            
    if args.evaders == 'rl':
        model_path = args.evader_model or find_latest_model('evaders')
        if model_path:
            from stable_baselines3 import PPO
            evader_model = PPO.load(model_path)
            print(f"✓ Loaded evader model: {model_path}")
        else:
            print("⚠ No evader model found, using heuristic")
    
    # Print controls
    print("\n" + "="*50)
    print("Controls:")
    print("  R - Reset simulation")
    print("  Q - Quit")
    print("="*50 + "\n")
    
    # Run simulation
    sim = LiveSimulation(
        config=config,
        pursuer_model=pursuer_model,
        evader_model=evader_model,
        speed=args.speed,
        trail_length=args.trail_length
    )
    
    try:
        sim.run()
    finally:
        sim.close()


if __name__ == '__main__':
    main()
