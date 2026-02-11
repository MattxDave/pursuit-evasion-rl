"""
Multi-Agent Simulation & Visualization
======================================
Run and visualize multi-agent pursuit-evasion scenarios.
"""

import argparse
import time
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrow
from matplotlib.lines import Line2D
import matplotlib.animation as animation

from stable_baselines3 import PPO

from src.multi_config import MultiAgentConfig, get_default_multi_config
from src.multi_environment import (
    MultiAgentPursuitEnv,
    PursuerTrainingEnv,
    EvaderTrainingEnv,
)
from src.multi_agents import PursuerState, EvaderRole


class MultiAgentSimulator:
    """Simulator for multi-agent pursuit-evasion."""
    
    def __init__(
        self,
        config: MultiAgentConfig,
        pursuer_model_path: Optional[str] = None,
        evader_model_path: Optional[str] = None,
    ):
        self.config = config
        self.env = MultiAgentPursuitEnv(
            config=config,
            train_pursuers=pursuer_model_path is not None,
            train_evaders=evader_model_path is not None,
            render_mode="human",
        )
        
        # Load models if provided
        self.pursuer_model = None
        self.evader_model = None
        
        if pursuer_model_path:
            self.pursuer_model = PPO.load(pursuer_model_path)
            print(f"Loaded pursuer model from {pursuer_model_path}")
        
        if evader_model_path:
            self.evader_model = PPO.load(evader_model_path)
            print(f"Loaded evader model from {evader_model_path}")
    
    def run_episode(
        self,
        render: bool = True,
        render_delay: float = 0.05,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Run a single episode and return statistics."""
        obs, info = self.env.reset()
        
        episode_stats = {
            'steps': 0,
            'captures': [],
            'goals_reached': [],
            'pursuer_rewards': 0.0,
            'evader_rewards': 0.0,
            'final_active_evaders': 0,
        }
        
        done = False
        
        while not done:
            # Get actions
            pursuer_actions = None
            evader_actions = None
            
            if self.pursuer_model:
                pursuer_obs = obs['pursuers'].flatten()
                pursuer_actions, _ = self.pursuer_model.predict(pursuer_obs, deterministic=True)
                pursuer_actions = pursuer_actions.reshape(self.env.num_pursuers, 2)
            
            if self.evader_model:
                evader_obs = obs['evaders'].flatten()
                evader_actions, _ = self.evader_model.predict(evader_obs, deterministic=True)
                # Evader actions are now 3-dim: [throttle, heading_adj, decoy_signal]
                evader_actions = evader_actions.reshape(self.env.num_evaders, 3)
            
            action = {
                'pursuers': pursuer_actions,
                'evaders': evader_actions,
            }
            
            obs, rewards, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            episode_stats['steps'] += 1
            episode_stats['pursuer_rewards'] += float(rewards['pursuers'].sum())
            episode_stats['evader_rewards'] += float(rewards['evaders'].sum())
            
            if info.get('captured'):
                episode_stats['captures'].extend(info['captured'])
            if info.get('reached_goal'):
                episode_stats['goals_reached'].extend(info['reached_goal'])
            
            if render:
                self.env.render()
                time.sleep(render_delay)
        
        episode_stats['final_active_evaders'] = info.get('num_active_evaders', 0)
        episode_stats['num_captured'] = info.get('num_captured', 0)
        episode_stats['num_at_goal'] = info.get('num_at_goal', 0)
        
        if verbose:
            print(f"\n--- Episode Complete ---")
            print(f"Steps: {episode_stats['steps']}")
            print(f"Captures: {episode_stats['num_captured']}")
            print(f"Goals Reached: {episode_stats['num_at_goal']}")
            print(f"Pursuer Reward: {episode_stats['pursuer_rewards']:.2f}")
            print(f"Evader Reward: {episode_stats['evader_rewards']:.2f}")
        
        return episode_stats
    
    def run_batch(
        self,
        num_episodes: int,
        render: bool = False,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Run multiple episodes and aggregate statistics."""
        all_stats = []
        
        for i in range(num_episodes):
            if verbose:
                print(f"\nEpisode {i + 1}/{num_episodes}")
            
            stats = self.run_episode(render=render, verbose=verbose)
            all_stats.append(stats)
        
        # Aggregate
        agg = {
            'num_episodes': num_episodes,
            'avg_steps': np.mean([s['steps'] for s in all_stats]),
            'avg_captures': np.mean([s['num_captured'] for s in all_stats]),
            'avg_goals': np.mean([s['num_at_goal'] for s in all_stats]),
            'capture_rate': np.mean([s['num_captured'] / self.config.evader.num_evaders for s in all_stats]),
            'goal_rate': np.mean([s['num_at_goal'] / self.config.evader.num_evaders for s in all_stats]),
            'avg_pursuer_reward': np.mean([s['pursuer_rewards'] for s in all_stats]),
            'avg_evader_reward': np.mean([s['evader_rewards'] for s in all_stats]),
        }
        
        if verbose:
            print(f"\n{'='*50}")
            print(f"Batch Results ({num_episodes} episodes)")
            print(f"{'='*50}")
            print(f"Avg Steps: {agg['avg_steps']:.1f}")
            print(f"Avg Captures: {agg['avg_captures']:.2f}")
            print(f"Avg Goals Reached: {agg['avg_goals']:.2f}")
            print(f"Capture Rate: {agg['capture_rate']*100:.1f}%")
            print(f"Goal Rate: {agg['goal_rate']*100:.1f}%")
            print(f"Avg Pursuer Reward: {agg['avg_pursuer_reward']:.2f}")
            print(f"Avg Evader Reward: {agg['avg_evader_reward']:.2f}")
        
        return agg
    
    def close(self):
        self.env.close()


def create_trajectory_animation(
    config: MultiAgentConfig,
    pursuer_model_path: Optional[str] = None,
    evader_model_path: Optional[str] = None,
    save_path: Optional[str] = None,
    fps: int = 20,
) -> None:
    """Create an animated visualization of an episode."""
    
    env = MultiAgentPursuitEnv(
        config=config,
        train_pursuers=pursuer_model_path is not None,
        train_evaders=evader_model_path is not None,
    )
    
    pursuer_model = PPO.load(pursuer_model_path) if pursuer_model_path else None
    evader_model = PPO.load(evader_model_path) if evader_model_path else None
    
    # Collect trajectory data
    obs, info = env.reset()
    
    trajectory_data = {
        'pursuer_positions': [[] for _ in range(env.num_pursuers)],
        'evader_positions': [[] for _ in range(env.num_evaders)],
        'pursuer_states': [[] for _ in range(env.num_pursuers)],
        'evader_roles': [[] for _ in range(env.num_evaders)],
        'stations': [s.position.copy() for s in env.stations],
        'goal': env.evader_goal.copy(),
    }
    
    done = False
    while not done:
        # Record positions
        for i, p in enumerate(env.pursuers):
            trajectory_data['pursuer_positions'][i].append(p.position.copy())
            trajectory_data['pursuer_states'][i].append(p.state)
        
        for i, e in enumerate(env.evaders):
            trajectory_data['evader_positions'][i].append(e.position.copy())
            trajectory_data['evader_roles'][i].append(e.role)
        
        # Get actions
        pursuer_actions = None
        evader_actions = None
        
        if pursuer_model:
            pursuer_obs = obs['pursuers'].flatten()
            pursuer_actions, _ = pursuer_model.predict(pursuer_obs, deterministic=True)
            pursuer_actions = pursuer_actions.reshape(env.num_pursuers, 2)
        
        if evader_model:
            evader_obs = obs['evaders'].flatten()
            evader_actions, _ = evader_model.predict(evader_obs, deterministic=True)
            evader_actions = evader_actions.reshape(env.num_evaders, 2)
        
        obs, _, terminated, truncated, _ = env.step({
            'pursuers': pursuer_actions,
            'evaders': evader_actions,
        })
        done = terminated or truncated
    
    env.close()
    
    # Create animation
    fig, ax = plt.subplots(figsize=(12, 12))
    R = config.arena.radius
    
    def init():
        ax.clear()
        ax.set_xlim(-R * 1.1, R * 1.1)
        ax.set_ylim(-R * 1.1, R * 1.1)
        ax.set_aspect('equal')
        return []
    
    def animate(frame):
        ax.clear()
        
        # Arena
        arena = Circle((0, 0), R, fill=False, color='black', linewidth=2)
        ax.add_patch(arena)
        
        # Stations
        for pos in trajectory_data['stations']:
            station = Circle(pos, config.station.station_radius, fill=True, color='blue', alpha=0.3)
            ax.add_patch(station)
            ax.plot(*pos, 'bs', markersize=10)
        
        # Goal
        goal = Circle(trajectory_data['goal'], config.arena.goal_radius, fill=True, color='green', alpha=0.3)
        ax.add_patch(goal)
        ax.plot(*trajectory_data['goal'], 'g*', markersize=15)
        
        # Pursuers
        for i in range(len(trajectory_data['pursuer_positions'])):
            positions = trajectory_data['pursuer_positions'][i]
            if frame < len(positions):
                pos = positions[frame]
                state = trajectory_data['pursuer_states'][i][frame]
                
                color = {
                    PursuerState.GUARDING: 'blue',
                    PursuerState.INTERCEPTING: 'red',
                    PursuerState.RETURNING: 'orange',
                    PursuerState.RECHARGING: 'cyan',
                }.get(state, 'blue')
                
                ax.plot(*pos, 'o', color=color, markersize=12)
                
                # Trail
                if frame > 0:
                    trail = np.array(positions[:frame+1])
                    ax.plot(trail[:, 0], trail[:, 1], '-', color=color, alpha=0.3, linewidth=1)
        
        # Evaders
        for i in range(len(trajectory_data['evader_positions'])):
            positions = trajectory_data['evader_positions'][i]
            if frame < len(positions):
                pos = positions[frame]
                role = trajectory_data['evader_roles'][i][frame]
                
                if role == EvaderRole.CAPTURED:
                    continue
                
                color = {
                    EvaderRole.LEADER: 'darkgreen',
                    EvaderRole.FOLLOWER: 'lime',
                    EvaderRole.DECOY: 'yellow',
                    EvaderRole.ESCAPED: 'cyan',
                }.get(role, 'green')
                
                marker = '^' if role == EvaderRole.LEADER else 'v'
                ax.plot(*pos, marker, color=color, markersize=10)
                
                # Trail
                if frame > 0:
                    trail = np.array(positions[:frame+1])
                    ax.plot(trail[:, 0], trail[:, 1], '-', color=color, alpha=0.3, linewidth=1)
        
        ax.set_xlim(-R * 1.1, R * 1.1)
        ax.set_ylim(-R * 1.1, R * 1.1)
        ax.set_aspect('equal')
        ax.set_title(f'Step: {frame}')
        
        return []
    
    num_frames = max(len(p) for p in trajectory_data['pursuer_positions'])
    anim = animation.FuncAnimation(
        fig, animate, init_func=init, frames=num_frames,
        interval=1000/fps, blit=True
    )
    
    if save_path:
        anim.save(save_path, writer='pillow', fps=fps)
        print(f"Animation saved to {save_path}")
    else:
        plt.show()
    
    plt.close(fig)


def plot_trajectories(
    config: MultiAgentConfig,
    pursuer_model_path: Optional[str] = None,
    evader_model_path: Optional[str] = None,
    save_path: Optional[str] = None,
) -> None:
    """Create a static plot of all trajectories."""
    
    env = MultiAgentPursuitEnv(
        config=config,
        train_pursuers=pursuer_model_path is not None,
        train_evaders=evader_model_path is not None,
    )
    
    pursuer_model = PPO.load(pursuer_model_path) if pursuer_model_path else None
    evader_model = PPO.load(evader_model_path) if evader_model_path else None
    
    obs, info = env.reset()
    
    pursuer_trajectories = [[] for _ in range(env.num_pursuers)]
    evader_trajectories = [[] for _ in range(env.num_evaders)]
    
    done = False
    while not done:
        for i, p in enumerate(env.pursuers):
            pursuer_trajectories[i].append(p.position.copy())
        for i, e in enumerate(env.evaders):
            evader_trajectories[i].append(e.position.copy())
        
        pursuer_actions = None
        evader_actions = None
        
        if pursuer_model:
            pursuer_obs = obs['pursuers'].flatten()
            pursuer_actions, _ = pursuer_model.predict(pursuer_obs, deterministic=True)
            pursuer_actions = pursuer_actions.reshape(env.num_pursuers, 2)
        
        if evader_model:
            evader_obs = obs['evaders'].flatten()
            evader_actions, _ = evader_model.predict(evader_obs, deterministic=True)
            evader_actions = evader_actions.reshape(env.num_evaders, 2)
        
        obs, _, terminated, truncated, _ = env.step({
            'pursuers': pursuer_actions,
            'evaders': evader_actions,
        })
        done = terminated or truncated
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 12))
    R = config.arena.radius
    
    # Arena
    arena = Circle((0, 0), R, fill=False, color='black', linewidth=2)
    ax.add_patch(arena)
    
    # Stations
    for station in env.stations:
        s = Circle(station.position, config.station.station_radius, fill=True, color='blue', alpha=0.2)
        ax.add_patch(s)
        ax.plot(*station.position, 'bs', markersize=12)
    
    # Goal (use station radius as goal size in multi-agent setup)
    goal = Circle(env.evader_goal, config.station.station_radius, fill=True, color='green', alpha=0.2)
    ax.add_patch(goal)
    ax.plot(*env.evader_goal, 'g*', markersize=20)
    
    # Pursuer trajectories
    colors_p = plt.cm.Blues(np.linspace(0.4, 0.9, env.num_pursuers))
    for i, traj in enumerate(pursuer_trajectories):
        if traj:
            traj = np.array(traj)
            ax.plot(traj[:, 0], traj[:, 1], '-', color=colors_p[i], linewidth=2, label=f'Pursuer {i}')
            ax.plot(*traj[0], 'o', color=colors_p[i], markersize=10)  # Start
            ax.plot(*traj[-1], 's', color=colors_p[i], markersize=10)  # End
    
    # Evader trajectories
    colors_e = plt.cm.Greens(np.linspace(0.4, 0.9, env.num_evaders))
    for i, traj in enumerate(evader_trajectories):
        if traj:
            traj = np.array(traj)
            ax.plot(traj[:, 0], traj[:, 1], '-', color=colors_e[i], linewidth=2, label=f'Evader {i}')
            ax.plot(*traj[0], '^', color=colors_e[i], markersize=10)  # Start
            ax.plot(*traj[-1], 'v', color=colors_e[i], markersize=10)  # End
    
    ax.set_xlim(-R * 1.1, R * 1.1)
    ax.set_ylim(-R * 1.1, R * 1.1)
    ax.set_aspect('equal')
    ax.legend(loc='upper right')
    ax.set_title('Multi-Agent Pursuit-Evasion Trajectories')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close(fig)
    env.close()


def main():
    parser = argparse.ArgumentParser(description="Simulate multi-agent pursuit-evasion")
    
    parser.add_argument("--pursuer-model", type=str, default=None, help="Path to pursuer model")
    parser.add_argument("--evader-model", type=str, default=None, help="Path to evader model")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes")
    parser.add_argument("--no-render", action="store_true", help="Disable rendering")
    parser.add_argument("--save-animation", type=str, default=None, help="Save animation to file")
    parser.add_argument("--save-plot", type=str, default=None, help="Save trajectory plot to file")
    
    # Config overrides
    parser.add_argument("--num-stations", type=int, default=2)
    parser.add_argument("--num-evaders", type=int, default=3)
    parser.add_argument("--arena-radius", type=float, default=50.0)
    
    args = parser.parse_args()
    
    config = get_default_multi_config()
    config.station.num_stations = args.num_stations
    config.evader.num_evaders = args.num_evaders
    config.arena.radius = args.arena_radius
    
    if args.save_animation:
        create_trajectory_animation(
            config,
            pursuer_model_path=args.pursuer_model,
            evader_model_path=args.evader_model,
            save_path=args.save_animation,
        )
    elif args.save_plot:
        plot_trajectories(
            config,
            pursuer_model_path=args.pursuer_model,
            evader_model_path=args.evader_model,
            save_path=args.save_plot,
        )
    else:
        sim = MultiAgentSimulator(
            config,
            pursuer_model_path=args.pursuer_model,
            evader_model_path=args.evader_model,
        )
        
        if args.episodes == 1:
            sim.run_episode(render=not args.no_render)
        else:
            sim.run_batch(args.episodes, render=not args.no_render)
        
        sim.close()


if __name__ == "__main__":
    main()
