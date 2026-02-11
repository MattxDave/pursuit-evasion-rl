#!/usr/bin/env python3
"""
Simulation & Visualization
==========================
Run trained models and visualize pursuit episodes.

Usage:
    python simulate.py                          # Run latest model
    python simulate.py --model models/best      # Specific model
    python simulate.py --episodes 5             # Multiple episodes
    python simulate.py --save-video output.mp4  # Save video
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, FFMpegWriter

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from stable_baselines3 import PPO

from src.environment import PursuitEnv
from src.config import Config


class Simulator:
    """Run and visualize trained pursuit models."""
    
    def __init__(
        self,
        model_path: str,
        config: Optional[Config] = None,
        evader_noise_std: float = 0.0,
        evader_noise_prob: float = 0.0,
    ):
        """
        Initialize simulator.
        
        Args:
            model_path: Path to trained model (.zip file)
            config: Environment config (uses default if None)
            evader_noise_std: Heading noise std (radians)
            evader_noise_prob: Per-step probability to apply noise
        """
        self.config = config or Config()
        self.config.evader_noise.heading_noise_std = evader_noise_std
        self.config.evader_noise.noise_prob = evader_noise_prob
        
        print(f"Loading model: {model_path}")
        self.model = PPO.load(model_path)
        
        self.env = PursuitEnv(config=self.config)
    
    def run_episode(
        self,
        seed: Optional[int] = None,
        evader_speed: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Run single episode and collect trajectory.
        
        Returns:
            Episode data including trajectory and outcome
        """
        obs, info = self.env.reset(seed=seed)
        
        # Override evader speed if specified
        if evader_speed is not None:
            direction = self.env.evader.velocity / (np.linalg.norm(self.env.evader.velocity) + 1e-8)
            self.env.evader.velocity = (evader_speed * direction).astype(np.float32)
        
        trajectory = {
            "pursuer": [self.env.pursuer.position.copy()],
            "evader": [self.env.evader.position.copy()],
            "goal": self.env.evader.goal.copy(),
            "evader_speed": self.env.evader.speed,
            "distances": [float(np.linalg.norm(
                self.env.evader.position - self.env.pursuer.position
            ))],
            "actions": [],
            "rewards": [],
            # Battery and speed tracking
            "battery": [100.0],  # Start at full
            "pursuer_speeds": [0.0],
            "evader_speeds": [self.env.evader.speed],
        }
        
        done = False
        total_reward = 0.0
        
        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            total_reward += reward
            
            trajectory["pursuer"].append(self.env.pursuer.position.copy())
            trajectory["evader"].append(self.env.evader.position.copy())
            trajectory["distances"].append(info.get("distance", 0))
            trajectory["actions"].append(action.copy())
            trajectory["rewards"].append(float(reward))
            
            # Track battery and speeds
            trajectory["battery"].append(info.get("battery_percent", 100.0))
            trajectory["pursuer_speeds"].append(info.get("pursuer_speed", 0.0))
            trajectory["evader_speeds"].append(info.get("evader_speed", self.env.evader.speed))
        
        trajectory["outcome"] = info.get("outcome", "unknown")
        trajectory["total_reward"] = total_reward
        trajectory["steps"] = len(trajectory["pursuer"]) - 1
        trajectory["capture_time"] = info.get("capture_time")
        trajectory["final_battery"] = trajectory["battery"][-1]
        
        return trajectory
    
    def visualize(
        self,
        trajectory: Dict[str, Any],
        speed: float = 1.0,
        save_path: Optional[str] = None,
    ):
        """Animate episode trajectory with detailed metrics."""
        
        pursuer = np.array(trajectory["pursuer"])
        evader = np.array(trajectory["evader"])
        goal = trajectory["goal"]
        battery = np.array(trajectory["battery"])
        pursuer_speeds = np.array(trajectory["pursuer_speeds"])
        evader_speeds = np.array(trajectory["evader_speeds"])
        
        # Setup figure with multiple panels
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 3, height_ratios=[2, 1], width_ratios=[2, 1, 1])
        
        # Main arena view
        ax_main = fig.add_subplot(gs[0, 0])
        R = self.config.env.radius
        ax_main.set_xlim(-R - 2, R + 2)
        ax_main.set_ylim(-R - 2, R + 2)
        ax_main.set_aspect("equal")
        ax_main.grid(True, alpha=0.3)
        ax_main.set_xlabel("X (meters)", fontsize=11)
        ax_main.set_ylabel("Y (meters)", fontsize=11)
        ax_main.set_title("Arena View", fontsize=13, fontweight="bold")
        
        # Arena boundary
        arena = patches.Circle((0, 0), R, fill=False, linewidth=2, 
                               color="navy", linestyle="--")
        ax_main.add_patch(arena)
        
        # Goal
        ax_main.plot(*goal, marker="*", markersize=20, color="gold", 
               markeredgecolor="orange", label="Goal")
        
        # Agent markers
        pursuer_marker, = ax_main.plot([], [], "o", markersize=14, color="blue", label="Pursuer")
        evader_marker, = ax_main.plot([], [], "o", markersize=12, color="red", label="Evader")
        
        # Trails
        pursuer_trail, = ax_main.plot([], [], "-", linewidth=2, color="blue", alpha=0.4)
        evader_trail, = ax_main.plot([], [], "-", linewidth=2, color="red", alpha=0.4)
        ax_main.legend(loc="upper right", fontsize=9)
        
        # Stats text panel (top right)
        ax_stats = fig.add_subplot(gs[0, 1:])
        ax_stats.axis("off")
        stats_text = ax_stats.text(
            0.05, 0.95, "", transform=ax_stats.transAxes,
            fontsize=12, verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9)
        )
        
        # Battery gauge (bottom left)
        ax_battery = fig.add_subplot(gs[1, 0])
        ax_battery.set_xlim(0, len(battery))
        ax_battery.set_ylim(0, 105)
        ax_battery.set_xlabel("Step", fontsize=10)
        ax_battery.set_ylabel("Battery (%)", fontsize=10)
        ax_battery.set_title("Battery Level", fontsize=11)
        ax_battery.grid(True, alpha=0.3)
        ax_battery.axhline(y=20, color="red", linestyle="--", alpha=0.5, label="Critical (20%)")
        battery_line, = ax_battery.plot([], [], "g-", linewidth=2, label="Battery")
        ax_battery.legend(loc="upper right", fontsize=8)
        
        # Speed comparison (bottom middle)
        ax_speed = fig.add_subplot(gs[1, 1])
        ax_speed.set_xlim(0, len(pursuer_speeds))
        max_speed = max(np.max(pursuer_speeds), np.max(evader_speeds)) * 1.1
        ax_speed.set_ylim(0, max_speed + 0.5)
        ax_speed.set_xlabel("Step", fontsize=10)
        ax_speed.set_ylabel("Speed (m/s)", fontsize=10)
        ax_speed.set_title("Speed Comparison", fontsize=11)
        ax_speed.grid(True, alpha=0.3)
        pursuer_speed_line, = ax_speed.plot([], [], "b-", linewidth=2, label="Pursuer")
        evader_speed_line, = ax_speed.plot([], [], "r-", linewidth=2, label="Evader")
        ax_speed.legend(loc="upper right", fontsize=8)
        
        # Distance plot (bottom right)
        ax_dist = fig.add_subplot(gs[1, 2])
        distances = np.array(trajectory["distances"])
        ax_dist.set_xlim(0, len(distances))
        ax_dist.set_ylim(0, np.max(distances) * 1.1)
        ax_dist.set_xlabel("Step", fontsize=10)
        ax_dist.set_ylabel("Distance (m)", fontsize=10)
        ax_dist.set_title("Distance to Evader", fontsize=11)
        ax_dist.grid(True, alpha=0.3)
        ax_dist.axhline(y=self.config.env.capture_radius, color="green", linestyle="--", alpha=0.5, label="Capture")
        distance_line, = ax_dist.plot([], [], "purple", linewidth=2, label="Distance")
        ax_dist.legend(loc="upper right", fontsize=8)
        
        outcome = trajectory["outcome"]
        outcome_colors = {
            "capture": "green", 
            "escaped": "red",
            "goal_reached": "orange", 
            "timeout": "gray"
        }
        
        def init():
            pursuer_marker.set_data([], [])
            evader_marker.set_data([], [])
            pursuer_trail.set_data([], [])
            evader_trail.set_data([], [])
            stats_text.set_text("")
            battery_line.set_data([], [])
            pursuer_speed_line.set_data([], [])
            evader_speed_line.set_data([], [])
            distance_line.set_data([], [])
            return (pursuer_marker, evader_marker, pursuer_trail, evader_trail, 
                   stats_text, battery_line, pursuer_speed_line, evader_speed_line, distance_line)
        
        def update(frame):
            idx = min(frame, len(pursuer) - 1)
            
            # Update markers
            pursuer_marker.set_data([pursuer[idx, 0]], [pursuer[idx, 1]])
            evader_marker.set_data([evader[idx, 0]], [evader[idx, 1]])
            
            # Update trails
            pursuer_trail.set_data(pursuer[:idx+1, 0], pursuer[:idx+1, 1])
            evader_trail.set_data(evader[:idx+1, 0], evader[:idx+1, 1])
            
            # Update plots
            x_range = np.arange(idx + 1)
            battery_line.set_data(x_range, battery[:idx+1])
            pursuer_speed_line.set_data(x_range, pursuer_speeds[:idx+1])
            evader_speed_line.set_data(x_range, evader_speeds[:idx+1])
            distance_line.set_data(x_range, distances[:idx+1])
            
            # Color battery line based on level
            if battery[idx] < 20:
                battery_line.set_color("red")
            elif battery[idx] < 50:
                battery_line.set_color("orange")
            else:
                battery_line.set_color("green")
            
            # Update stats
            dist = distances[idx]
            t = idx * self.config.env.dt
            
            stats = "╔══════════════════════════════╗\n"
            stats += "║     PURSUIT-EVASION STATS    ║\n"
            stats += "╠══════════════════════════════╣\n"
            stats += f"║ Time:          {t:>7.1f} s     ║\n"
            stats += f"║ Step:          {idx:>7d}       ║\n"
            stats += "╠══════════════════════════════╣\n"
            stats += f"║ Distance:      {dist:>7.2f} m     ║\n"
            stats += "╠══════════════════════════════╣\n"
            stats += f"║ PURSUER                      ║\n"
            stats += f"║   Speed:       {pursuer_speeds[idx]:>7.2f} m/s  ║\n"
            stats += f"║   Battery:     {battery[idx]:>7.1f} %     ║\n"
            stats += f"║   Max Speed:   {self.config.env.pursuer_max_speed:>7.1f} m/s  ║\n"
            stats += "╠══════════════════════════════╣\n"
            stats += f"║ EVADER                       ║\n"
            stats += f"║   Speed:       {evader_speeds[idx]:>7.2f} m/s  ║\n"
            stats += "╚══════════════════════════════╝\n"
            
            if idx == len(pursuer) - 1:
                stats += "\n"
                stats += "╔══════════════════════════════╗\n"
                stats += f"║  OUTCOME: {outcome.upper():^17}  ║\n"
                stats += "╠══════════════════════════════╣\n"
                stats += f"║  Final Battery:  {battery[idx]:>6.1f} %    ║\n"
                stats += f"║  Total Steps:    {idx:>6d}      ║\n"
                if trajectory["capture_time"]:
                    stats += f"║  Capture Time:   {trajectory['capture_time']:>6.1f} s    ║\n"
                stats += f"║  Total Reward:   {trajectory['total_reward']:>6.1f}      ║\n"
                stats += "╚══════════════════════════════╝"
            
            stats_text.set_text(stats)
            
            # Main title
            fig.suptitle(
                f"Pursuit-Evasion Simulation | Evader: {trajectory['evader_speed']:.1f} m/s",
                fontsize=14, fontweight="bold"
            )
            
            return (pursuer_marker, evader_marker, pursuer_trail, evader_trail, 
                   stats_text, battery_line, pursuer_speed_line, evader_speed_line, distance_line)
        
        interval = int(self.config.env.dt * 1000 / speed)
        anim = FuncAnimation(
            fig, update, frames=len(pursuer),
            init_func=init, interval=interval, blit=True, repeat=False
        )
        
        if save_path:
            print(f"Saving video to: {save_path}")
            writer = FFMpegWriter(fps=int(30 * speed))
            anim.save(save_path, writer=writer, dpi=150)
            plt.close(fig)
        else:
            plt.tight_layout()
            plt.show()
        
        return anim
    
    def benchmark(
        self,
        n_episodes: int = 100,
        speeds: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """Run benchmark across different evader speeds."""
        
        if speeds is None:
            speeds = [1.0, 2.0, 3.0, 4.0, 5.0]
        
        results = {}
        
        for speed in speeds:
            print(f"\nBenchmarking evader speed: {speed:.1f} m/s")
            
            outcomes = {"capture": 0, "escaped": 0, "goal_reached": 0, "timeout": 0}
            capture_times = []
            rewards = []
            
            for ep in range(n_episodes):
                traj = self.run_episode(seed=ep, evader_speed=speed)
                
                outcome = traj["outcome"]
                outcomes[outcome] = outcomes.get(outcome, 0) + 1
                rewards.append(traj["total_reward"])
                
                if traj["capture_time"]:
                    capture_times.append(traj["capture_time"])
            
            capture_rate = outcomes["capture"] / n_episodes * 100
            avg_reward = np.mean(rewards)
            avg_capture_time = np.mean(capture_times) if capture_times else float("nan")
            
            results[speed] = {
                "outcomes": outcomes,
                "capture_rate": capture_rate,
                "avg_reward": avg_reward,
                "avg_capture_time": avg_capture_time,
            }
            
            print(f"  Capture rate: {capture_rate:.1f}%")
            print(f"  Avg reward: {avg_reward:.2f}")
            print(f"  Avg capture time: {avg_capture_time:.2f}s")
        
        return results
    
    def plot_benchmark(self, results: Dict[str, Any], save_path: Optional[str] = None):
        """Plot benchmark results."""
        
        speeds = sorted(results.keys())
        capture_rates = [results[s]["capture_rate"] for s in speeds]
        avg_rewards = [results[s]["avg_reward"] for s in speeds]
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Capture rate
        bars1 = axes[0].bar(speeds, capture_rates, color="steelblue", edgecolor="navy")
        axes[0].set_xlabel("Evader Speed (m/s)", fontsize=12)
        axes[0].set_ylabel("Capture Rate (%)", fontsize=12)
        axes[0].set_title("Capture Rate vs Evader Speed", fontsize=14)
        axes[0].set_ylim(0, 105)
        axes[0].axhline(y=50, color="red", linestyle="--", alpha=0.5, label="50% baseline")
        axes[0].legend()
        axes[0].grid(axis="y", alpha=0.3)
        
        # Add value labels
        for bar, rate in zip(bars1, capture_rates):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                        f'{rate:.0f}%', ha='center', fontsize=10)
        
        # Average reward
        bars2 = axes[1].bar(speeds, avg_rewards, color="coral", edgecolor="darkred")
        axes[1].set_xlabel("Evader Speed (m/s)", fontsize=12)
        axes[1].set_ylabel("Average Reward", fontsize=12)
        axes[1].set_title("Average Reward vs Evader Speed", fontsize=14)
        axes[1].grid(axis="y", alpha=0.3)
        axes[1].axhline(y=0, color="black", linestyle="-", alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved benchmark plot to: {save_path}")
        
        plt.show()


def find_model(model_dir: str = "models") -> str:
    """Find most recent model."""
    model_path = Path(model_dir)
    
    # Look for best model
    for pattern in ["**/best/*.zip", "**/best_model.zip", "**/final_model.zip", "**/*.zip"]:
        matches = list(model_path.glob(pattern))
        if matches:
            # Sort by modification time, newest first
            matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return str(matches[0])
    
    raise FileNotFoundError(f"No model found in {model_dir}")


def main():
    parser = argparse.ArgumentParser(description="Simulate trained pursuit model")
    
    parser.add_argument("--model", type=str, default=None,
                       help="Path to model file")
    parser.add_argument("--model-dir", type=str, default="models",
                       help="Directory to search for models")
    parser.add_argument("--episodes", type=int, default=1,
                       help="Number of episodes to run")
    parser.add_argument("--evader-speed", type=float, default=None,
                       help="Fixed evader speed")
    parser.add_argument("--evader-noise-std", type=float, default=0.0,
                       help="Evader heading noise std (radians)")
    parser.add_argument("--evader-noise-prob", type=float, default=0.0,
                       help="Per-step probability to apply evader heading noise")
    parser.add_argument("--speed", type=float, default=1.0,
                       help="Playback speed multiplier")
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed")
    parser.add_argument("--save-video", type=str, default=None,
                       help="Save video to file")
    parser.add_argument("--benchmark", action="store_true",
                       help="Run benchmark mode")
    parser.add_argument("--benchmark-episodes", type=int, default=50,
                       help="Episodes per speed in benchmark")
    parser.add_argument("--no-display", action="store_true",
                       help="Don't show visualization")
    
    args = parser.parse_args()
    
    # Find model
    if args.model:
        model_path = args.model
    else:
        model_path = find_model(args.model_dir)
    
    print(f"Using model: {model_path}")
    
    # Create simulator
    sim = Simulator(
        model_path,
        evader_noise_std=args.evader_noise_std,
        evader_noise_prob=args.evader_noise_prob,
    )
    
    # Benchmark mode
    if args.benchmark:
        results = sim.benchmark(n_episodes=args.benchmark_episodes)
        sim.plot_benchmark(results, save_path="benchmark_results.png")
        return
    
    # Run episodes
    for ep in range(args.episodes):
        print(f"\n{'='*50}")
        print(f"Episode {ep + 1}/{args.episodes}")
        print(f"{'='*50}")
        
        seed = (args.seed + ep) if args.seed else None
        trajectory = sim.run_episode(seed=seed, evader_speed=args.evader_speed)
        
        print(f"Outcome: {trajectory['outcome']}")
        print(f"Evader speed: {trajectory['evader_speed']:.2f} m/s")
        print(f"Steps: {trajectory['steps']}")
        print(f"Total reward: {trajectory['total_reward']:.2f}")
        print(f"Final battery: {trajectory['final_battery']:.1f}%")
        print(f"Avg pursuer speed: {np.mean(trajectory['pursuer_speeds'][1:]):.2f} m/s")
        if trajectory["capture_time"]:
            print(f"Capture time: {trajectory['capture_time']:.2f}s")
        
        # Visualize
        if not args.no_display:
            save_path = None
            if args.save_video:
                if args.episodes > 1:
                    base, ext = os.path.splitext(args.save_video)
                    save_path = f"{base}_ep{ep+1}{ext}"
                else:
                    save_path = args.save_video
            
            sim.visualize(trajectory, speed=args.speed, save_path=save_path)


if __name__ == "__main__":
    main()
