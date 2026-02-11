"""
Compare Hybrid RL Model vs Pure Classical Controller

Evaluates both approaches on identical episodes to directly compare performance.
"""

import argparse
import numpy as np
from stable_baselines3 import PPO
from src.environment import PursuitEnv
from src.config import Config
from src.agents import Pursuer, Evader
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple
from pathlib import Path
def resolve_model_path(path_str: str) -> str:
    """Resolve model path; accept a directory and pick best/final zip."""
    p = Path(path_str)
    if p.is_file():
        return str(p)
    if p.is_dir():
        candidates = [
            p / "best" / "best_model.zip",
            p / "best_model.zip",
            p / "final_model.zip",
        ]
        for c in candidates:
            if c.exists():
                return str(c)
    raise FileNotFoundError(f"Could not resolve model path from: {path_str}")


@dataclass
class EpisodeResult:
    """Results from a single episode."""
    outcome: str  # 'capture', 'goal_reached', 'timeout', 'boundary'
    steps: int
    reward: float
    final_battery: float
    avg_speed: float
    capture_time: float | None
    trajectory_pursuer: List[Tuple[float, float]]
    trajectory_evader: List[Tuple[float, float]]


def run_classical_episode(config: Config, evader_start: np.ndarray, 
                          pursuer_start: np.ndarray, evader_speed: float,
                          goal_position: np.ndarray,
                          heading_noise_std: float = 0.0,
                          noise_prob: float = 0.0) -> EpisodeResult:
    """Run episode with pure classical lead-intercept controller."""
    
    env_cfg = config.env
    
    # Create and initialize pursuer
    pursuer = Pursuer(
        max_speed=env_cfg.pursuer_max_speed,
        battery_capacity=env_cfg.battery_capacity,
        battery=env_cfg.battery_capacity,
        battery_drain_rate=env_cfg.battery_drain_rate,
        battery_min_speed_factor=env_cfg.battery_min_speed_factor,
    )
    pursuer.position = pursuer_start.copy().astype(np.float32)
    
    # Create and initialize evader
    evader = Evader(
        heading_noise_std=heading_noise_std,
        noise_prob=noise_prob,
    )
    evader.reset(
        position=evader_start.copy(),
        goal=goal_position.copy(),
        speed=evader_speed,
        heading_noise_std=heading_noise_std,
        noise_prob=noise_prob,
        rng=np.random.default_rng(),
    )
    
    trajectory_p = [tuple(pursuer.position)]
    trajectory_e = [tuple(evader.position)]
    
    total_reward = 0.0
    speeds = []
    
    for step in range(env_cfg.max_steps):
        # Classical control: always max throttle (1.0), no nudge (0.0)
        # The Pursuer.step() method handles battery limiting internally
        pursuer.step(
            target_pos=evader.position,
            target_vel=evader.velocity,
            throttle=1.0,  # Always full throttle for classical
            heading_nudge=0.0,  # No heading adjustment for classical
            dt=env_cfg.dt
        )
        speeds.append(pursuer.actual_speed)
        
        # Move evader
        evader.step(env_cfg.dt)
        
        trajectory_p.append(tuple(pursuer.position))
        trajectory_e.append(tuple(evader.position))
        
        # Check termination conditions
        dist = np.linalg.norm(evader.position - pursuer.position)
        evader_to_goal_dist = np.linalg.norm(evader.position - goal_position)
        pursuer_dist_from_origin = np.linalg.norm(pursuer.position)
        evader_dist_from_origin = np.linalg.norm(evader.position)
        
        # Capture
        if dist < env_cfg.capture_radius:
            total_reward = 50.0 - step * 0.1  # Reward for faster capture
            return EpisodeResult(
                outcome='capture',
                steps=step + 1,
                reward=total_reward,
                final_battery=pursuer.battery_percent,
                avg_speed=np.mean(speeds),
                capture_time=(step + 1) * env_cfg.dt,
                trajectory_pursuer=trajectory_p,
                trajectory_evader=trajectory_e
            )
        
        # Goal reached
        if evader_to_goal_dist < env_cfg.capture_radius:
            total_reward = -20.0
            return EpisodeResult(
                outcome='goal_reached',
                steps=step + 1,
                reward=total_reward,
                final_battery=pursuer.battery_percent,
                avg_speed=np.mean(speeds),
                capture_time=None,
                trajectory_pursuer=trajectory_p,
                trajectory_evader=trajectory_e
            )
        
        # Boundary
        if pursuer_dist_from_origin > env_cfg.radius or evader_dist_from_origin > env_cfg.radius:
            total_reward = -10.0
            return EpisodeResult(
                outcome='boundary',
                steps=step + 1,
                reward=total_reward,
                final_battery=pursuer.battery_percent,
                avg_speed=np.mean(speeds),
                capture_time=None,
                trajectory_pursuer=trajectory_p,
                trajectory_evader=trajectory_e
            )
    
    # Timeout
    total_reward = -5.0
    return EpisodeResult(
        outcome='timeout',
        steps=env_cfg.max_steps,
        reward=total_reward,
        final_battery=pursuer.battery_percent,
        avg_speed=np.mean(speeds),
        capture_time=None,
        trajectory_pursuer=trajectory_p,
        trajectory_evader=trajectory_e
    )


def run_hybrid_episode(model: PPO, config: Config, evader_start: np.ndarray,
                       pursuer_start: np.ndarray, evader_speed: float,
                       goal_position: np.ndarray) -> EpisodeResult:
    """Run episode with hybrid RL controller."""
    
    # Create environment with fixed initial conditions
    env = PursuitEnv(config=config)
    obs, info = env.reset()
    
    # Override with our specific initial conditions
    env.pursuer.position = pursuer_start.copy()
    env.evader.position = evader_start.copy()
    env.goal_position = goal_position.copy()
    env.pursuer.battery = config.env.battery_capacity  # Reset battery
    
    # Reset evader with specific speed and goal
    env.evader.reset(
        position=evader_start.copy(),
        goal=goal_position.copy(),
        speed=evader_speed
    )
    
    # Recompute observation
    obs = env._get_observation()
    
    trajectory_p = [tuple(env.pursuer.position)]
    trajectory_e = [tuple(env.evader.position)]
    
    total_reward = 0.0
    speeds = []
    done = False
    step = 0
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        total_reward += reward
        speeds.append(info['pursuer_speed'])
        trajectory_p.append(tuple(env.pursuer.position))
        trajectory_e.append(tuple(env.evader.position))
        step += 1
    
    capture_time = step * config.env.dt if info['outcome'] == 'capture' else None
    
    return EpisodeResult(
        outcome=info['outcome'],
        steps=step,
        reward=total_reward,
        final_battery=info['battery_percent'],
        avg_speed=np.mean(speeds),
        capture_time=capture_time,
        trajectory_pursuer=trajectory_p,
        trajectory_evader=trajectory_e
    )


def generate_test_scenarios(config: Config, num_episodes: int, 
                           evader_speed: float, seed: int = 42) -> List[dict]:
    """Generate fixed test scenarios for fair comparison."""
    np.random.seed(seed)
    scenarios = []
    
    env_cfg = config.env
    
    for _ in range(num_episodes):
        # Random goal on arena boundary
        goal_angle = np.random.uniform(0, 2 * np.pi)
        goal_position = np.array([
            env_cfg.radius * np.cos(goal_angle),
            env_cfg.radius * np.sin(goal_angle)
        ])
        
        # Random evader position
        evader_dist = np.random.uniform(5, env_cfg.radius - 5)
        evader_angle = np.random.uniform(0, 2 * np.pi)
        evader_start = np.array([
            evader_dist * np.cos(evader_angle),
            evader_dist * np.sin(evader_angle)
        ])
        
        # Pursuer starts at origin
        pursuer_start = np.array([0.0, 0.0])
        
        scenarios.append({
            'goal_position': goal_position,
            'evader_start': evader_start,
            'pursuer_start': pursuer_start,
            'evader_speed': evader_speed
        })
    
    return scenarios


def plot_comparison(hybrid_results: List[EpisodeResult], 
                    classical_results: List[EpisodeResult],
                    evader_speed: float, save_path: str = None):
    """Create comparison plots."""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Hybrid RL vs Classical Controller (Evader Speed: {evader_speed} m/s)', 
                 fontsize=14, fontweight='bold')
    
    # 1. Capture rate comparison
    ax = axes[0, 0]
    hybrid_captures = sum(1 for r in hybrid_results if r.outcome == 'capture')
    classical_captures = sum(1 for r in classical_results if r.outcome == 'capture')
    n_episodes = len(hybrid_results)
    
    bars = ax.bar(['Hybrid RL', 'Classical'], 
                  [hybrid_captures/n_episodes*100, classical_captures/n_episodes*100],
                  color=['#2ecc71', '#3498db'])
    ax.set_ylabel('Capture Rate (%)')
    ax.set_title('Capture Rate')
    ax.set_ylim(0, 100)
    for bar, val in zip(bars, [hybrid_captures, classical_captures]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                f'{val}/{n_episodes}', ha='center', fontsize=10)
    
    # 2. Average reward
    ax = axes[0, 1]
    hybrid_avg_reward = np.mean([r.reward for r in hybrid_results])
    classical_avg_reward = np.mean([r.reward for r in classical_results])
    bars = ax.bar(['Hybrid RL', 'Classical'], [hybrid_avg_reward, classical_avg_reward],
                  color=['#2ecc71', '#3498db'])
    ax.set_ylabel('Average Reward')
    ax.set_title('Average Reward')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # 3. Average battery remaining
    ax = axes[0, 2]
    hybrid_battery = np.mean([r.final_battery for r in hybrid_results])
    classical_battery = np.mean([r.final_battery for r in classical_results])
    bars = ax.bar(['Hybrid RL', 'Classical'], [hybrid_battery, classical_battery],
                  color=['#2ecc71', '#3498db'])
    ax.set_ylabel('Final Battery (%)')
    ax.set_title('Battery Efficiency')
    ax.set_ylim(0, 100)
    
    # 4. Average speed used
    ax = axes[1, 0]
    hybrid_speed = np.mean([r.avg_speed for r in hybrid_results])
    classical_speed = np.mean([r.avg_speed for r in classical_results])
    bars = ax.bar(['Hybrid RL', 'Classical'], [hybrid_speed, classical_speed],
                  color=['#2ecc71', '#3498db'])
    ax.set_ylabel('Average Speed (m/s)')
    ax.set_title('Speed Usage')
    
    # 5. Capture time (for successful captures only)
    ax = axes[1, 1]
    hybrid_times = [r.capture_time for r in hybrid_results if r.capture_time is not None]
    classical_times = [r.capture_time for r in classical_results if r.capture_time is not None]
    
    if hybrid_times and classical_times:
        ax.bar(['Hybrid RL', 'Classical'], 
               [np.mean(hybrid_times), np.mean(classical_times)],
               color=['#2ecc71', '#3498db'])
        ax.set_ylabel('Capture Time (s)')
        ax.set_title('Avg Capture Time (captures only)')
    else:
        ax.text(0.5, 0.5, 'No captures to compare', ha='center', va='center', 
                transform=ax.transAxes)
        ax.set_title('Avg Capture Time')
    
    # 6. Outcome distribution
    ax = axes[1, 2]
    outcomes = ['capture', 'goal_reached', 'timeout', 'boundary']
    hybrid_outcomes = [sum(1 for r in hybrid_results if r.outcome == o) for o in outcomes]
    classical_outcomes = [sum(1 for r in classical_results if r.outcome == o) for o in outcomes]
    
    x = np.arange(len(outcomes))
    width = 0.35
    ax.bar(x - width/2, hybrid_outcomes, width, label='Hybrid RL', color='#2ecc71')
    ax.bar(x + width/2, classical_outcomes, width, label='Classical', color='#3498db')
    ax.set_xticks(x)
    ax.set_xticklabels(outcomes, rotation=45, ha='right')
    ax.set_ylabel('Count')
    ax.set_title('Outcome Distribution')
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()


def plot_trajectories(hybrid_results: List[EpisodeResult],
                      classical_results: List[EpisodeResult],
                      scenarios: List[dict], config: Config,
                      num_to_plot: int = 4, save_path: str = None):
    """Plot trajectory comparisons for a few episodes."""
    
    num_to_plot = min(num_to_plot, len(hybrid_results))
    fig, axes = plt.subplots(2, num_to_plot, figsize=(4*num_to_plot, 8))
    
    if num_to_plot == 1:
        axes = axes.reshape(2, 1)
    
    fig.suptitle('Trajectory Comparison: Hybrid RL vs Classical', fontsize=14, fontweight='bold')
    
    env_cfg = config.env
    
    for i in range(num_to_plot):
        scenario = scenarios[i]
        hybrid = hybrid_results[i]
        classical = classical_results[i]
        
        for row, (result, title) in enumerate([(hybrid, 'Hybrid RL'), (classical, 'Classical')]):
            ax = axes[row, i]
            
            # Draw arena
            theta = np.linspace(0, 2*np.pi, 100)
            ax.plot(env_cfg.radius * np.cos(theta), 
                   env_cfg.radius * np.sin(theta), 'k-', alpha=0.3)
            
            # Draw goal
            goal = scenario['goal_position']
            goal_circle = plt.Circle(goal, env_cfg.capture_radius, color='green', alpha=0.3)
            ax.add_patch(goal_circle)
            ax.plot(goal[0], goal[1], 'g*', markersize=15, label='Goal')
            
            # Draw trajectories
            traj_p = np.array(result.trajectory_pursuer)
            traj_e = np.array(result.trajectory_evader)
            
            ax.plot(traj_p[:, 0], traj_p[:, 1], 'b-', linewidth=2, label='Pursuer')
            ax.plot(traj_e[:, 0], traj_e[:, 1], 'r-', linewidth=2, label='Evader')
            
            # Start and end points
            ax.plot(traj_p[0, 0], traj_p[0, 1], 'bo', markersize=10)
            ax.plot(traj_e[0, 0], traj_e[0, 1], 'ro', markersize=10)
            ax.plot(traj_p[-1, 0], traj_p[-1, 1], 'bs', markersize=10)
            ax.plot(traj_e[-1, 0], traj_e[-1, 1], 'rs', markersize=10)
            
            ax.set_xlim(-env_cfg.radius*1.1, env_cfg.radius*1.1)
            ax.set_ylim(-env_cfg.radius*1.1, env_cfg.radius*1.1)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            
            outcome_color = 'green' if result.outcome == 'capture' else 'red'
            ax.set_title(f'{title}\n{result.outcome} ({result.steps} steps)', 
                        color=outcome_color, fontsize=10)
            
            if i == 0:
                ax.set_ylabel(title)
            if row == 0 and i == 0:
                ax.legend(loc='upper left', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Trajectories saved to: {save_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Compare Hybrid RL vs Classical Controller')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--episodes', type=int, default=50, help='Number of test episodes')
    parser.add_argument('--evader-speed', type=float, default=3.0, help='Evader speed (m/s)')
    parser.add_argument('--evader-noise-std', type=float, default=0.0, help='Evader heading noise std (radians)')
    parser.add_argument('--evader-noise-prob', type=float, default=0.0, help='Per-step probability of applying heading noise')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--save-dir', type=str, default='eval_comparison', help='Directory to save plots')
    parser.add_argument('--no-plot', action='store_true', help='Skip plotting')
    args = parser.parse_args()
    
    # Setup
    config = Config()
    config.evader_noise.heading_noise_std = args.evader_noise_std
    config.evader_noise.noise_prob = args.evader_noise_prob
    print(f"\n{'='*60}")
    print("HYBRID RL vs CLASSICAL CONTROLLER COMPARISON")
    print(f"{'='*60}")
    resolved_model = resolve_model_path(args.model)
    print(f"Model: {resolved_model}")
    print(f"Episodes: {args.episodes}")
    print(f"Evader speed: {args.evader_speed} m/s")
    print(f"Evader noise: std={args.evader_noise_std} rad, prob={args.evader_noise_prob}")
    print(f"Pursuer max speed: {config.env.pursuer_max_speed} m/s")
    print(f"{'='*60}\n")
    
    # Load model
    print("Loading hybrid model...")
    model = PPO.load(resolved_model)
    
    # Generate test scenarios
    print("Generating test scenarios...")
    scenarios = generate_test_scenarios(config, args.episodes, args.evader_speed, args.seed)
    
    # Run evaluations
    print("\nRunning Hybrid RL episodes...")
    hybrid_results = []
    for i, scenario in enumerate(scenarios):
        result = run_hybrid_episode(model, config, 
                                    scenario['evader_start'],
                                    scenario['pursuer_start'],
                                    scenario['evader_speed'],
                                    scenario['goal_position'])
        hybrid_results.append(result)
        if (i + 1) % 10 == 0:
            print(f"  Completed {i+1}/{args.episodes}")
    
    print("\nRunning Classical episodes...")
    classical_results = []
    for i, scenario in enumerate(scenarios):
        result = run_classical_episode(config,
                                       scenario['evader_start'],
                                       scenario['pursuer_start'],
                                       scenario['evader_speed'],
                                       scenario['goal_position'],
                                       heading_noise_std=args.evader_noise_std,
                                       noise_prob=args.evader_noise_prob)
        classical_results.append(result)
        if (i + 1) % 10 == 0:
            print(f"  Completed {i+1}/{args.episodes}")
    
    # Print summary statistics
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    
    hybrid_captures = sum(1 for r in hybrid_results if r.outcome == 'capture')
    classical_captures = sum(1 for r in classical_results if r.outcome == 'capture')
    
    print(f"\n{'Metric':<25} {'Hybrid RL':>15} {'Classical':>15}")
    print("-" * 55)
    print(f"{'Capture Rate':<25} {hybrid_captures}/{args.episodes} ({hybrid_captures/args.episodes*100:.1f}%){' ':>3} "
          f"{classical_captures}/{args.episodes} ({classical_captures/args.episodes*100:.1f}%)")
    print(f"{'Average Reward':<25} {np.mean([r.reward for r in hybrid_results]):>15.2f} "
          f"{np.mean([r.reward for r in classical_results]):>15.2f}")
    print(f"{'Average Battery':<25} {np.mean([r.final_battery for r in hybrid_results]):>14.1f}% "
          f"{np.mean([r.final_battery for r in classical_results]):>14.1f}%")
    print(f"{'Average Speed':<25} {np.mean([r.avg_speed for r in hybrid_results]):>13.2f} m/s "
          f"{np.mean([r.avg_speed for r in classical_results]):>13.2f} m/s")
    
    hybrid_times = [r.capture_time for r in hybrid_results if r.capture_time is not None]
    classical_times = [r.capture_time for r in classical_results if r.capture_time is not None]
    if hybrid_times:
        print(f"{'Avg Capture Time':<25} {np.mean(hybrid_times):>14.2f}s", end="")
    else:
        print(f"{'Avg Capture Time':<25} {'N/A':>15}", end="")
    if classical_times:
        print(f" {np.mean(classical_times):>14.2f}s")
    else:
        print(f" {'N/A':>15}")
    
    # Outcome breakdown
    print(f"\n{'Outcome Breakdown:'}")
    print("-" * 55)
    for outcome in ['capture', 'goal_reached', 'timeout', 'boundary']:
        h_count = sum(1 for r in hybrid_results if r.outcome == outcome)
        c_count = sum(1 for r in classical_results if r.outcome == outcome)
        print(f"  {outcome:<20} {h_count:>15} {c_count:>15}")
    
    # Plot results
    if not args.no_plot:
        import os
        os.makedirs(args.save_dir, exist_ok=True)
        
        plot_comparison(hybrid_results, classical_results, args.evader_speed,
                       save_path=f"{args.save_dir}/comparison_stats.png")
        
        plot_trajectories(hybrid_results, classical_results, scenarios, config,
                         num_to_plot=4, save_path=f"{args.save_dir}/comparison_trajectories.png")
    
    print(f"\n{'='*60}")
    print("Comparison complete!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
