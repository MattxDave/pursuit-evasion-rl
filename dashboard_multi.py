"""
Multi-Agent Training Dashboard
==============================
Visualize training progress and evaluate model performance.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import argparse
from stable_baselines3 import PPO
from collections import defaultdict

from src.multi_config import get_default_multi_config, MultiAgentConfig
from src.multi_environment import MultiAgentPursuitEnv
from src.multi_agents import EvaderRole


def load_tensorboard_data(log_dir: str) -> Dict[str, pd.DataFrame]:
    """Load training data from TensorBoard logs."""
    try:
        from tensorboard.backend.event_processing import event_accumulator
        
        data = {}
        log_path = Path(log_dir)
        
        # Find all event files
        for event_file in log_path.rglob('events.out.tfevents.*'):
            ea = event_accumulator.EventAccumulator(str(event_file.parent))
            ea.Reload()
            
            for tag in ea.Tags()['scalars']:
                events = ea.Scalars(tag)
                df = pd.DataFrame([(e.step, e.value) for e in events], 
                                 columns=['step', tag])
                if tag not in data:
                    data[tag] = df
                else:
                    data[tag] = pd.concat([data[tag], df]).drop_duplicates('step').sort_values('step')
        
        return data
    except ImportError:
        print("TensorBoard not installed. Using fallback data loading.")
        return {}


def evaluate_models(config: MultiAgentConfig,
                    pursuer_model_path: Optional[str] = None,
                    evader_model_path: Optional[str] = None,
                    num_episodes: int = 50) -> Dict:
    """Evaluate trained models and collect statistics."""
    
    # Load models
    pursuer_model = PPO.load(pursuer_model_path) if pursuer_model_path else None
    evader_model = PPO.load(evader_model_path) if evader_model_path else None
    
    env = MultiAgentPursuitEnv(config)
    n_pursuers = config.station.num_stations * config.station.pursuers_per_station
    n_evaders = config.evader.num_evaders
    
    stats = {
        'episode_lengths': [],
        'captures_per_episode': [],
        'escapes_per_episode': [],
        'pursuer_win_rate': 0,
        'evader_win_rate': 0,
        'time_to_first_capture': [],
        'time_to_first_escape': [],
        'pursuer_distances': [[] for _ in range(n_pursuers)],
        'evader_distances': [[] for _ in range(n_evaders)],
        'intercept_events': [],
    }
    
    pursuer_wins = 0
    evader_wins = 0
    
    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False
        step = 0
        
        captures = 0
        escapes = 0
        first_capture_time = None
        first_escape_time = None
        
        prev_positions = {
            'pursuers': [p.position.copy() for p in env.pursuers],
            'evaders': [e.position.copy() for e in env.evaders]
        }
        distances = {'pursuers': [0.0] * n_pursuers, 'evaders': [0.0] * n_evaders}
        
        captured_evaders = set()
        escaped_evaders = set()
        
        while not done:
            actions = {}
            
            # Pursuer actions
            if pursuer_model:
                for i in range(n_pursuers):
                    p_obs = env._get_pursuer_obs(i)
                    action, _ = pursuer_model.predict(p_obs, deterministic=True)
                    actions[f'pursuer_{i}'] = action
            else:
                for i in range(n_pursuers):
                    actions[f'pursuer_{i}'] = env.action_space[f'pursuer_{i}'].sample()
            
            # Evader actions
            if evader_model:
                for i in range(n_evaders):
                    if env.evaders[i].role not in [EvaderRole.CAPTURED, EvaderRole.ESCAPED]:
                        e_obs = env._get_evader_obs(i)
                        action, _ = evader_model.predict(e_obs, deterministic=True)
                        actions[f'evader_{i}'] = action
                    else:
                        actions[f'evader_{i}'] = np.zeros(2)
            else:
                for i in range(n_evaders):
                    actions[f'evader_{i}'] = env.action_space[f'evader_{i}'].sample()
            
            obs, reward, terminated, truncated, info = env.step(actions)
            done = terminated or truncated
            step += 1
            
            # Track distances
            for i, p in enumerate(env.pursuers):
                distances['pursuers'][i] += np.linalg.norm(p.position - prev_positions['pursuers'][i])
                prev_positions['pursuers'][i] = p.position.copy()
            
            for i, e in enumerate(env.evaders):
                distances['evaders'][i] += np.linalg.norm(e.position - prev_positions['evaders'][i])
                prev_positions['evaders'][i] = e.position.copy()
            
            # Track captures/escapes
            for i, e in enumerate(env.evaders):
                if e.role == EvaderRole.CAPTURED and i not in captured_evaders:
                    captures += 1
                    captured_evaders.add(i)
                    if first_capture_time is None:
                        first_capture_time = step
                elif e.role == EvaderRole.ESCAPED and i not in escaped_evaders:
                    escapes += 1
                    escaped_evaders.add(i)
                    if first_escape_time is None:
                        first_escape_time = step
        
        # Record episode stats
        stats['episode_lengths'].append(step)
        stats['captures_per_episode'].append(captures)
        stats['escapes_per_episode'].append(escapes)
        
        if first_capture_time:
            stats['time_to_first_capture'].append(first_capture_time)
        if first_escape_time:
            stats['time_to_first_escape'].append(first_escape_time)
        
        for i in range(n_pursuers):
            stats['pursuer_distances'][i].append(distances['pursuers'][i])
        for i in range(n_evaders):
            stats['evader_distances'][i].append(distances['evaders'][i])
        
        # Determine winner
        if captures == n_evaders:
            pursuer_wins += 1
        elif escapes > 0:
            evader_wins += 1
        
        if (ep + 1) % 10 == 0:
            print(f"  Episode {ep + 1}/{num_episodes}: {captures} captures, {escapes} escapes")
    
    env.close()
    
    stats['pursuer_win_rate'] = pursuer_wins / num_episodes
    stats['evader_win_rate'] = evader_wins / num_episodes
    stats['draw_rate'] = 1 - stats['pursuer_win_rate'] - stats['evader_win_rate']
    
    return stats


def plot_training_curves(log_dirs: Dict[str, str], save_path: Optional[str] = None):
    """Plot training curves from multiple runs."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    colors = {'pursuers': '#e41a1c', 'evaders': '#377eb8'}
    
    for name, log_dir in log_dirs.items():
        color = colors.get(name, 'gray')
        data = load_tensorboard_data(log_dir)
        
        if not data:
            print(f"No data found for {name}")
            continue
        
        # Episode reward
        if 'rollout/ep_rew_mean' in data:
            df = data['rollout/ep_rew_mean']
            axes[0, 0].plot(df['step'], df['rollout/ep_rew_mean'], 
                          label=name.capitalize(), color=color, linewidth=1.5)
        
        # Episode length
        if 'rollout/ep_len_mean' in data:
            df = data['rollout/ep_len_mean']
            axes[0, 1].plot(df['step'], df['rollout/ep_len_mean'],
                          label=name.capitalize(), color=color, linewidth=1.5)
        
        # Value loss
        if 'train/value_loss' in data:
            df = data['train/value_loss']
            axes[1, 0].plot(df['step'], df['train/value_loss'],
                          label=name.capitalize(), color=color, alpha=0.7)
        
        # Policy loss
        if 'train/policy_gradient_loss' in data:
            df = data['train/policy_gradient_loss']
            axes[1, 1].plot(df['step'], df['train/policy_gradient_loss'],
                          label=name.capitalize(), color=color, alpha=0.7)
    
    axes[0, 0].set_xlabel('Timesteps')
    axes[0, 0].set_ylabel('Mean Episode Reward')
    axes[0, 0].set_title('Training Reward')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_xlabel('Timesteps')
    axes[0, 1].set_ylabel('Mean Episode Length')
    axes[0, 1].set_title('Episode Length')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_xlabel('Timesteps')
    axes[1, 0].set_ylabel('Value Loss')
    axes[1, 0].set_title('Value Function Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_xlabel('Timesteps')
    axes[1, 1].set_ylabel('Policy Gradient Loss')
    axes[1, 1].set_title('Policy Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved training curves to {save_path}")
    else:
        plt.show()


def plot_evaluation_dashboard(stats: Dict, config: MultiAgentConfig, 
                              title: str = "Model Evaluation",
                              save_path: Optional[str] = None):
    """Create a comprehensive evaluation dashboard."""
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)
    
    # Win rate pie chart
    ax1 = fig.add_subplot(gs[0, 0])
    win_data = [stats['pursuer_win_rate'], stats['evader_win_rate'], stats['draw_rate']]
    labels = ['Pursuer Win', 'Evader Win', 'Draw/Timeout']
    colors = ['#e41a1c', '#377eb8', '#999999']
    explode = (0.05, 0.05, 0)
    ax1.pie(win_data, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
           startangle=90, textprops={'fontsize': 10})
    ax1.set_title('Win Rate Distribution', fontsize=12, fontweight='bold')
    
    # Episode length histogram
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(stats['episode_lengths'], bins=20, color='#4daf4a', edgecolor='black', alpha=0.7)
    ax2.axvline(np.mean(stats['episode_lengths']), color='red', linestyle='--', 
               label=f'Mean: {np.mean(stats["episode_lengths"]):.1f}')
    ax2.set_xlabel('Episode Length (steps)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Episode Length Distribution', fontsize=12, fontweight='bold')
    ax2.legend()
    
    # Captures/Escapes bar chart
    ax3 = fig.add_subplot(gs[0, 2])
    x = np.arange(len(stats['captures_per_episode']))
    width = 0.4
    ax3.bar(x - width/2, stats['captures_per_episode'], width, label='Captures', color='#e41a1c', alpha=0.7)
    ax3.bar(x + width/2, stats['escapes_per_episode'], width, label='Escapes', color='#377eb8', alpha=0.7)
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Count')
    ax3.set_title('Captures vs Escapes per Episode', fontsize=12, fontweight='bold')
    ax3.legend()
    
    # Time to first event
    ax4 = fig.add_subplot(gs[1, 0])
    data_to_plot = []
    labels_box = []
    if stats['time_to_first_capture']:
        data_to_plot.append(stats['time_to_first_capture'])
        labels_box.append('First Capture')
    if stats['time_to_first_escape']:
        data_to_plot.append(stats['time_to_first_escape'])
        labels_box.append('First Escape')
    
    if data_to_plot:
        bp = ax4.boxplot(data_to_plot, labels=labels_box, patch_artist=True)
        colors_box = ['#e41a1c', '#377eb8']
        for patch, color in zip(bp['boxes'], colors_box[:len(data_to_plot)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
    ax4.set_ylabel('Timesteps')
    ax4.set_title('Time to First Event', fontsize=12, fontweight='bold')
    
    # Pursuer distance traveled
    ax5 = fig.add_subplot(gs[1, 1])
    pursuer_dist_means = [np.mean(d) for d in stats['pursuer_distances']]
    pursuer_dist_stds = [np.std(d) for d in stats['pursuer_distances']]
    x_pursuers = range(1, len(pursuer_dist_means) + 1)
    ax5.bar(x_pursuers, pursuer_dist_means, yerr=pursuer_dist_stds, 
           color='#e41a1c', alpha=0.7, capsize=5, edgecolor='black')
    ax5.set_xlabel('Pursuer ID')
    ax5.set_ylabel('Distance (m)')
    ax5.set_title('Pursuer Distance Traveled', fontsize=12, fontweight='bold')
    ax5.set_xticks(x_pursuers)
    
    # Evader distance traveled
    ax6 = fig.add_subplot(gs[1, 2])
    evader_dist_means = [np.mean(d) for d in stats['evader_distances']]
    evader_dist_stds = [np.std(d) for d in stats['evader_distances']]
    x_evaders = range(1, len(evader_dist_means) + 1)
    ax6.bar(x_evaders, evader_dist_means, yerr=evader_dist_stds,
           color='#377eb8', alpha=0.7, capsize=5, edgecolor='black')
    ax6.set_xlabel('Evader ID')
    ax6.set_ylabel('Distance (m)')
    ax6.set_title('Evader Distance Traveled', fontsize=12, fontweight='bold')
    ax6.set_xticks(x_evaders)
    
    # Summary statistics table
    ax7 = fig.add_subplot(gs[2, :])
    ax7.axis('off')
    
    summary_data = [
        ['Metric', 'Value', 'Std Dev'],
        ['Episodes Evaluated', f'{len(stats["episode_lengths"])}', '-'],
        ['Mean Episode Length', f'{np.mean(stats["episode_lengths"]):.1f}', f'{np.std(stats["episode_lengths"]):.1f}'],
        ['Mean Captures', f'{np.mean(stats["captures_per_episode"]):.2f}', f'{np.std(stats["captures_per_episode"]):.2f}'],
        ['Mean Escapes', f'{np.mean(stats["escapes_per_episode"]):.2f}', f'{np.std(stats["escapes_per_episode"]):.2f}'],
        ['Pursuer Win Rate', f'{stats["pursuer_win_rate"]*100:.1f}%', '-'],
        ['Evader Win Rate', f'{stats["evader_win_rate"]*100:.1f}%', '-'],
        ['Mean Time to First Capture', 
         f'{np.mean(stats["time_to_first_capture"]):.1f}' if stats['time_to_first_capture'] else 'N/A',
         f'{np.std(stats["time_to_first_capture"]):.1f}' if stats['time_to_first_capture'] else '-'],
        ['Mean Time to First Escape',
         f'{np.mean(stats["time_to_first_escape"]):.1f}' if stats['time_to_first_escape'] else 'N/A',
         f'{np.std(stats["time_to_first_escape"]):.1f}' if stats['time_to_first_escape'] else '-'],
    ]
    
    table = ax7.table(cellText=summary_data[1:], colLabels=summary_data[0],
                      cellLoc='center', loc='center',
                      colWidths=[0.3, 0.2, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.5)
    
    # Style header
    for i in range(3):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    # Alternate row colors
    for i in range(1, len(summary_data)):
        for j in range(3):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#D9E2F3')
    
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved evaluation dashboard to {save_path}")
    else:
        plt.show()
    
    return fig


def compare_models(config: MultiAgentConfig,
                   model_configs: List[Dict],
                   num_episodes: int = 30,
                   save_path: Optional[str] = None):
    """Compare multiple model configurations."""
    
    all_stats = {}
    
    for mc in model_configs:
        name = mc.get('name', 'Unknown')
        print(f"\nEvaluating: {name}")
        
        stats = evaluate_models(
            config,
            pursuer_model_path=mc.get('pursuer_model'),
            evader_model_path=mc.get('evader_model'),
            num_episodes=num_episodes
        )
        all_stats[name] = stats
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    names = list(all_stats.keys())
    x = np.arange(len(names))
    width = 0.25
    
    # Win rates
    ax = axes[0, 0]
    pursuer_wins = [all_stats[n]['pursuer_win_rate'] * 100 for n in names]
    evader_wins = [all_stats[n]['evader_win_rate'] * 100 for n in names]
    draws = [all_stats[n]['draw_rate'] * 100 for n in names]
    
    ax.bar(x - width, pursuer_wins, width, label='Pursuer Win', color='#e41a1c')
    ax.bar(x, evader_wins, width, label='Evader Win', color='#377eb8')
    ax.bar(x + width, draws, width, label='Draw', color='#999999')
    ax.set_ylabel('Win Rate (%)')
    ax.set_title('Win Rate Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Episode lengths
    ax = axes[0, 1]
    means = [np.mean(all_stats[n]['episode_lengths']) for n in names]
    stds = [np.std(all_stats[n]['episode_lengths']) for n in names]
    ax.bar(x, means, yerr=stds, color='#4daf4a', capsize=5, edgecolor='black')
    ax.set_ylabel('Episode Length (steps)')
    ax.set_title('Mean Episode Length')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Captures
    ax = axes[1, 0]
    means = [np.mean(all_stats[n]['captures_per_episode']) for n in names]
    stds = [np.std(all_stats[n]['captures_per_episode']) for n in names]
    ax.bar(x, means, yerr=stds, color='#e41a1c', alpha=0.7, capsize=5, edgecolor='black')
    ax.set_ylabel('Captures per Episode')
    ax.set_title('Mean Captures')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Escapes
    ax = axes[1, 1]
    means = [np.mean(all_stats[n]['escapes_per_episode']) for n in names]
    stds = [np.std(all_stats[n]['escapes_per_episode']) for n in names]
    ax.bar(x, means, yerr=stds, color='#377eb8', alpha=0.7, capsize=5, edgecolor='black')
    ax.set_ylabel('Escapes per Episode')
    ax.set_title('Mean Escapes')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Model Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison to {save_path}")
    else:
        plt.show()
    
    return all_stats


def main():
    parser = argparse.ArgumentParser(description='Multi-Agent Training Dashboard')
    parser.add_argument('--mode', type=str, choices=['train', 'eval', 'compare'],
                        default='eval', help='Dashboard mode')
    parser.add_argument('--pursuer-model', type=str, default=None,
                        help='Path to trained pursuer model')
    parser.add_argument('--evader-model', type=str, default=None,
                        help='Path to trained evader model')
    parser.add_argument('--pursuer-log', type=str, default=None,
                        help='Path to pursuer training logs')
    parser.add_argument('--evader-log', type=str, default=None,
                        help='Path to evader training logs')
    parser.add_argument('--episodes', type=int, default=50,
                        help='Number of evaluation episodes')
    parser.add_argument('--save-dir', type=str, default='dashboard_output',
                        help='Directory to save outputs')
    
    args = parser.parse_args()
    
    config = get_default_multi_config()
    
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    
    if args.mode == 'train':
        # Plot training curves
        log_dirs = {}
        if args.pursuer_log:
            log_dirs['pursuers'] = args.pursuer_log
        if args.evader_log:
            log_dirs['evaders'] = args.evader_log
        
        if log_dirs:
            plot_training_curves(log_dirs, save_path=f'{args.save_dir}/training_curves.png')
        else:
            print("No log directories provided for training mode")
    
    elif args.mode == 'eval':
        # Evaluate models
        print("Evaluating models...")
        stats = evaluate_models(
            config,
            pursuer_model_path=args.pursuer_model,
            evader_model_path=args.evader_model,
            num_episodes=args.episodes
        )
        
        title = "Model Evaluation"
        if args.pursuer_model and args.evader_model:
            title = "Both Models Trained"
        elif args.pursuer_model:
            title = "Trained Pursuers vs Heuristic Evaders"
        elif args.evader_model:
            title = "Heuristic Pursuers vs Trained Evaders"
        else:
            title = "Heuristic vs Heuristic (Baseline)"
        
        plot_evaluation_dashboard(stats, config, title=title,
                                 save_path=f'{args.save_dir}/evaluation_dashboard.png')
    
    elif args.mode == 'compare':
        # Compare multiple configurations
        model_configs = [
            {'name': 'Baseline', 'pursuer_model': None, 'evader_model': None},
        ]
        
        if args.pursuer_model:
            model_configs.append({
                'name': 'Trained Pursuers',
                'pursuer_model': args.pursuer_model,
                'evader_model': None
            })
        
        if args.evader_model:
            model_configs.append({
                'name': 'Trained Evaders',
                'pursuer_model': None,
                'evader_model': args.evader_model
            })
        
        if args.pursuer_model and args.evader_model:
            model_configs.append({
                'name': 'Both Trained',
                'pursuer_model': args.pursuer_model,
                'evader_model': args.evader_model
            })
        
        compare_models(config, model_configs, num_episodes=args.episodes,
                      save_path=f'{args.save_dir}/model_comparison.png')


if __name__ == '__main__':
    main()
