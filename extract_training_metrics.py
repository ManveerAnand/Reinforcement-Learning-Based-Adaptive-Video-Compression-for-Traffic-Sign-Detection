"""
Extract and Visualize Training Metrics
=======================================
Reads training logs and generates publication-ready plots and tables.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_training_log():
    """Load DQN training log"""
    log_path = Path('runs/rl_training_adaptive/training_log_adaptive.json')
    
    if not log_path.exists():
        print(f"❌ Training log not found at {log_path}")
        return None
    
    # Read file line by line to handle potential JSON formatting issues
    with open(log_path, 'r') as f:
        content = f.read()
        # Try to load, if fails, try to fix common issues
        try:
            log_data = json.loads(content)
        except json.JSONDecodeError as e:
            print(f"⚠️ JSON formatting issue at char {e.pos}, trying to recover...")
            # Truncate at error position and try again
            content = content[:e.pos] + '\n}'
            try:
                log_data = json.loads(content)
                print("✅ Recovered partial data")
            except:
                print(f"❌ Could not recover from JSON error: {e}")
                return None
    
    # Handle different JSON structures
    if isinstance(log_data, dict):
        num_episodes = log_data.get('num_episodes', len(log_data.get('episode_rewards', [])))
    else:
        num_episodes = len(log_data)
    
    print(f"✅ Loaded training log with {num_episodes} episodes")
    return log_data

def analyze_training_progress(log_data):
    """Analyze training convergence"""
    print("\n" + "="*80)
    print(" "*25 + "TRAINING ANALYSIS")
    print("="*80)
    
    # Extract key metrics - handle different JSON structures
    if isinstance(log_data, dict):
        # New format: {num_episodes, episode_rewards, avg_losses, etc}
        rewards = log_data.get('episode_rewards', [])
        losses = log_data.get('avg_losses', [])
        epsilons = log_data.get('epsilons', [])
        episodes = list(range(1, len(rewards) + 1))
    else:
        # Old format: array of episode objects
        episodes = [entry['episode'] for entry in log_data]
        rewards = [entry['episode_reward'] for entry in log_data]
        losses = [entry.get('avg_loss', 0) for entry in log_data]
        epsilons = [entry.get('epsilon', 0) for entry in log_data]
    
    # Calculate statistics
    total_episodes = len(episodes)
    final_reward = rewards[-1] if rewards else 0
    max_reward = max(rewards) if rewards else 0
    avg_reward_last_100 = np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards)
    
    # Find convergence point (when reward stabilizes)
    window_size = 50
    if len(rewards) >= window_size * 2:
        moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        variance = [np.var(moving_avg[max(0, i-50):i+1]) for i in range(len(moving_avg))]
        convergence_episode = np.argmin(variance[100:]) + 100 if len(variance) > 100 else len(rewards)
    else:
        convergence_episode = total_episodes
    
    print(f"\n📊 Training Statistics:")
    print(f"   • Total Episodes: {total_episodes}")
    print(f"   • Final Episode Reward: {final_reward:.2f}")
    print(f"   • Maximum Reward: {max_reward:.2f}")
    print(f"   • Average Reward (Last 100): {avg_reward_last_100:.2f}")
    print(f"   • Convergence Episode: ~{convergence_episode}")
    print(f"   • Final Epsilon: {epsilons[-1]:.4f}" if epsilons else "   • Epsilon: N/A")
    
    return {
        'episodes': episodes,
        'rewards': rewards,
        'losses': losses,
        'epsilons': epsilons,
        'convergence_episode': convergence_episode,
        'avg_reward_last_100': avg_reward_last_100
    }

def plot_training_curves(metrics):
    """Generate publication-ready training plots"""
    print("\n📈 Generating training curves...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('DQN Training Progress', fontsize=16, fontweight='bold')
    
    # Plot 1: Episode Rewards
    ax1 = axes[0, 0]
    ax1.plot(metrics['episodes'], metrics['rewards'], alpha=0.3, color='blue', label='Raw')
    
    # Moving average
    window = 50
    if len(metrics['rewards']) >= window:
        moving_avg = np.convolve(metrics['rewards'], np.ones(window)/window, mode='valid')
        ax1.plot(metrics['episodes'][window-1:], moving_avg, color='red', linewidth=2, label=f'{window}-Episode MA')
    
    ax1.axvline(metrics['convergence_episode'], color='green', linestyle='--', alpha=0.7, label='Convergence')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Episode Reward')
    ax1.set_title('Episode Rewards Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Loss
    ax2 = axes[0, 1]
    if metrics['losses'] and any(metrics['losses']):
        ax2.plot(metrics['episodes'], metrics['losses'], color='orange', alpha=0.5)
        if len(metrics['losses']) >= window:
            loss_ma = np.convolve(metrics['losses'], np.ones(window)/window, mode='valid')
            ax2.plot(metrics['episodes'][window-1:], loss_ma, color='red', linewidth=2)
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Average Loss')
        ax2.set_title('Training Loss')
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'Loss data not available', ha='center', va='center', transform=ax2.transAxes)
    
    # Plot 3: Epsilon Decay
    ax3 = axes[1, 0]
    if metrics['epsilons'] and any(metrics['epsilons']):
        ax3.plot(metrics['episodes'], metrics['epsilons'], color='purple', linewidth=2)
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Epsilon (Exploration Rate)')
        ax3.set_title('Exploration Rate Decay')
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'Epsilon data not available', ha='center', va='center', transform=ax3.transAxes)
    
    # Plot 4: Reward Distribution (Last 100 episodes)
    ax4 = axes[1, 1]
    last_100_rewards = metrics['rewards'][-100:] if len(metrics['rewards']) >= 100 else metrics['rewards']
    ax4.hist(last_100_rewards, bins=30, color='green', alpha=0.7, edgecolor='black')
    ax4.axvline(np.mean(last_100_rewards), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(last_100_rewards):.2f}')
    ax4.set_xlabel('Episode Reward')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Reward Distribution (Last 100 Episodes)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_path = Path('outputs/figures')
    output_path.mkdir(parents=True, exist_ok=True)
    plot_file = output_path / 'training_curves.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"✅ Training curves saved to: {plot_file}")
    
    # Also show the plot
    # plt.show()
    plt.close()

def generate_training_table(metrics):
    """Generate LaTeX table for training results"""
    print("\n📝 Generating training table...")
    
    output_file = Path('outputs') / 'training_metrics_table.tex'
    
    with open(output_file, 'w') as f:
        f.write("% Training Metrics Table\n")
        f.write("\\begin{table}[t]\n")
        f.write("\\centering\n")
        f.write("\\caption{DQN Training Performance Metrics}\n")
        f.write("\\label{tab:training_metrics}\n")
        f.write("\\begin{tabular}{lc}\n")
        f.write("\\hline\n")
        f.write("Metric & Value \\\\\n")
        f.write("\\hline\n")
        f.write(f"Total Episodes & {len(metrics['episodes'])} \\\\\n")
        f.write(f"Convergence Episode & {metrics['convergence_episode']} \\\\\n")
        f.write(f"Final Reward & {metrics['rewards'][-1]:.2f} \\\\\n")
        f.write(f"Maximum Reward & {max(metrics['rewards']):.2f} \\\\\n")
        f.write(f"Avg Reward (Last 100) & {metrics['avg_reward_last_100']:.2f} \\\\\n")
        if metrics['epsilons'] and any(metrics['epsilons']):
            f.write(f"Final Epsilon & {metrics['epsilons'][-1]:.4f} \\\\\n")
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    
    print(f"✅ LaTeX table saved to: {output_file}")

def main():
    print("\n" + "🚀 Extracting Training Metrics...")
    print("="*80)
    
    # Load training log
    log_data = load_training_log()
    
    if log_data is None:
        print("\n❌ Cannot proceed without training log")
        return
    
    # Analyze training
    metrics = analyze_training_progress(log_data)
    
    # Generate plots
    plot_training_curves(metrics)
    
    # Generate LaTeX table
    generate_training_table(metrics)
    
    print("\n✅ Training metrics extraction complete!")

if __name__ == "__main__":
    main()
