import json
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent
OUTPUT_DIR = BASE_DIR / "paper" / "icann2026_submission" / "figures_v2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Colors and styling
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

def plot_rl_training():
    log_path = BASE_DIR / "runs" / "rl_training_v2" / "training_log_v2.json"
    if not log_path.exists():
        print(f"Error: {log_path} not found.")
        return

    with open(log_path, 'r') as f:
        data = json.load(f)
    
    rewards = data.get("episode_rewards", [])
    if not rewards:
        print("No episode rewards found.")
        return
        
    episodes = np.arange(1, len(rewards) + 1)
    
    # Calculate rolling statistics
    window = 50
    moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
    moving_avg_x = np.arange(window, len(rewards) + 1)
    
    rewards_std = []
    for i in range(len(rewards) - window + 1):
        rewards_std.append(np.std(rewards[i:i+window]))
    rewards_std = np.array(rewards_std)

    plt.figure(figsize=(8, 5), dpi=300)
    
    # Plot raw rewards with low alpha
    plt.plot(episodes, rewards, alpha=0.2, color=COLORS[0], linewidth=1)
    
    # Plot moving average
    plt.plot(moving_avg_x, moving_avg, color=COLORS[0], linewidth=2, label=f'{window}-Episode Moving Avg')
    
    # Plot standard deviation band
    plt.fill_between(moving_avg_x, moving_avg - rewards_std, moving_avg + rewards_std, color=COLORS[0], alpha=0.2)
    
    plt.title('DQN Agent Training Convergence (Safety-Aware Reward)', fontsize=14, pad=15)
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Total Reward per Episode', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12, loc='lower right')
    plt.tight_layout()
    
    out_file = OUTPUT_DIR / "RL_training.png"
    plt.savefig(out_file)
    plt.close()
    print(f"Saved {out_file}")

def plot_baselines_comparison():
    fixed_path = BASE_DIR / "outputs" / "benchmarks" / "fixed_B_baselines.json"
    rl_path = BASE_DIR / "outputs" / "benchmarks" / "rl_agent_summary_v2.json"
    
    if not fixed_path.exists() or not rl_path.exists():
        print("Missing benchmark JSON files.")
        return
        
    with open(fixed_path, 'r') as f:
        fixed_data = json.load(f)["baselines"]
        
    with open(rl_path, 'r') as f:
        rl_data = json.load(f)

    # Prepare data
    labels = ["Fixed B=6", "Fixed B=10", "Fixed B=14", "Fixed B=18", "Adaptive (RL)"]
    detections = [
        fixed_data["6"]["avg_detections"],
        fixed_data["10"]["avg_detections"],
        fixed_data["14"]["avg_detections"],
        fixed_data["18"]["avg_detections"],
        rl_data["avg_detections"]
    ]
    
    # Bandwidth savings: B=6 is 83.3, B=10 is 90.0, B=14 is 92.9, B=18 is 94.4
    bw_savings = [
        fixed_data["6"]["bw_savings"],
        fixed_data["10"]["bw_savings"],
        fixed_data["14"]["bw_savings"],
        fixed_data["18"]["bw_savings"],
        rl_data["bandwidth_savings"]
    ]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=300)

    # Plot Detections
    bars1 = ax1.bar(x - width/2, detections, width, label='Average Detections', color=COLORS[0], alpha=0.8)
    ax1.set_ylabel('Average Detections per Video', fontsize=12, color=COLORS[0])
    ax1.tick_params(axis='y', labelcolor=COLORS[0])
    ax1.set_ylim(0, max(detections) * 1.25)
    
    # Plot Bandwidth Savings on secondary axis
    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width/2, bw_savings, width, label='Bandwidth Savings (%)', color=COLORS[1], alpha=0.8)
    ax2.set_ylabel('Bandwidth Savings (%)', fontsize=12, color=COLORS[1])
    ax2.tick_params(axis='y', labelcolor=COLORS[1])
    ax2.set_ylim(0, 110)

    # Labels and Titles
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=11, fontweight='bold')
    plt.title('Performance Comparison: Fixed Ratios vs. Scene-Adaptive RL', fontsize=15, pad=15)
    
    # Value annotations
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, yval + 2, f"{yval:.1f}", ha='center', va='bottom', fontsize=10, color='black')
        
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, yval + 2, f"{yval:.1f}%", ha='center', va='bottom', fontsize=10, color='black')

    # Legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=12)

    plt.tight_layout()
    out_file = OUTPUT_DIR / "baselines_comparison.png"
    plt.savefig(out_file, bbox_inches='tight')
    plt.close()
    print(f"Saved {out_file}")

if __name__ == "__main__":
    print("Generating figures...")
    try:
        plot_rl_training()
    except Exception as e:
        print(f"Failed to plot RL training: {e}")
        # fallback if seaborn is not available
        plt.style.use('default')
        plot_rl_training()
        
    try:
        plot_baselines_comparison()
    except Exception as e:
        print(f"Failed to plot baselines comparison: {e}")
        plt.style.use('default')
        plot_baselines_comparison()
    
    print("Done!")
