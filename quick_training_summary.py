"""
Quick Training Summary - Extract what we can from incomplete JSON
"""

import json
import re
from pathlib import Path

def extract_partial_data():
    """Extract usable data from incomplete JSON"""
    log_path = Path('runs/rl_training_adaptive/training_log_adaptive.json')
    
    with open(log_path, 'r') as f:
        content = f.read()
    
    # Extract episode_rewards array using regex
    rewards_match = re.search(r'"episode_rewards":\s*\[([\d\.,\s]+)\]', content, re.DOTALL)
    if rewards_match:
        rewards_str = rewards_match.group(1)
        rewards = [float(x.strip()) for x in rewards_str.split(',') if x.strip()]
    else:
        rewards = []
    
    # Extract avg_losses if present
    losses_match = re.search(r'"avg_losses":\s*\[([\d\.,\s]+)\]', content, re.DOTALL)
    if losses_match:
        losses_str = losses_match.group(1)
        losses = [float(x.strip()) for x in losses_str.split(',') if x.strip() and x.strip() != 'null']
    else:
        losses = []
    
    # Extract num_episodes
    num_match = re.search(r'"num_episodes":\s*(\d+)', content)
    num_episodes = int(num_match.group(1)) if num_match else len(rewards)
    
    # Extract training_time
    time_match = re.search(r'"training_time":\s*([\d\.]+)', content)
    training_time = float(time_match.group(1)) if time_match else 0
    
    return {
        'num_episodes': num_episodes,
        'training_time': training_time,
        'episode_rewards': rewards,
        'avg_losses': losses
    }

def main():
    print("🔍 Extracting Training Data (Partial Recovery Mode)")
    print("="*80)
    
    data = extract_partial_data()
    
    rewards = data['episode_rewards']
    
    print(f"\n✅ Recovered Data:")
    print(f"   • Episodes: {data['num_episodes']}")
    print(f"   • Rewards extracted: {len(rewards)}")
    print(f"   • Training time: {data['training_time']/3600:.2f} hours")
    
    if rewards:
        import numpy as np
        print(f"\n📊 Reward Statistics:")
        print(f"   • Mean: {np.mean(rewards):.2f}")
        print(f"   • Std: {np.std(rewards):.2f}")
        print(f"   • Min: {np.min(rewards):.2f}")
        print(f"   • Max: {np.max(rewards):.2f}")
        print(f"   • Final: {rewards[-1]:.2f}")
        print(f"   • Last 100 avg: {np.mean(rewards[-100:]):.2f}")
        
        # Save summary
        output = Path('outputs') / 'training_summary.txt'
        with open(output, 'w') as f:
            f.write("DQN TRAINING SUMMARY\n")
            f.write("="*60 + "\n\n")
            f.write(f"Total Episodes: {data['num_episodes']}\n")
            f.write(f"Training Time: {data['training_time']/3600:.2f} hours\n")
            f.write(f"Final Reward: {rewards[-1]:.2f}\n")
            f.write(f"Mean Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}\n")
            f.write(f"Max Reward: {np.max(rewards):.2f}\n")
            f.write(f"Last 100 Episodes Average: {np.mean(rewards[-100:]):.2f}\n")
        
        print(f"\n✅ Summary saved to: {output}")

if __name__ == "__main__":
    main()
