"""
Adaptive RL Training with Scene-Dependent Reward
=================================================
Train DQN agent with dynamic reward weighting based on scene complexity.

Key Features:
- Complex scenes (high motion/edges) → Prefer accuracy (low B)
- Simple scenes (low motion) → Prefer bandwidth (high B)
- Penalty for inappropriate B selection
- Forces agent to learn feature-dependent policies

Expected outcome: Agent adapts B based on optical flow, edge density, blur
"""

from src.phase1.video_compression_env import VideoCompressionEnv
import sys
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import json
from datetime import datetime
import time

# Add project root
sys.path.append(str(Path(__file__).parent.parent))


class ReplayBuffer:
    """Experience replay buffer for DQN"""

    def __init__(self, buffer_size=10000):
        self.buffer = deque(maxlen=buffer_size)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size=64):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )

    def __len__(self):
        return len(self.buffer)


class QNetwork(nn.Module):
    """Q-Network for DQN"""

    def __init__(self, state_size=7, action_size=3, hidden_size=128):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class DQNAgent:
    """DQN Agent with scene-adaptive learning"""

    def __init__(self, state_size=7, action_size=3, hidden_size=128, lr=0.001, gamma=0.99):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Q-Networks
        self.qnetwork_local = QNetwork(
            state_size, action_size, hidden_size).to(self.device)
        self.qnetwork_target = QNetwork(
            state_size, action_size, hidden_size).to(self.device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=lr)

        # Replay buffer
        self.memory = ReplayBuffer(buffer_size=10000)
        self.batch_size = 64
        self.update_every = 4
        self.step_count = 0

    def act(self, state, epsilon=0.1):
        """Epsilon-greedy action selection"""
        if random.random() > epsilon:
            state = torch.from_numpy(
                state).float().unsqueeze(0).to(self.device)
            self.qnetwork_local.eval()
            with torch.no_grad():
                action_values = self.qnetwork_local(state)
            self.qnetwork_local.train()
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def step_train(self, state, action, reward, next_state, done):
        """Store experience and learn"""
        self.memory.add(state, action, reward, next_state, done)

        self.step_count += 1
        if self.step_count % self.update_every == 0 and len(self.memory) >= self.batch_size:
            experiences = self.memory.sample(self.batch_size)
            self.learn(experiences)

    def learn(self, experiences):
        """Update Q-network using batch of experiences"""
        states, actions, rewards, next_states, dones = experiences

        # Convert to tensors
        states = torch.from_numpy(states).float().to(self.device)
        actions = torch.from_numpy(actions).long().unsqueeze(1).to(self.device)
        rewards = torch.from_numpy(
            rewards).float().unsqueeze(1).to(self.device)
        next_states = torch.from_numpy(next_states).float().to(self.device)
        dones = torch.from_numpy(dones.astype(
            np.uint8)).float().unsqueeze(1).to(self.device)

        # Get Q values
        Q_targets_next = self.qnetwork_target(
            next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = nn.MSELoss()(Q_expected, Q_targets)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        """Soft update target network"""
        for target_param, local_param in zip(self.qnetwork_target.parameters(),
                                             self.qnetwork_local.parameters()):
            target_param.data.copy_(local_param.data)


def train_agent(
    video_path='data/cure-tsd/data/01_01_00_00_00.mp4',
    label_path='data/cure-tsd/labels/01_01.txt',
    num_episodes=500,
    max_steps=30,  # Reduced from 300 to 30 (10-second chunks at 1fps)
    epsilon_start=1.0,
    epsilon_end=0.01,
    epsilon_decay=0.995,
    save_dir='runs/rl_training_adaptive'
):
    """
    Train DQN agent with scene-adaptive reward
    """
    print("="*80)
    print("ADAPTIVE RL TRAINING - Scene-Dependent Reward")
    print("="*80)
    print(f"Video: {video_path}")
    print(f"Labels: {label_path}")
    print(f"Episodes: {num_episodes}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print("="*80)

    # Create environment
    print("\nInitializing environment (computing initial features)...")
    env = VideoCompressionEnv(video_path, label_path)
    print("✅ Environment ready!")

    # Create agent
    print("Creating DQN agent...")
    agent = DQNAgent(state_size=7, action_size=3, hidden_size=128, lr=0.001)
    print("✅ Agent created!")
    print("\nStarting training...\n")

    # Training metrics
    episode_rewards = []
    episode_B_values = []
    episode_complexity_scores = []
    epsilon = epsilon_start

    # Create save directory
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    training_start = time.time()

    print("Episode 1 starting (first episode takes ~60s due to feature extraction)...")

    for episode in range(1, num_episodes + 1):
        episode_start = time.time()

        state, info = env.reset()
        episode_reward = 0
        B_values_episode = []
        complexity_scores_episode = []

        for step in range(max_steps):
            # Agent selects action
            action = agent.act(state, epsilon)

            # Environment step
            next_state, reward, terminated, truncated, info = env.step(action)

            # Store experience
            agent.step_train(state, action, reward, next_state, terminated)

            # Track metrics
            episode_reward += reward
            B_values_episode.append(info['B'])
            if 'reward_components' in info:
                complexity_scores_episode.append(
                    info['reward_components'].get('complexity_score', 0.5)
                )

            state = next_state

            if terminated:
                break

        # Update target network every 10 episodes
        if episode % 10 == 0:
            agent.update_target_network()

        # Decay epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        # Store episode metrics
        episode_rewards.append(episode_reward)
        avg_B = np.mean(B_values_episode)
        episode_B_values.append(avg_B)
        avg_complexity = np.mean(
            complexity_scores_episode) if complexity_scores_episode else 0.5
        episode_complexity_scores.append(avg_complexity)

        episode_time = time.time() - episode_start

        # Print progress
        if episode % 10 == 0:
            recent_rewards = episode_rewards[-10:]
            recent_B = episode_B_values[-10:]
            recent_complexity = episode_complexity_scores[-10:]

            print(f"\nEpisode {episode}/{num_episodes}")
            print(f"  Avg Reward (last 10): {np.mean(recent_rewards):.3f}")
            print(f"  Avg B (last 10): {np.mean(recent_B):.2f}")
            print(
                f"  Avg Complexity (last 10): {np.mean(recent_complexity):.3f}")
            print(f"  Epsilon: {epsilon:.3f}")
            print(f"  Buffer Size: {len(agent.memory)}")
            print(f"  Time: {episode_time:.1f}s")

        # Save checkpoint every 50 episodes
        if episode % 50 == 0:
            checkpoint = {
                'episode': episode,
                'qnetwork_local': agent.qnetwork_local.state_dict(),
                'qnetwork_target': agent.qnetwork_target.state_dict(),
                'optimizer': agent.optimizer.state_dict(),
                'epsilon': epsilon
            }
            torch.save(checkpoint, save_path / f'checkpoint_ep{episode}.pth')

    training_time = time.time() - training_start

    # Save final model
    final_checkpoint = {
        'qnetwork_local': agent.qnetwork_local.state_dict(),
        'qnetwork_target': agent.qnetwork_target.state_dict(),
        'optimizer': agent.optimizer.state_dict()
    }
    torch.save(final_checkpoint, save_path / 'best_model_adaptive.pth')

    # Save training log
    training_log = {
        'num_episodes': num_episodes,
        'training_time': training_time,
        'episode_rewards': episode_rewards,
        'episode_B_values': episode_B_values,
        'episode_complexity_scores': episode_complexity_scores,
        'final_epsilon': epsilon,
        'avg_reward': float(np.mean(episode_rewards[-50:])),
        'avg_B': float(np.mean(episode_B_values[-50:])),
        'avg_complexity': float(np.mean(episode_complexity_scores[-50:]))
    }

    with open(save_path / 'training_log_adaptive.json', 'w') as f:
        json.dump(training_log, f, indent=2)

    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"Total Time: {training_time/3600:.2f} hours")
    print(f"Final Avg Reward: {training_log['avg_reward']:.3f}")
    print(f"Final Avg B: {training_log['avg_B']:.2f}")
    print(f"Final Avg Complexity: {training_log['avg_complexity']:.3f}")
    print(f"Model saved to: {save_path / 'best_model_adaptive.pth'}")
    print("="*80)

    return agent, training_log


if __name__ == '__main__':
    # Train the agent
    agent, log = train_agent(
        video_path='data/cure-tsd/data/01_01_00_00_00.mp4',
        label_path='data/cure-tsd/labels/01_01.txt',
        num_episodes=500,
        epsilon_decay=0.995,
        save_dir='runs/rl_training_adaptive'
    )
