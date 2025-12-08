"""
PPO Training for Continuous Action Spaces.

Used for: Pendulum

Hyperparameters (from paper):
- gamma = 0.99
- Actor/Policy learning rate: 0.0001
- Critic/Value learning rate: 0.001
- PPO epsilon (clip): 0.2
- Entropy coefficient: 0.001
- Mini-batch size: 64
- Update every: 2048 steps
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from collections import deque
from dataclasses import dataclass

from models.ppo import PPOActorCritic, PPOPolicy


@dataclass
class PPOBuffer:
    """Buffer to store trajectories for PPO updates."""
    states: List[np.ndarray]
    actions: List[np.ndarray]
    actions_raw: List[np.ndarray]  # Pre-squashed actions for log prob computation
    rewards: List[float]
    values: List[float]
    log_probs: List[float]
    dones: List[bool]

    def __init__(self):
        self.clear()

    def clear(self):
        self.states = []
        self.actions = []
        self.actions_raw = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

    def add(self, state, action, action_raw, reward, value, log_prob, done):
        self.states.append(state)
        self.actions.append(action)
        self.actions_raw.append(action_raw)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)

    def __len__(self):
        return len(self.states)


class PendulumStateWrapper:
    """Wrapper to inject random variables into Pendulum states."""

    def __init__(self, env: gym.Env, inject_random: bool = True, n_random_vars: int = 3):
        self.env = env
        self.inject_random = inject_random
        self.n_random_vars = n_random_vars

    def reset(self, seed: Optional[int] = None):
        obs, info = self.env.reset(seed=seed)
        if self.inject_random:
            random_vars = np.random.uniform(-5, 5, size=self.n_random_vars)
            obs = np.concatenate([obs, random_vars])
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self.inject_random:
            random_vars = np.random.uniform(-5, 5, size=self.n_random_vars)
            obs = np.concatenate([obs, random_vars])
        return obs, reward, terminated, truncated, info

    @property
    def action_space(self):
        return self.env.action_space


def compute_gae(
    rewards: List[float],
    values: List[float],
    dones: List[bool],
    next_value: float,
    gamma: float = 0.99,
    gae_lambda: float = 0.95
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Generalized Advantage Estimation (GAE).

    Args:
        rewards: List of rewards
        values: List of value estimates
        dones: List of done flags
        next_value: Value of next state
        gamma: Discount factor
        gae_lambda: GAE lambda parameter

    Returns:
        advantages: GAE advantages
        returns: Target returns for value function
    """
    n = len(rewards)
    advantages = np.zeros(n)
    returns = np.zeros(n)

    last_gae = 0
    last_value = next_value

    for t in reversed(range(n)):
        if dones[t]:
            delta = rewards[t] - values[t]
            last_gae = delta
        else:
            delta = rewards[t] + gamma * last_value - values[t]
            last_gae = delta + gamma * gae_lambda * last_gae

        advantages[t] = last_gae
        returns[t] = advantages[t] + values[t]
        last_value = values[t]

    return advantages, returns


def train_ppo(
    state_type: str,
    seed: int,
    device: torch.device,
    verbose: bool = True
) -> Tuple[PPOPolicy, Dict]:
    """
    Train PPO on Pendulum environment.

    Args:
        state_type: 'full' or 'terc'
        seed: Random seed
        device: PyTorch device
        verbose: Print training progress

    Returns:
        policy: Trained policy network
        info: Training info dictionary
    """
    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    use_terc_state = (state_type == 'terc')

    # Create environment
    env = gym.make('Pendulum-v1')
    original_dim = 3  # Pendulum has 3-dim state: [cos(theta), sin(theta), theta_dot]
    action_dim = 1

    # State dimension depends on whether we inject random variables
    if use_terc_state:
        state_dim = original_dim
        env = PendulumStateWrapper(env, inject_random=False)
    else:
        state_dim = original_dim + 3
        env = PendulumStateWrapper(env, inject_random=True)

    # Create actor-critic
    model = PPOActorCritic(state_dim, action_dim, action_scale=2.0).to(device)

    # Optimizers (per paper specification)
    policy_optimizer = torch.optim.Adam(model.policy.parameters(), lr=0.0001)
    value_optimizer = torch.optim.Adam(model.value.parameters(), lr=0.001)

    # PPO hyperparameters (from paper)
    gamma = 0.99
    gae_lambda = 0.95
    clip_epsilon = 0.2
    entropy_coef = 0.001
    batch_size = 64
    update_interval = 2048
    n_epochs = 10  # Number of epochs per update

    # Convergence criteria
    target_reward = -200
    window_size = 100
    max_episodes = 500

    # Training state
    buffer = PPOBuffer()
    episode_rewards = deque(maxlen=window_size)
    training_rewards = []
    total_steps = 0
    converged = False

    obs, _ = env.reset(seed=seed)
    episode_reward = 0
    episode_count = 0

    while episode_count < max_episodes and not converged:
        # Collect trajectories
        for _ in range(update_interval):
            state = torch.FloatTensor(obs).unsqueeze(0).to(device)

            with torch.no_grad():
                mean = model.policy(state)
                std = torch.exp(model.policy.log_std)
                dist = torch.distributions.Normal(mean, std)
                action_raw = dist.sample()
                log_prob = dist.log_prob(action_raw).sum(dim=-1)
                action = torch.tanh(action_raw) * model.action_scale
                value = model.value(state)

            action_np = action.cpu().numpy().flatten()
            action_raw_np = action_raw.cpu().numpy().flatten()

            next_obs, reward, terminated, truncated, _ = env.step(action_np)
            done = terminated or truncated

            buffer.add(
                state=obs,
                action=action_np,
                action_raw=action_raw_np,
                reward=reward,
                value=value.item(),
                log_prob=log_prob.item(),
                done=done
            )

            episode_reward += reward
            total_steps += 1
            obs = next_obs

            if done:
                episode_rewards.append(episode_reward)
                training_rewards.append(episode_reward)
                episode_count += 1

                # Check convergence
                if len(episode_rewards) >= window_size:
                    avg_reward = np.mean(episode_rewards)
                    if avg_reward >= target_reward:
                        converged = True
                        if verbose:
                            print(f"Converged at episode {episode_count} with avg reward {avg_reward:.2f}")
                        break

                # Progress logging
                if verbose and episode_count % 10 == 0:
                    avg = np.mean(list(episode_rewards)[-10:]) if len(episode_rewards) > 0 else 0
                    print(f"Episode {episode_count}: Avg reward (last 10): {avg:.2f}")

                # Reset for new episode
                obs, _ = env.reset(seed=seed + episode_count)
                episode_reward = 0

        if converged or episode_count >= max_episodes:
            break

        # PPO Update
        if len(buffer) > 0:
            # Compute returns and advantages
            with torch.no_grad():
                next_state = torch.FloatTensor(obs).unsqueeze(0).to(device)
                next_value = model.value(next_state).item()

            advantages, returns = compute_gae(
                buffer.rewards, buffer.values, buffer.dones,
                next_value, gamma, gae_lambda
            )

            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Convert to tensors
            states = torch.FloatTensor(np.array(buffer.states)).to(device)
            actions_raw = torch.FloatTensor(np.array(buffer.actions_raw)).to(device)
            old_log_probs = torch.FloatTensor(buffer.log_probs).to(device)
            advantages_t = torch.FloatTensor(advantages).to(device)
            returns_t = torch.FloatTensor(returns).to(device)

            # Multiple epochs of updates
            n_samples = len(buffer)
            indices = np.arange(n_samples)

            for _ in range(n_epochs):
                np.random.shuffle(indices)

                for start in range(0, n_samples, batch_size):
                    end = start + batch_size
                    batch_indices = indices[start:end]

                    batch_states = states[batch_indices]
                    batch_actions_raw = actions_raw[batch_indices]
                    batch_old_log_probs = old_log_probs[batch_indices]
                    batch_advantages = advantages_t[batch_indices]
                    batch_returns = returns_t[batch_indices]

                    # Get current log probs and values
                    new_log_probs, entropy = model.policy.evaluate_actions(
                        batch_states, batch_actions_raw
                    )
                    values = model.value(batch_states).squeeze(-1)

                    # Policy loss (PPO clip objective)
                    ratio = torch.exp(new_log_probs - batch_old_log_probs)
                    surr1 = ratio * batch_advantages
                    surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * batch_advantages
                    policy_loss = -torch.min(surr1, surr2).mean() - entropy_coef * entropy.mean()

                    # Value loss
                    value_loss = F.mse_loss(values, batch_returns)

                    # Update policy
                    policy_optimizer.zero_grad()
                    policy_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.policy.parameters(), 0.5)
                    policy_optimizer.step()

                    # Update value
                    value_optimizer.zero_grad()
                    value_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.value.parameters(), 0.5)
                    value_optimizer.step()

            buffer.clear()

    # Final stats
    final_avg = np.mean(list(episode_rewards)) if episode_rewards else 0
    info = {
        'converged': converged,
        'episodes_trained': episode_count,
        'final_avg_reward': final_avg,
        'total_steps': total_steps,
        'training_rewards': training_rewards,
    }

    if verbose:
        print(f"Training complete. Final avg reward: {final_avg:.2f}")

    return model.policy, info


if __name__ == '__main__':
    # Test training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("\nTesting PPO training on Pendulum (full state)...")
    policy, info = train_ppo(
        state_type='full',
        seed=42,
        device=device,
        verbose=True
    )
    print(f"Training info: converged={info['converged']}, episodes={info['episodes_trained']}")
