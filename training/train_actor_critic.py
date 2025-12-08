"""
One-step Temporal Difference Actor-Critic Training.

Used for: CartPole, LunarLander, SecretKeyGame

Hyperparameters (from paper):
- gamma = 0.99
- Actor learning rate: 0.0001
- Critic learning rate: 0.001
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Callable
from collections import deque

from models.actor_critic import ActorCritic, Actor, Critic
from environments.secret_key_game import make_secret_key_game


def make_gym_env(env_name: str, use_terc_state: bool, seed: int) -> Tuple[gym.Env, int, int]:
    """
    Create Gymnasium environment with optional random variable injection.

    Args:
        env_name: Environment name ('cartpole', 'lunarlander', 'skg25', 'skg50')
        use_terc_state: If True, use TERC-selected state (no random vars for gym envs)
        seed: Random seed

    Returns:
        env: Gymnasium environment
        state_dim: State dimension
        n_actions: Number of actions
    """
    if env_name == 'cartpole':
        env = gym.make('CartPole-v1')
        original_dim = 4
        n_actions = 2
    elif env_name == 'lunarlander':
        env = gym.make('LunarLander-v3')
        original_dim = 8
        n_actions = 4
    elif env_name in ['skg25', 'skg50']:
        n_keys = 25 if env_name == 'skg25' else 50
        env = make_secret_key_game(n_keys=n_keys, use_terc_state=use_terc_state, seed=seed)
        state_dim = 3 if use_terc_state else n_keys
        n_actions = 81
        return env, state_dim, n_actions
    else:
        raise ValueError(f"Unknown environment: {env_name}")

    # For Gym environments, state_dim depends on whether we use random vars
    if use_terc_state:
        state_dim = original_dim  # TERC = original state only
    else:
        state_dim = original_dim + 3  # Full = original + 3 random vars

    return env, state_dim, n_actions


class StateWrapper:
    """Wrapper to inject random variables into gym environment states."""

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


def train_actor_critic(
    env_name: str,
    state_type: str,
    seed: int,
    device: torch.device,
    verbose: bool = True
) -> Tuple[Actor, Dict]:
    """
    Train Actor-Critic using one-step TD learning.

    Args:
        env_name: Environment name
        state_type: 'full' or 'terc'
        seed: Random seed
        device: PyTorch device
        verbose: Print training progress

    Returns:
        actor: Trained actor network
        info: Training info dictionary
    """
    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    use_terc_state = (state_type == 'terc')

    # Create environment
    env, state_dim, n_actions = make_gym_env(env_name, use_terc_state, seed)

    # Wrap gym environments to inject random variables
    if env_name in ['cartpole', 'lunarlander']:
        inject_random = not use_terc_state  # Inject random vars only for full state
        env = StateWrapper(env, inject_random=inject_random)

    # Create actor and critic
    actor = Actor(state_dim, n_actions).to(device)
    critic = Critic(state_dim).to(device)

    # Optimizers (per paper specification)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=0.0001)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=0.001)

    # Hyperparameters
    gamma = 0.99

    # Convergence criteria
    convergence_config = {
        'cartpole': {'target': 475, 'window': 100, 'max_episodes': 2000},
        'lunarlander': {'target': 200, 'window': 100, 'max_episodes': 5000},
        'skg25': {'target': -5, 'window': 1000, 'max_episodes': 20000},
        'skg50': {'target': -5, 'window': 1000, 'max_episodes': 20000},
    }
    config = convergence_config[env_name]

    # Training loop
    episode_rewards = deque(maxlen=config['window'])
    training_rewards = []
    converged = False

    for episode in range(config['max_episodes']):
        obs, _ = env.reset(seed=seed + episode if env_name in ['cartpole', 'lunarlander'] else None)
        state = torch.FloatTensor(obs).unsqueeze(0).to(device)

        episode_reward = 0
        done = False

        while not done:
            # Get action from actor
            with torch.no_grad():
                action, log_prob = actor.get_action(state)
            action_np = action.item()

            # Take step in environment
            next_obs, reward, terminated, truncated, _ = env.step(action_np)
            done = terminated or truncated

            next_state = torch.FloatTensor(next_obs).unsqueeze(0).to(device)
            reward_tensor = torch.FloatTensor([reward]).to(device)

            # Compute TD error for critic update
            with torch.no_grad():
                next_value = critic(next_state).squeeze()
                if done:
                    target = reward_tensor
                else:
                    target = reward_tensor + gamma * next_value

            current_value = critic(state).squeeze()
            td_error = target - current_value

            # Update critic
            critic_loss = td_error.pow(2)
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            # Update actor using TD error as advantage
            # Recompute log_prob with gradient
            logits = actor(state)
            probs = F.softmax(logits, dim=-1)
            log_prob = torch.log(probs[0, action_np] + 1e-8)

            actor_loss = -log_prob * td_error.detach()
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            episode_reward += reward
            state = next_state

        episode_rewards.append(episode_reward)
        training_rewards.append(episode_reward)

        # Check convergence
        if len(episode_rewards) >= config['window']:
            avg_reward = np.mean(episode_rewards)
            if avg_reward >= config['target']:
                converged = True
                if verbose:
                    print(f"Converged at episode {episode + 1} with avg reward {avg_reward:.2f}")
                break

        # Progress logging
        if verbose and (episode + 1) % 100 == 0:
            avg = np.mean(list(episode_rewards)[-100:]) if len(episode_rewards) > 0 else 0
            print(f"Episode {episode + 1}: Avg reward (last 100): {avg:.2f}")

    # Final stats
    final_avg = np.mean(list(episode_rewards)) if episode_rewards else 0
    info = {
        'converged': converged,
        'episodes_trained': episode + 1,
        'final_avg_reward': final_avg,
        'training_rewards': training_rewards,
    }

    if verbose:
        print(f"Training complete. Final avg reward: {final_avg:.2f}")

    return actor, info


if __name__ == '__main__':
    # Test training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Quick test with CartPole
    print("\nTesting Actor-Critic training on CartPole (full state)...")
    actor, info = train_actor_critic(
        env_name='cartpole',
        state_type='full',
        seed=42,
        device=device,
        verbose=True
    )
    print(f"Training info: {info}")
