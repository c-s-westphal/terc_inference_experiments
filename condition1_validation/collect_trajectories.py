#!/usr/bin/env python3
"""
Trajectory collection for Condition 1 validation.

Collects (state, action) pairs from trained agents or optimal policies
for use in computing mutual information for Condition 1 validation.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import gymnasium as gym
from typing import Tuple, Optional, Dict
import json

from environments.secret_key_game import make_secret_key_game
from environments.ipd import make_ipd_env
from models.actor_critic import Actor
from models.ppo import PPOPolicy
from condition1_validation.subset_enumeration import get_env_config


def collect_gym_trajectories(
    env_name: str,
    n_samples: int,
    seed: int,
    model_path: Optional[Path] = None,
    device: torch.device = torch.device('cpu')
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Collect trajectories from Gym environments using trained policy or random.

    Args:
        env_name: Environment name ('cartpole', 'lunarlander', 'pendulum')
        n_samples: Number of (state, action) pairs to collect
        seed: Random seed
        model_path: Path to trained model (if None, uses random policy)
        device: PyTorch device

    Returns:
        states: Array of shape (n_samples, state_dim) - TERC variables only
        actions: Array of shape (n_samples,) or (n_samples, action_dim)
    """
    config = get_env_config(env_name)

    # Create environment
    if env_name == 'cartpole':
        env = gym.make('CartPole-v1')
        state_dim = 4  # Original state (TERC)
    elif env_name == 'lunarlander':
        env = gym.make('LunarLander-v3')
        state_dim = 8  # Original state (TERC)
    elif env_name == 'pendulum':
        env = gym.make('Pendulum-v1')
        state_dim = 3  # Original state (TERC)
    else:
        raise ValueError(f"Unknown Gym environment: {env_name}")

    # Load model if provided
    model = None
    if model_path and model_path.exists():
        if env_name == 'pendulum':
            model = PPOPolicy(state_dim, 1, action_scale=2.0)
        else:
            model = Actor(state_dim, config['n_actions'])
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

    states = []
    actions = []

    np.random.seed(seed)
    torch.manual_seed(seed)

    obs, _ = env.reset(seed=seed)
    episode_count = 0

    while len(states) < n_samples:
        # Get action
        if model is not None:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
                if env_name == 'pendulum':
                    action_tensor, _ = model.get_action(state_tensor)
                    action = action_tensor.cpu().numpy().flatten()
                else:
                    action_tensor, _ = model.get_action(state_tensor)
                    action = action_tensor.item()
        else:
            # Random policy
            action = env.action_space.sample()

        # Store TERC state (original env state without random vars)
        states.append(obs[:state_dim].copy())

        # Store action
        if env_name == 'pendulum':
            actions.append(action if isinstance(action, np.ndarray) else np.array([action]))
        else:
            actions.append(action)

        # Step environment
        obs, reward, terminated, truncated, _ = env.step(action)

        if terminated or truncated:
            episode_count += 1
            obs, _ = env.reset(seed=seed + episode_count)

    env.close()

    states = np.array(states[:n_samples])
    actions = np.array(actions[:n_samples])

    return states, actions


def collect_skg_trajectories(
    n_keys: int,
    n_samples: int,
    seed: int,
    model_path: Optional[Path] = None,
    device: torch.device = torch.device('cpu')
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Collect trajectories from Secret Key Game.

    For SKG, we use the TERC state (3 key values) and optimal/learned actions.

    Args:
        n_keys: Number of keys (25 or 50)
        n_samples: Number of samples to collect
        seed: Random seed
        model_path: Path to trained model
        device: PyTorch device

    Returns:
        states: Array of shape (n_samples, 3) - the 3 secret key values
        actions: Array of shape (n_samples,) - discrete actions [0, 80]
    """
    env = make_secret_key_game(n_keys=n_keys, use_terc_state=True, seed=seed)

    # Load model if provided
    model = None
    if model_path and model_path.exists():
        model = Actor(3, 81)  # TERC state (3) -> 81 actions
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

    states = []
    actions = []

    np.random.seed(seed)
    torch.manual_seed(seed)

    for i in range(n_samples):
        obs, info = env.reset(seed=seed + i)

        if model is not None:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
                action_tensor, _ = model.get_action(state_tensor)
                action = action_tensor.item()
        else:
            # Optimal policy: guess the secret
            action = info['secret'] + 40  # Convert [-40, 40] to [0, 80]

        states.append(obs.copy())
        actions.append(action)

    states = np.array(states)
    actions = np.array(actions)

    return states, actions


def collect_ipd_trajectories(
    n_tats: int,
    n_samples: int,
    seed: int,
    history_length: int = 10,
    model_path: Optional[Path] = None,
    device: torch.device = torch.device('cpu')
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Collect trajectories from IPD with TF(N)T opponent.

    For IPD, TERC state is the last (n_tats - 1) actions.

    Args:
        n_tats: N for TF(N)T opponent
        n_samples: Number of samples to collect
        seed: Random seed
        history_length: Full history length
        model_path: Path to trained model
        device: PyTorch device

    Returns:
        states: Array of shape (n_samples, n_tats - 1) - TERC state
        actions: Array of shape (n_samples,) - discrete actions [0, 1]
    """
    env = make_ipd_env(
        n_tats=n_tats,
        history_length=history_length,
        use_terc_state=True,  # Use TERC state directly
        max_steps=1000,
        seed=seed
    )

    terc_size = n_tats - 1

    # Load model if provided
    model = None
    if model_path and model_path.exists():
        model = Actor(terc_size, 2)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

    states = []
    actions = []

    np.random.seed(seed)
    torch.manual_seed(seed)

    obs, _ = env.reset(seed=seed)
    episode_count = 0

    while len(states) < n_samples:
        if model is not None:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
                action_tensor, _ = model.get_action(state_tensor)
                action = action_tensor.item()
        else:
            # Simple heuristic: cooperate unless close to triggering opponent's defection
            # (This is a reasonable strategy against TF(N)T)
            recent_defections = int(obs.sum())
            if recent_defections >= n_tats - 2:
                action = 0  # Cooperate to avoid triggering
            else:
                action = np.random.randint(0, 2)  # Random

        states.append(obs.copy())
        actions.append(action)

        obs, reward, terminated, truncated, _ = env.step(action)

        if terminated or truncated:
            episode_count += 1
            obs, _ = env.reset(seed=seed + episode_count)

    states = np.array(states[:n_samples])
    actions = np.array(actions[:n_samples])

    return states, actions


def collect_trajectories(
    env_name: str,
    n_samples: int,
    seed: int,
    model_dir: Optional[Path] = None,
    device: torch.device = torch.device('cpu')
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Collect trajectories for any supported environment.

    Args:
        env_name: Environment name
        n_samples: Number of samples to collect
        seed: Random seed
        model_dir: Directory containing trained models
        device: PyTorch device

    Returns:
        states: TERC state array
        actions: Action array
    """
    # Find model path if model_dir provided
    model_path = None
    if model_dir:
        # Try to find TERC model first, then full model
        model_path = model_dir / f"{env_name}_terc_seed{seed}.pt"
        if not model_path.exists():
            model_path = model_dir / f"{env_name}_full_seed{seed}.pt"
        if not model_path.exists():
            model_path = None

    if env_name in ['cartpole', 'lunarlander', 'pendulum']:
        return collect_gym_trajectories(env_name, n_samples, seed, model_path, device)
    elif env_name == 'skg25':
        return collect_skg_trajectories(25, n_samples, seed, model_path, device)
    elif env_name == 'skg50':
        return collect_skg_trajectories(50, n_samples, seed, model_path, device)
    elif env_name.startswith('ipd_tf'):
        # Extract N from 'ipd_tfNt'
        n_tats = int(env_name.split('_tf')[1].rstrip('t'))
        return collect_ipd_trajectories(n_tats, n_samples, seed, model_path=model_path, device=device)
    else:
        raise ValueError(f"Unknown environment: {env_name}")


def save_trajectories(
    states: np.ndarray,
    actions: np.ndarray,
    env_name: str,
    seed: int,
    output_dir: Path
):
    """Save trajectories to npz file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{env_name}_seed{seed}_trajectories.npz"
    np.savez(output_file, states=states, actions=actions)
    print(f"Saved {len(states)} samples to {output_file}")


def load_trajectories(
    env_name: str,
    seed: int,
    data_dir: Path
) -> Tuple[np.ndarray, np.ndarray]:
    """Load trajectories from npz file."""
    data_file = data_dir / f"{env_name}_seed{seed}_trajectories.npz"
    if not data_file.exists():
        raise FileNotFoundError(f"Trajectory file not found: {data_file}")

    data = np.load(data_file)
    return data['states'], data['actions']


if __name__ == '__main__':
    # Test trajectory collection
    print("Testing trajectory collection...")

    # Test CartPole
    print("\n1. CartPole (random policy):")
    states, actions = collect_trajectories('cartpole', 1000, seed=42)
    print(f"   States shape: {states.shape}")
    print(f"   Actions shape: {actions.shape}")
    print(f"   Action distribution: {np.bincount(actions)}")

    # Test Pendulum
    print("\n2. Pendulum (random policy):")
    states, actions = collect_trajectories('pendulum', 1000, seed=42)
    print(f"   States shape: {states.shape}")
    print(f"   Actions shape: {actions.shape}")
    print(f"   Action range: [{actions.min():.2f}, {actions.max():.2f}]")

    # Test SKG-25
    print("\n3. SKG-25 (optimal policy):")
    states, actions = collect_trajectories('skg25', 1000, seed=42)
    print(f"   States shape: {states.shape}")
    print(f"   Actions shape: {actions.shape}")
    print(f"   State range: [{states.min()}, {states.max()}]")

    # Test IPD TF3T
    print("\n4. IPD TF3T (heuristic policy):")
    states, actions = collect_trajectories('ipd_tf3t', 1000, seed=42)
    print(f"   States shape: {states.shape}")
    print(f"   Actions shape: {actions.shape}")
    print(f"   Action distribution: {np.bincount(actions)}")

    # Test IPD TF5T
    print("\n5. IPD TF5T (heuristic policy):")
    states, actions = collect_trajectories('ipd_tf5t', 1000, seed=42)
    print(f"   States shape: {states.shape}")
    print(f"   Actions shape: {actions.shape}")

    print("\nAll trajectory collection tests passed!")
