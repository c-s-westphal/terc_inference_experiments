"""
Secret Key Game Environment

A custom RL environment where an agent must guess a secret value derived from
a polynomial interpolation of 3 secret keys within a larger state vector.

Based on the TERC paper specification:
- State: Vector of N integers, each in range [0, 10]
- Secret keys: 3 randomly chosen indices (fixed throughout game)
- Secret: y-intercept of 2nd-order Lagrange polynomial through (1, y1), (2, y2), (3, y3)
- Action space: Discrete, 81 actions representing integers [-40, 40]
- Reward: -|action - secret|
- Episode: Single step (new random state each step, secret key indices fixed)
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Tuple, Dict, Any


class SecretKeyGame(gym.Env):
    """
    Secret Key Game Environment.

    Args:
        n_keys: Total number of keys in the state vector (default: 25)
        seed: Random seed for reproducibility (default: None)
    """

    metadata = {'render_modes': []}

    def __init__(self, n_keys: int = 25, seed: Optional[int] = None):
        super().__init__()

        self.n_keys = n_keys
        self._np_random = np.random.RandomState(seed)

        # State space: N integers in [0, 10]
        self.observation_space = spaces.Box(
            low=0, high=10, shape=(n_keys,), dtype=np.float32
        )

        # Action space: 81 discrete actions representing [-40, 40]
        self.action_space = spaces.Discrete(81)

        # Select 3 secret key indices (fixed for this game instance)
        self.secret_key_indices = self._np_random.choice(n_keys, size=3, replace=False)
        self.secret_key_indices.sort()  # Sort for consistent ordering

        # Initialize state
        self.state = None
        self.secret = None

    def _action_to_guess(self, action: int) -> int:
        """Convert discrete action index to guess value [-40, 40]."""
        return action - 40

    def _compute_secret(self, state: np.ndarray) -> int:
        """
        Compute the secret value using Lagrange interpolation.

        The secret keys define points (1, y1), (2, y2), (3, y3) which determine
        a unique 2nd-order polynomial. The secret is the y-intercept (x=0).
        """
        # Get the y-values at the secret key indices
        y1 = state[self.secret_key_indices[0]]
        y2 = state[self.secret_key_indices[1]]
        y3 = state[self.secret_key_indices[2]]

        # Lagrange interpolation at x=0 for points (1, y1), (2, y2), (3, y3)
        # L0(0) = (0-2)(0-3) / (1-2)(1-3) = 6 / 2 = 3
        # L1(0) = (0-1)(0-3) / (2-1)(2-3) = 3 / -1 = -3
        # L2(0) = (0-1)(0-2) / (3-1)(3-2) = 2 / 2 = 1
        # P(0) = y1 * 3 + y2 * (-3) + y3 * 1 = 3*y1 - 3*y2 + y3

        secret = 3 * y1 - 3 * y2 + y3

        # Clamp to [-40, 40] to ensure valid action space
        secret = int(np.clip(np.round(secret), -40, 40))

        return secret

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment with a new random state."""
        if seed is not None:
            self._np_random = np.random.RandomState(seed)

        # Generate new random state
        self.state = self._np_random.randint(0, 11, size=self.n_keys).astype(np.float32)

        # Compute the secret for this state
        self.secret = self._compute_secret(self.state)

        info = {
            'secret': self.secret,
            'secret_key_indices': self.secret_key_indices.tolist(),
            'secret_key_values': self.state[self.secret_key_indices].tolist()
        }

        return self.state.copy(), info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.

        Args:
            action: Discrete action in [0, 80] representing guess in [-40, 40]

        Returns:
            observation: New state (generated fresh each step)
            reward: -|guess - secret|
            terminated: Always True (single-step episodes)
            truncated: Always False
            info: Additional information
        """
        if self.state is None:
            raise RuntimeError("Environment not reset. Call reset() first.")

        # Convert action to guess
        guess = self._action_to_guess(action)

        # Compute reward: negative absolute error
        reward = -abs(guess - self.secret)

        info = {
            'guess': guess,
            'secret': self.secret,
            'error': abs(guess - self.secret),
            'secret_key_indices': self.secret_key_indices.tolist()
        }

        # Episode terminates after single step
        terminated = True
        truncated = False

        # Generate new state for next episode (but episode is done)
        # This prepares for the next reset
        new_state = self._np_random.randint(0, 11, size=self.n_keys).astype(np.float32)

        return new_state, float(reward), terminated, truncated, info

    def get_terc_state(self, full_state: np.ndarray) -> np.ndarray:
        """
        Extract TERC-selected state (only the 3 secret key values).

        Args:
            full_state: Full state vector of shape (n_keys,)

        Returns:
            TERC state vector of shape (3,)
        """
        return full_state[self.secret_key_indices].copy()

    def get_secret_key_indices(self) -> np.ndarray:
        """Return the indices of the secret keys."""
        return self.secret_key_indices.copy()


class SecretKeyGameWrapper(gym.Wrapper):
    """
    Wrapper to provide either full or TERC-selected states.

    Args:
        env: SecretKeyGame environment
        use_terc_state: If True, observations are TERC-selected (3 dims)
    """

    def __init__(self, env: SecretKeyGame, use_terc_state: bool = False):
        super().__init__(env)
        self.use_terc_state = use_terc_state

        if use_terc_state:
            # TERC state is just the 3 secret key values
            self.observation_space = spaces.Box(
                low=0, high=10, shape=(3,), dtype=np.float32
            )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if self.use_terc_state:
            obs = self.env.get_terc_state(obs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self.use_terc_state:
            obs = self.env.get_terc_state(obs)
        return obs, reward, terminated, truncated, info


def make_secret_key_game(
    n_keys: int = 25,
    use_terc_state: bool = False,
    seed: Optional[int] = None
) -> gym.Env:
    """
    Factory function to create Secret Key Game environment.

    Args:
        n_keys: Number of keys in state vector (25 or 50)
        use_terc_state: If True, use TERC-selected state (3 dims)
        seed: Random seed

    Returns:
        Gymnasium environment
    """
    env = SecretKeyGame(n_keys=n_keys, seed=seed)
    if use_terc_state:
        env = SecretKeyGameWrapper(env, use_terc_state=True)
    return env


if __name__ == '__main__':
    # Test the environment
    print("Testing SecretKeyGame with n_keys=25...")
    env = make_secret_key_game(n_keys=25, use_terc_state=False, seed=42)
    obs, info = env.reset()
    print(f"Full state shape: {obs.shape}")
    print(f"Secret: {info['secret']}")
    print(f"Secret key indices: {info['secret_key_indices']}")
    print(f"Secret key values: {info['secret_key_values']}")

    action = 40 + info['secret']  # Optimal action
    obs, reward, done, truncated, info = env.step(action)
    print(f"Optimal action reward: {reward}")

    print("\nTesting with TERC state...")
    env_terc = make_secret_key_game(n_keys=25, use_terc_state=True, seed=42)
    obs_terc, info_terc = env_terc.reset()
    print(f"TERC state shape: {obs_terc.shape}")
    print(f"TERC state: {obs_terc}")

    print("\nTesting SecretKeyGame with n_keys=50...")
    env50 = make_secret_key_game(n_keys=50, use_terc_state=False, seed=42)
    obs50, info50 = env50.reset()
    print(f"Full state shape: {obs50.shape}")
    print(f"Secret: {info50['secret']}")
