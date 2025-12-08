#!/usr/bin/env python3
"""
Iterated Prisoner's Dilemma (IPD) Environment.

Implements IPD with Tit-for-N-Tats (TF(N)T) opponent strategies.
TF(N)T: Cooperate until opponent defects N times consecutively, then defect.

State representation: History of opponent's last M actions (binary: 0=cooperate, 1=defect)
Action space: Binary (0=cooperate, 1=defect)

For TF(N)T opponent:
- TERC-selected variables are the last (N-1) opponent actions
- Because the optimal response depends on whether opponent will defect,
  which depends on your last (N-1) defections
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Tuple, Dict, Any, List


# Payoff matrix for Prisoner's Dilemma
# Format: (player_payoff, opponent_payoff)
# Actions: 0 = Cooperate, 1 = Defect
PAYOFF_MATRIX = {
    (0, 0): (3, 3),  # Both cooperate: R, R (Reward)
    (0, 1): (0, 5),  # Player cooperates, opponent defects: S, T (Sucker, Temptation)
    (1, 0): (5, 0),  # Player defects, opponent cooperates: T, S
    (1, 1): (1, 1),  # Both defect: P, P (Punishment)
}


class TitForNTats:
    """
    Tit-for-N-Tats strategy.

    Cooperates until opponent defects N times consecutively, then defects.
    Forgives (returns to cooperation) after opponent cooperates once.
    """

    def __init__(self, n: int):
        """
        Args:
            n: Number of consecutive defections needed to trigger defection
        """
        self.n = n
        self.opponent_defect_streak = 0

    def reset(self):
        """Reset the strategy state."""
        self.opponent_defect_streak = 0

    def get_action(self) -> int:
        """Return action based on current state."""
        if self.opponent_defect_streak >= self.n:
            return 1  # Defect
        return 0  # Cooperate

    def update(self, opponent_action: int):
        """Update state based on opponent's action."""
        if opponent_action == 1:  # Opponent defected
            self.opponent_defect_streak += 1
        else:  # Opponent cooperated
            self.opponent_defect_streak = 0


class IPDEnvironment(gym.Env):
    """
    Iterated Prisoner's Dilemma environment with TF(N)T opponent.

    State: History of player's last M actions (needed for opponent's decision)
    Action: 0 (cooperate) or 1 (defect)
    """

    metadata = {'render_modes': []}

    def __init__(
        self,
        n_tats: int = 3,
        history_length: int = 10,
        max_steps: int = 100,
        seed: Optional[int] = None
    ):
        """
        Args:
            n_tats: N for TF(N)T opponent (defects after N consecutive player defections)
            history_length: Number of past actions to include in state
            max_steps: Maximum steps per episode
            seed: Random seed
        """
        super().__init__()

        self.n_tats = n_tats
        self.history_length = history_length
        self.max_steps = max_steps
        self._np_random = np.random.RandomState(seed)

        # State: player's last `history_length` actions (binary)
        # This is what the opponent (TF(N)T) bases its decision on
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(history_length,), dtype=np.float32
        )

        # Action: cooperate (0) or defect (1)
        self.action_space = spaces.Discrete(2)

        # Opponent strategy
        self.opponent = TitForNTats(n_tats)

        # Episode state
        self.player_history: List[int] = []
        self.step_count = 0

    def _get_state(self) -> np.ndarray:
        """Get current state (padded history of player actions)."""
        # Pad with cooperations (0) if history is shorter than required
        padded = [0] * (self.history_length - len(self.player_history)) + self.player_history[-self.history_length:]
        return np.array(padded, dtype=np.float32)

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment."""
        if seed is not None:
            self._np_random = np.random.RandomState(seed)

        self.opponent.reset()
        self.player_history = []
        self.step_count = 0

        state = self._get_state()
        info = {
            'n_tats': self.n_tats,
            'history_length': self.history_length,
        }

        return state, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step.

        Args:
            action: Player's action (0=cooperate, 1=defect)

        Returns:
            observation, reward, terminated, truncated, info
        """
        # Get opponent's action (based on player's history)
        opponent_action = self.opponent.get_action()

        # Calculate payoffs
        player_payoff, opponent_payoff = PAYOFF_MATRIX[(action, opponent_action)]

        # Update opponent's state based on player's action
        self.opponent.update(action)

        # Update player history
        self.player_history.append(action)
        self.step_count += 1

        # Get new state
        state = self._get_state()

        # Check termination
        terminated = False
        truncated = self.step_count >= self.max_steps

        info = {
            'player_action': action,
            'opponent_action': opponent_action,
            'player_payoff': player_payoff,
            'opponent_payoff': opponent_payoff,
            'step': self.step_count,
        }

        return state, float(player_payoff), terminated, truncated, info

    def get_terc_indices(self) -> List[int]:
        """
        Get indices of TERC-selected variables.

        For TF(N)T opponent, the relevant variables are the last (N-1) player actions,
        as these determine whether the opponent will switch to defection.

        Returns indices from the END of the history (most recent actions).
        """
        # The last (n_tats - 1) positions in the history
        # These are indices from the end: history_length - (n_tats - 1) to history_length - 1
        terc_size = self.n_tats - 1
        start_idx = self.history_length - terc_size
        return list(range(start_idx, self.history_length))


class IPDStateWrapper(gym.Wrapper):
    """
    Wrapper to provide either full or TERC-selected states for IPD.
    """

    def __init__(self, env: IPDEnvironment, use_terc_state: bool = False):
        super().__init__(env)
        self.use_terc_state = use_terc_state
        self.terc_indices = env.get_terc_indices()

        if use_terc_state:
            terc_size = len(self.terc_indices)
            self.observation_space = spaces.Box(
                low=0, high=1, shape=(terc_size,), dtype=np.float32
            )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if self.use_terc_state:
            obs = obs[self.terc_indices]
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self.use_terc_state:
            obs = obs[self.terc_indices]
        return obs, reward, terminated, truncated, info


def make_ipd_env(
    n_tats: int = 3,
    history_length: int = 10,
    use_terc_state: bool = False,
    max_steps: int = 100,
    seed: Optional[int] = None
) -> gym.Env:
    """
    Factory function to create IPD environment.

    Args:
        n_tats: N for TF(N)T opponent
        history_length: Number of past actions in state
        use_terc_state: If True, use only TERC-selected variables
        max_steps: Maximum steps per episode
        seed: Random seed

    Returns:
        IPD environment
    """
    env = IPDEnvironment(
        n_tats=n_tats,
        history_length=history_length,
        max_steps=max_steps,
        seed=seed
    )
    if use_terc_state:
        env = IPDStateWrapper(env, use_terc_state=True)
    return env


if __name__ == '__main__':
    # Test the IPD environment
    print("Testing IPD Environment with TF3T opponent...")

    env = make_ipd_env(n_tats=3, history_length=10, seed=42)
    obs, info = env.reset()
    print(f"Initial state shape: {obs.shape}")
    print(f"Initial state: {obs}")
    print(f"TERC indices: {env.unwrapped.get_terc_indices()}")

    total_reward = 0
    print("\nPlaying 20 steps with alternating strategy:")
    for i in range(20):
        action = i % 2  # Alternate cooperate/defect
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        print(f"Step {i+1}: action={action}, opponent={info['opponent_action']}, "
              f"reward={reward}, state={obs[-3:]}")

    print(f"\nTotal reward: {total_reward}")

    # Test with TERC state
    print("\n\nTesting with TERC state (TF3T → 2 variables):")
    env_terc = make_ipd_env(n_tats=3, history_length=10, use_terc_state=True, seed=42)
    obs_terc, _ = env_terc.reset()
    print(f"TERC state shape: {obs_terc.shape}")
    print(f"TERC state: {obs_terc}")

    # Test TF5T
    print("\n\nTesting TF5T opponent (4 TERC variables):")
    env5 = make_ipd_env(n_tats=5, history_length=10, seed=42)
    obs5, _ = env5.reset()
    print(f"Full state shape: {obs5.shape}")
    print(f"TERC indices: {env5.unwrapped.get_terc_indices()}")

    env5_terc = make_ipd_env(n_tats=5, history_length=10, use_terc_state=True, seed=42)
    obs5_terc, _ = env5_terc.reset()
    print(f"TERC state shape: {obs5_terc.shape}")
