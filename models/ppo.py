"""
PPO Networks for Continuous Action Spaces.

Architecture (per specification for Pendulum):
- Policy: Linear(input_dim, 64) -> Tanh -> Linear(64, 1) with learned std
- Value: Linear(input_dim, 64) -> Tanh -> Linear(64, 1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import numpy as np


class PPOPolicy(nn.Module):
    """
    Policy network for continuous action spaces with learned standard deviation.

    Architecture: Linear(input_dim, 64) -> Tanh -> Linear(64, action_dim)
    Outputs mean of Gaussian; std is a learned parameter.
    """

    def __init__(self, input_dim: int, action_dim: int = 1, action_scale: float = 2.0):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc_mean = nn.Linear(64, action_dim)

        # Learned log standard deviation (independent of state)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

        self.action_dim = action_dim
        self.action_scale = action_scale  # Pendulum action range is [-2, 2]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning action mean.

        Args:
            x: State tensor of shape (batch_size, input_dim)

        Returns:
            Action mean of shape (batch_size, action_dim)
        """
        x = torch.tanh(self.fc1(x))
        mean = self.fc_mean(x)
        return mean

    def get_action(
        self,
        x: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action from the policy.

        Args:
            x: State tensor
            deterministic: If True, return mean action

        Returns:
            action: Sampled/mean action (scaled to action range)
            log_prob: Log probability of the action
        """
        mean = self.forward(x)
        std = torch.exp(self.log_std).expand_as(mean)

        if deterministic:
            action = mean
        else:
            dist = torch.distributions.Normal(mean, std)
            action = dist.rsample()  # Reparameterized sample

        # Compute log probability
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(action).sum(dim=-1)

        # Apply tanh squashing and scale to action range
        action_squashed = torch.tanh(action) * self.action_scale

        return action_squashed, log_prob

    def evaluate_actions(
        self,
        x: torch.Tensor,
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate log probability and entropy for given actions.

        Note: actions should be the pre-squashed values (before tanh)

        Args:
            x: State tensor of shape (batch_size, input_dim)
            actions: Action tensor of shape (batch_size, action_dim) - raw values

        Returns:
            log_probs: Log probabilities of the actions
            entropy: Policy entropy
        """
        mean = self.forward(x)
        std = torch.exp(self.log_std).expand_as(mean)

        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        return log_prob, entropy


class PPOValue(nn.Module):
    """
    Value network for state value estimation.

    Architecture: Linear(input_dim, 64) -> Tanh -> Linear(64, 1)
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning state value.

        Args:
            x: State tensor of shape (batch_size, input_dim)

        Returns:
            Value estimate of shape (batch_size, 1)
        """
        x = torch.tanh(self.fc1(x))
        return self.fc2(x)


class PPOActorCritic(nn.Module):
    """
    Combined PPO Actor-Critic for continuous action spaces.

    This class provides both policy and value networks as a single module.
    """

    def __init__(self, input_dim: int, action_dim: int = 1, action_scale: float = 2.0):
        super().__init__()
        self.policy = PPOPolicy(input_dim, action_dim, action_scale)
        self.value = PPOValue(input_dim)
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.action_scale = action_scale

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning action mean and value estimate.

        Args:
            x: State tensor of shape (batch_size, input_dim)

        Returns:
            mean: Action mean of shape (batch_size, action_dim)
            value: Value estimate of shape (batch_size, 1)
        """
        mean = self.policy(x)
        value = self.value(x)
        return mean, value

    def get_action(
        self,
        x: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action and get value estimate.

        Args:
            x: State tensor
            deterministic: If True, return mean action

        Returns:
            action: Sampled/mean action (scaled)
            log_prob: Log probability
            value: Value estimate
        """
        action, log_prob = self.policy.get_action(x, deterministic)
        value = self.value(x)
        return action, log_prob, value.squeeze(-1)

    def evaluate_actions(
        self,
        x: torch.Tensor,
        actions_raw: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for PPO update.

        Args:
            x: State tensor
            actions_raw: Raw action values (before tanh squashing)

        Returns:
            log_probs: Log probabilities
            values: Value estimates
            entropy: Policy entropy
        """
        log_probs, entropy = self.policy.evaluate_actions(x, actions_raw)
        values = self.value(x).squeeze(-1)
        return log_probs, values, entropy


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Test the PPO models
    print("Testing PPO models...")

    # Test with different input dimensions (Pendulum)
    configs = [
        ('Pendulum Full', 6, 1),
        ('Pendulum TERC', 3, 1),
    ]

    for name, input_dim, action_dim in configs:
        model = PPOActorCritic(input_dim, action_dim)
        n_params = count_parameters(model)
        policy_params = count_parameters(model.policy)
        value_params = count_parameters(model.value)

        # Test forward pass
        x = torch.randn(32, input_dim)
        mean, value = model(x)

        # Test action sampling
        action, log_prob, val = model.get_action(x)

        print(f"{name}: input={input_dim}, action={action_dim}, "
              f"total_params={n_params}, policy_params={policy_params}, "
              f"value_params={value_params}")
        print(f"  Output shapes: mean={mean.shape}, value={value.shape}, "
              f"action={action.shape}")
        print(f"  Action range: [{action.min().item():.2f}, {action.max().item():.2f}]")
