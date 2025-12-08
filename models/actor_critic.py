"""
Actor-Critic Networks for One-step TD Learning.

Architecture (per specification):
- Actor: Linear(input_dim, 64) -> ReLU -> Linear(64, n_actions)
- Critic: Linear(input_dim, 64) -> ReLU -> Linear(64, 1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class Actor(nn.Module):
    """
    Actor network that outputs action logits for discrete action spaces.

    Architecture: Linear(input_dim, 64) -> ReLU -> Linear(64, n_actions)
    """

    def __init__(self, input_dim: int, n_actions: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning action logits.

        Args:
            x: State tensor of shape (batch_size, input_dim)

        Returns:
            Action logits of shape (batch_size, n_actions)
        """
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def get_action_probs(self, x: torch.Tensor) -> torch.Tensor:
        """Get action probabilities via softmax."""
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)

    def get_action(self, x: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action from the policy.

        Args:
            x: State tensor
            deterministic: If True, return argmax action

        Returns:
            action: Selected action
            log_prob: Log probability of the action
        """
        logits = self.forward(x)
        probs = F.softmax(logits, dim=-1)

        if deterministic:
            action = torch.argmax(probs, dim=-1)
        else:
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()

        log_prob = torch.log(probs.gather(-1, action.unsqueeze(-1)) + 1e-8).squeeze(-1)
        return action, log_prob


class Critic(nn.Module):
    """
    Critic network that outputs state value.

    Architecture: Linear(input_dim, 64) -> ReLU -> Linear(64, 1)
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
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class ActorCritic(nn.Module):
    """
    Combined Actor-Critic network.

    This class provides both actor and critic as a single module
    for convenience, while keeping them as separate networks
    with separate optimizers as per the paper specification.
    """

    def __init__(self, input_dim: int, n_actions: int):
        super().__init__()
        self.actor = Actor(input_dim, n_actions)
        self.critic = Critic(input_dim)
        self.input_dim = input_dim
        self.n_actions = n_actions

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning both action logits and value estimate.

        Args:
            x: State tensor of shape (batch_size, input_dim)

        Returns:
            logits: Action logits of shape (batch_size, n_actions)
            value: Value estimate of shape (batch_size, 1)
        """
        logits = self.actor(x)
        value = self.critic(x)
        return logits, value

    def get_action(self, x: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action and get value estimate.

        Args:
            x: State tensor
            deterministic: If True, return argmax action

        Returns:
            action: Selected action
            log_prob: Log probability of the action
            value: Value estimate
        """
        action, log_prob = self.actor.get_action(x, deterministic)
        value = self.critic(x)
        return action, log_prob, value.squeeze(-1)

    def evaluate_actions(self, x: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for given states.

        Args:
            x: State tensor of shape (batch_size, input_dim)
            actions: Action tensor of shape (batch_size,)

        Returns:
            log_probs: Log probabilities of the actions
            values: Value estimates
            entropy: Policy entropy
        """
        logits = self.actor(x)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)

        log_probs = dist.log_prob(actions)
        values = self.critic(x).squeeze(-1)
        entropy = dist.entropy()

        return log_probs, values, entropy


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Test the models
    print("Testing Actor-Critic models...")

    # Test with different input dimensions
    configs = [
        ('CartPole Full', 7, 2),
        ('CartPole TERC', 4, 2),
        ('LunarLander Full', 11, 4),
        ('LunarLander TERC', 8, 4),
        ('SKG-25 Full', 25, 81),
        ('SKG-25 TERC', 3, 81),
        ('SKG-50 Full', 50, 81),
        ('SKG-50 TERC', 3, 81),
    ]

    for name, input_dim, n_actions in configs:
        model = ActorCritic(input_dim, n_actions)
        n_params = count_parameters(model)
        actor_params = count_parameters(model.actor)
        critic_params = count_parameters(model.critic)

        # Test forward pass
        x = torch.randn(32, input_dim)
        logits, value = model(x)

        print(f"{name}: input={input_dim}, actions={n_actions}, "
              f"total_params={n_params}, actor_params={actor_params}, "
              f"critic_params={critic_params}")
        print(f"  Output shapes: logits={logits.shape}, value={value.shape}")
