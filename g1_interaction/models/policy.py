# SPDX-License-Identifier: MIT

"""
Custom policy architectures for G1 retargeting.

This file can be extended with specialized policy architectures
such as recurrent policies, attention-based policies, or 
multi-modal policies for different retargeting tasks.
"""

import torch
import torch.nn as nn
from typing import Tuple


class RetargetPolicy(nn.Module):
    """
    Custom policy for human motion retargeting.
    
    This is a placeholder for future custom architectures.
    Currently uses the default ActorCritic from rsl_rl.
    """
    
    def __init__(
        self,
        num_obs: int,
        num_actions: int,
        hidden_dims: list = [512, 256, 128],
        activation: str = 'elu'
    ):
        super().__init__()
        
        self.num_obs = num_obs
        self.num_actions = num_actions
        
        # Actor network
        actor_layers = []
        actor_layers.append(nn.Linear(num_obs, hidden_dims[0]))
        actor_layers.append(self._get_activation(activation))
        
        for i in range(len(hidden_dims) - 1):
            actor_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            actor_layers.append(self._get_activation(activation))
            
        actor_layers.append(nn.Linear(hidden_dims[-1], num_actions))
        self.actor = nn.Sequential(*actor_layers)
        
        # Critic network
        critic_layers = []
        critic_layers.append(nn.Linear(num_obs, hidden_dims[0]))
        critic_layers.append(self._get_activation(activation))
        
        for i in range(len(hidden_dims) - 1):
            critic_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            critic_layers.append(self._get_activation(activation))
            
        critic_layers.append(nn.Linear(hidden_dims[-1], 1))
        self.critic = nn.Sequential(*critic_layers)
        
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            obs: Observation tensor
            
        Returns:
            Tuple of (action_mean, value)
        """
        action_mean = self.actor(obs)
        value = self.critic(obs)
        return action_mean, value
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function by name."""
        if activation == 'elu':
            return nn.ELU()
        elif activation == 'relu':
            return nn.ReLU()
        elif activation == 'tanh':
            return nn.Tanh()
        elif activation == 'leaky_relu':
            return nn.LeakyReLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")


class RecurrentRetargetPolicy(nn.Module):
    """
    Recurrent policy for temporal motion retargeting.
    
    Uses LSTM to maintain history and improve temporal consistency.
    """
    
    def __init__(
        self,
        num_obs: int,
        num_actions: int,
        hidden_dims: list = [256, 128],
        rnn_hidden_size: int = 256,
        rnn_num_layers: int = 1,
        activation: str = 'elu'
    ):
        super().__init__()
        
        self.num_obs = num_obs
        self.num_actions = num_actions
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_num_layers = rnn_num_layers
        
        # Encoder
        encoder_layers = []
        encoder_layers.append(nn.Linear(num_obs, hidden_dims[0]))
        encoder_layers.append(self._get_activation(activation))
        for i in range(len(hidden_dims) - 1):
            encoder_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            encoder_layers.append(self._get_activation(activation))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # LSTM
        self.lstm = nn.LSTM(
            hidden_dims[-1],
            rnn_hidden_size,
            rnn_num_layers,
            batch_first=True
        )
        
        # Actor head
        self.actor = nn.Linear(rnn_hidden_size, num_actions)
        
        # Critic head
        self.critic = nn.Linear(rnn_hidden_size, 1)
        
    def forward(
        self,
        obs: torch.Tensor,
        hidden_state: Tuple[torch.Tensor, torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass with recurrent state.
        
        Args:
            obs: Observation tensor (batch, obs_dim)
            hidden_state: LSTM hidden state
            
        Returns:
            Tuple of (action_mean, value, new_hidden_state)
        """
        # Encode observations
        encoded = self.encoder(obs)
        
        # Add sequence dimension
        encoded = encoded.unsqueeze(1)
        
        # LSTM forward
        if hidden_state is None:
            lstm_out, new_hidden = self.lstm(encoded)
        else:
            lstm_out, new_hidden = self.lstm(encoded, hidden_state)
            
        # Remove sequence dimension
        lstm_out = lstm_out.squeeze(1)
        
        # Actor and critic
        action_mean = self.actor(lstm_out)
        value = self.critic(lstm_out)
        
        return action_mean, value, new_hidden
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function by name."""
        if activation == 'elu':
            return nn.ELU()
        elif activation == 'relu':
            return nn.ReLU()
        elif activation == 'tanh':
            return nn.Tanh()
        elif activation == 'leaky_relu':
            return nn.LeakyReLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

