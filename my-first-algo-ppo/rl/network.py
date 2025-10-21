import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PPONetwork(nn.Module):
    
    def __init__(self, obs_dim=6, action_dim=4, hidden_dim=64):
        super(PPONetwork, self).__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # Shared feature extraction layers
        self.shared_layers = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Actor network (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)  # Output action probabilities
        )
        
        # Critic network (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Output state value
        )
        
    def forward(self, obs):
        """
        Forward pass through the network
        
        Args:
            obs: Observation tensor of shape (batch_size, obs_dim)
            
        Returns:
            action_probs: Action probabilities of shape (batch_size, action_dim)
            value: State value of shape (batch_size, 1)
        """
        # Extract shared features
        features = self.shared_layers(obs)
        
        # Get action probabilities and state value
        action_probs = self.actor(features)
        value = self.critic(features)
        
        return action_probs, value
    
    def get_action(self, obs, deterministic=False):
        """
        Sample action from the policy
        
        Args:
            obs: Observation tensor of shape (batch_size, obs_dim)
            deterministic: If True, return the most likely action
            
        Returns:
            action: Sampled action
            log_prob: Log probability of the action
            value: State value
        """
        action_probs, value = self.forward(obs)
        
        if deterministic:
            action = torch.argmax(action_probs, dim=-1)
        else:
            # Sample action from categorical distribution
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
        
        # Calculate log probability
        log_prob = torch.log(action_probs.gather(1, action.unsqueeze(1)) + 1e-8)
        
        return action, log_prob.squeeze(1), value.squeeze(1)
    
    def evaluate_actions(self, obs, actions):
        """
        Evaluate actions for PPO loss calculation
        
        Args:
            obs: Observation tensor of shape (batch_size, obs_dim)
            actions: Action tensor of shape (batch_size,)
            
        Returns:
            log_probs: Log probabilities of actions
            values: State values
            entropy: Policy entropy
        """
        action_probs, values = self.forward(obs)
        
        # Calculate log probabilities
        log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1)) + 1e-8)
        
        # Calculate entropy for regularization
        entropy = -(action_probs * torch.log(action_probs + 1e-8)).sum(dim=-1)
        
        return log_probs.squeeze(1), values.squeeze(1), entropy
