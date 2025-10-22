import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CNNPPONetwork(nn.Module):
    
    def __init__(self, board_channels=4, scalar_dim=7, action_dim=4, hidden_dim=128):
        super(CNNPPONetwork, self).__init__()
        
        self.board_channels = board_channels
        self.scalar_dim = scalar_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # CNN for spatial processing
        self.conv_layers = nn.Sequential(
            # First conv block
            nn.Conv2d(board_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 28x28 -> 14x14
            
            # Second conv block  
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 14x14 -> 7x7
            
            # Third conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))  # 7x7 -> 4x4
        )
        
        # Flatten spatial features
        self.spatial_features_dim = 128 * 4 * 4  # 2048
        
        # Combine spatial + scalar features
        self.combined_dim = self.spatial_features_dim + scalar_dim
        
        # Shared layers
        self.shared_layers = nn.Sequential(
            nn.Linear(self.combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic head (value)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, board_tensor, scalar_tensor):
        """
        Forward pass through the network
        
        Args:
            board_tensor: Board tensor of shape (batch_size, board_channels, 28, 28)
            scalar_tensor: Scalar tensor of shape (batch_size, scalar_dim)
            
        Returns:
            action_probs: Action probabilities of shape (batch_size, action_dim)
            value: State value of shape (batch_size, 1)
        """
        # Process spatial information
        spatial_features = self.conv_layers(board_tensor)
        spatial_features = spatial_features.view(spatial_features.size(0), -1)
        
        # Combine with scalar features
        combined_features = torch.cat([spatial_features, scalar_tensor], dim=1)
        
        # Get shared features
        shared_features = self.shared_layers(combined_features)
        
        # Get policy and value
        action_probs = self.actor(shared_features)
        value = self.critic(shared_features)
        
        return action_probs, value
    
    def get_action(self, board_tensor, scalar_tensor, deterministic=False):
        """
        Sample action from the policy
        
        Args:
            board_tensor: Board tensor of shape (batch_size, board_channels, 28, 28)
            scalar_tensor: Scalar tensor of shape (batch_size, scalar_dim)
            deterministic: If True, return the most likely action
            
        Returns:
            action: Sampled action
            log_prob: Log probability of the action
            value: State value
        """
        action_probs, value = self.forward(board_tensor, scalar_tensor)
        
        if deterministic:
            action = torch.argmax(action_probs, dim=-1)
        else:
            # Sample action from categorical distribution
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
        
        # Calculate log probability
        log_prob = torch.log(action_probs.gather(1, action.unsqueeze(1)) + 1e-8)
        
        return action, log_prob.squeeze(1), value.squeeze(1)
    
    def evaluate_actions(self, board_tensor, scalar_tensor, actions):
        """
        Evaluate actions for PPO loss calculation
        
        Args:
            board_tensor: Board tensor of shape (batch_size, board_channels, 28, 28)
            scalar_tensor: Scalar tensor of shape (batch_size, scalar_dim)
            actions: Action tensor of shape (batch_size,)
            
        Returns:
            log_probs: Log probabilities of actions
            values: State values
            entropy: Policy entropy
        """
        action_probs, values = self.forward(board_tensor, scalar_tensor)
        
        # Calculate log probabilities
        log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1)) + 1e-8)
        
        # Calculate entropy for regularization
        entropy = -(action_probs * torch.log(action_probs + 1e-8)).sum(dim=-1)
        
        return log_probs.squeeze(1), values.squeeze(1), entropy


# Legacy compatibility - keep the old class name for now
class PPONetwork(CNNPPONetwork):
    def __init__(self, obs_dim=6, action_dim=4, hidden_dim=64):
        # Convert old interface to new CNN interface
        super().__init__(board_channels=4, scalar_dim=7, action_dim=action_dim, hidden_dim=hidden_dim)
