import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CNNPPONetwork(nn.Module):
    
    def __init__(self, board_channels=5, scalar_dim=7, action_dim=8, hidden_dim=128):
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
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Critic head (value)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Initialize weights properly
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights properly"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        # Special initialization for actor head to encourage exploration
        with torch.no_grad():
            # Add small random noise to actor weights to break symmetry
            for layer in self.actor:
                if isinstance(layer, nn.Linear):
                    layer.weight.add_(torch.randn_like(layer.weight) * 0.01)
    
    def forward(self, board_tensor, scalar_tensor):
        """
        Forward pass through the network
        
        Args:
            board_tensor: Board tensor of shape (batch_size, board_channels, 28, 28)
            scalar_tensor: Scalar tensor of shape (batch_size, scalar_dim)
            
        Returns:
            action_logits: Action logits of shape (batch_size, action_dim)
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
        action_logits = self.actor(shared_features)
        value = self.critic(shared_features)
        
        return action_logits, value
    
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
        action_logits, value = self.forward(board_tensor, scalar_tensor)
        dist = torch.distributions.Categorical(logits=action_logits)
        
        if deterministic:
            action = torch.argmax(action_logits, dim=-1)
        else:
            action = dist.sample()
        
        log_prob = dist.log_prob(action)
        
        return action, log_prob, value.squeeze(1)
    
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
        action_logits, values = self.forward(board_tensor, scalar_tensor)
        
        dist = torch.distributions.Categorical(logits=action_logits)
        
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        return log_probs, values.squeeze(1), entropy


# Legacy compatibility - keep the old class name for now
class PPONetwork(CNNPPONetwork):
    def __init__(self, obs_dim=6, action_dim=8, hidden_dim=64):
        # Convert old interface to new CNN interface
        super().__init__(board_channels=5, scalar_dim=7, action_dim=action_dim, hidden_dim=hidden_dim)