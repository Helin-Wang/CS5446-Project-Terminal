"""
This module implements the PPO agent using neural networks.

Replaces the previous Q-learning tabular approach with a neural network-based PPO agent.
"""
import torch
import torch.nn as nn
import os
import numpy as np
from rl.config import MACROS
from rl.network import CNNPPONetwork


class Agent:
    def __init__(self, model_path=None, board_channels=4, scalar_dim=7, action_dim=3, hidden_dim=128):
        """
        Initialize the PPO agent
        
        Args:
            model_path: Path to load/save the model. If None, will initialize new model.
            board_channels: Number of board channels (4)
            scalar_dim: Scalar features dimension (7)
            action_dim: Action dimension (number of macros)
            hidden_dim: Hidden layer dimension
        """
        self.board_channels = board_channels
        self.scalar_dim = scalar_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # Initialize the CNN PPO network
        self.network = CNNPPONetwork(board_channels=board_channels, scalar_dim=scalar_dim, action_dim=action_dim, hidden_dim=hidden_dim)
        
        # Set device and move network to appropriate device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network.to(self.device)
        
        # Training statistics
        self.steps = 0
        self.episodes = 0
        
        # Load model if path is provided and file exists
        if model_path is not None:
            if os.path.exists(model_path):
                self.load_model(model_path)
    
    
    def act(self, state, deterministic=False):
        """
        Choose an action based on the current state
        
        Args:
            state: Current observation state (board_tensor, scalar_tensor)
            deterministic: If True, choose the most likely action
            
        Returns:
            tuple: (action, log_prob, value) where action is int, log_prob and value are float
        """
        self.steps += 1
        
        # Extract board and scalar tensors from state
        board_tensor, scalar_tensor = state
        
        # Move to correct device and add batch dimension
        device = self.device
        
        if board_tensor.dim() == 3:  # [channels, height, width]
            board_tensor = board_tensor.unsqueeze(0).to(device)  # Add batch dimension
        else:
            board_tensor = board_tensor.to(device)
            
        if scalar_tensor.dim() == 1:  # [features]
            scalar_tensor = scalar_tensor.unsqueeze(0).to(device)  # Add batch dimension
        else:
            scalar_tensor = scalar_tensor.to(device)
        
        # Get action from network
        with torch.no_grad():  # No gradients needed for inference
            action, log_prob, value = self.network.get_action(board_tensor, scalar_tensor, deterministic=deterministic)
        
        # Convert to Python types
        return action.item(), log_prob.item(), value.item()
    
    def update(self, rollout_data):
        """
        Update the network parameters using PPO training logic
        
        Args:
            rollout_data: List of episodes, where each episode is a list of transitions
                         Each transition contains: state, action, reward, terminal, etc.
        """
        if not rollout_data:
            import sys
            print("No rollout data provided for training", file=sys.stderr)
            return {
                'avg_policy_loss': 0.0,
                'avg_value_loss': 0.0,
                'avg_entropy_loss': 0.0,
                'avg_total_loss': 0.0,
                'epoch_losses': {
                    'policy': [],
                    'value': [],
                    'entropy': [],
                    'total': []
                }
            }
        
        # Process rollout data and perform PPO update
        return self._ppo_update(rollout_data)
    
    def _ppo_update(self, rollout_data, ppo_epochs=4, clip_ratio=0.1, value_clip_ratio=0.2, 
                   learning_rate=3e-4, entropy_coef=0.1, value_coef=0.5, max_grad_norm=0.5,
                   mini_batch_size=64):
        """
        Perform PPO update using rollout data with proper training design
        
        Args:
            rollout_data: List of episodes containing transitions
            ppo_epochs: Number of PPO update epochs
            clip_ratio: PPO clipping ratio for policy (reduced from 0.2 to 0.1 for more policy updates)
            value_clip_ratio: Clipping ratio for value function
            entropy_coef: Entropy regularization coefficient (increased from 0.01 to 0.1 for more exploration)
            value_coef: Value loss coefficient
            max_grad_norm: Maximum gradient norm for clipping
            mini_batch_size: Mini-batch size for training
        """
        import torch
        import torch.nn.functional as F
        import random
        
        # Set device and ensure network is on correct device
        device = self.device
        self.network.to(device)
        
        # Set up optimizer (reuse if exists, otherwise create new)
        if not hasattr(self, 'optimizer'):
            self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)
        else:
            # Update learning rate if changed
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = learning_rate
        
        # Process all episodes into a single batch
        board_states, scalar_states, actions, rewards, terminals, old_log_probs, old_values = self._process_rollout_data(rollout_data)
        
        if len(board_states) == 0:
            import sys
            print("No valid transitions found in rollout data", file=sys.stderr)
            return {
                'avg_policy_loss': 0.0,
                'avg_value_loss': 0.0,
                'avg_entropy_loss': 0.0,
                'avg_total_loss': 0.0,
                'epoch_losses': {
                    'policy': [],
                    'value': [],
                    'entropy': [],
                    'total': []
                }
            }
        
        # Convert to tensors and move to device
        board_states = torch.stack(board_states).to(device)
        scalar_states = torch.stack(scalar_states).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        terminals = torch.BoolTensor(terminals).to(device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(device)
        old_values = torch.FloatTensor(old_values).to(device)
        
        # Compute advantages and returns using GAE
        advantages, returns = self._compute_advantages_returns(rewards, terminals, old_values)
        advantages = advantages.to(device)
        returns = returns.to(device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        import sys
        print(f"PPO Update: {len(board_states)} transitions, advantages range: [{advantages.min():.3f}, {advantages.max():.3f}]", file=sys.stderr)
        
        # Create indices for mini-batch training
        dataset_size = len(board_states)
        indices = list(range(dataset_size))
        
        # Track losses across all epochs
        epoch_policy_losses = []
        epoch_value_losses = []
        epoch_entropy_losses = []
        epoch_total_losses = []
        
        # Perform PPO updates with mini-batches
        for epoch in range(ppo_epochs):
            # Shuffle indices for each epoch
            random.shuffle(indices)
            
            # Track losses for this epoch
            epoch_policy_loss = 0.0
            epoch_value_loss = 0.0
            epoch_entropy_loss = 0.0
            epoch_total_loss = 0.0
            num_batches = 0
            
            # Mini-batch training
            for start_idx in range(0, dataset_size, mini_batch_size):
                end_idx = min(start_idx + mini_batch_size, dataset_size)
                batch_indices = indices[start_idx:end_idx]
                
                # Get mini-batch
                batch_board_states = board_states[batch_indices]
                batch_scalar_states = scalar_states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_old_values = old_values[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Forward pass to get current policy and value estimates
                log_probs, values, entropy = self.network.evaluate_actions(batch_board_states, batch_scalar_states, batch_actions)
                
                # Compute ratios using stored old_log_probs
                ratios = torch.exp(log_probs - batch_old_log_probs)
                
                # Compute policy loss (clipped)
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1 - clip_ratio, 1 + clip_ratio) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Compute value loss (with clipping)
                value_pred_clipped = batch_old_values + torch.clamp(
                    values - batch_old_values, -value_clip_ratio, value_clip_ratio
                )
                value_loss1 = F.mse_loss(values, batch_returns)
                value_loss2 = F.mse_loss(value_pred_clipped, batch_returns)
                value_loss = torch.max(value_loss1, value_loss2)
                
                # Compute entropy loss
                entropy_loss = -entropy.mean()
                
                # Accumulate losses for this epoch
                epoch_policy_loss += policy_loss.item()
                epoch_value_loss += value_loss.item()
                epoch_entropy_loss += entropy_loss.item()
                epoch_total_loss += (policy_loss + value_coef * value_loss + entropy_coef * entropy_loss).item()
                num_batches += 1
                
                # Standard PPO update with single optimizer
                self.optimizer.zero_grad()
                total_loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss
                total_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_grad_norm)
                self.optimizer.step()
            
            # Average losses for this epoch
            avg_policy_loss = epoch_policy_loss / num_batches
            avg_value_loss = epoch_value_loss / num_batches
            avg_entropy_loss = epoch_entropy_loss / num_batches
            avg_total_loss = epoch_total_loss / num_batches
            
            # Store epoch losses
            epoch_policy_losses.append(avg_policy_loss)
            epoch_value_losses.append(avg_value_loss)
            epoch_entropy_losses.append(avg_entropy_loss)
            epoch_total_losses.append(avg_total_loss)
            
            if epoch == 0:  # Print only first epoch
                import sys
                print(f"Epoch {epoch}: Policy Loss: {avg_policy_loss:.4f}, Value Loss: {avg_value_loss:.4f}, Entropy Loss: {avg_entropy_loss:.4f}", file=sys.stderr)
        
        # Calculate average losses across all epochs
        avg_policy_loss = sum(epoch_policy_losses) / len(epoch_policy_losses)
        avg_value_loss = sum(epoch_value_losses) / len(epoch_value_losses)
        avg_entropy_loss = sum(epoch_entropy_losses) / len(epoch_entropy_losses)
        avg_total_loss = sum(epoch_total_losses) / len(epoch_total_losses)
        
        import sys
        print("PPO update completed", file=sys.stderr)
        
        # Return loss information
        return {
            'avg_policy_loss': avg_policy_loss,
            'avg_value_loss': avg_value_loss,
            'avg_entropy_loss': avg_entropy_loss,
            'avg_total_loss': avg_total_loss,
            'epoch_losses': {
                'policy': epoch_policy_losses,
                'value': epoch_value_losses,
                'entropy': epoch_entropy_losses,
                'total': epoch_total_losses
            }
        }
    
    def _process_rollout_data(self, rollout_data):
        """
        Process rollout data into board_states, scalar_states, actions, rewards, terminals, log_probs, values
        
        Args:
            rollout_data: List of episodes
            
        Returns:
            board_states, scalar_states, actions, rewards, terminals, log_probs, values: Lists of processed data
        """
        board_states = []
        scalar_states = []
        actions = []
        rewards = []
        terminals = []
        log_probs = []
        values = []
        
        for episode in rollout_data:
            if not episode:  # Skip empty episodes
                continue
                
            for transition in episode:
                state = transition['state']
                
                # Convert serialized state back to tensors
                if isinstance(state, dict) and 'board' in state and 'scalar' in state:
                    # New CNN format: {"board": [...], "scalar": [...]}
                    board_tensor = torch.FloatTensor(state['board'])
                    scalar_tensor = torch.FloatTensor(state['scalar'])
                    board_states.append(board_tensor)
                    scalar_states.append(scalar_tensor)
                elif isinstance(state, tuple) and len(state) == 2:
                    # Direct tensor format (for testing)
                    board_tensor, scalar_tensor = state
                    board_states.append(board_tensor)
                    scalar_states.append(scalar_tensor)
                else:
                    raise ValueError(f"Invalid state format: {type(state)} - {state}")
                
                actions.append(transition['action'])
                rewards.append(transition['reward'])
                terminals.append(transition['terminal'])
                
                # Ensure log_prob and value are present (critical for PPO)
                if 'log_prob' not in transition:
                    raise ValueError(f"Missing log_prob in transition: {transition}")
                if 'value' not in transition:
                    raise ValueError(f"Missing value in transition: {transition}")
                
                log_probs.append(transition['log_prob'])
                values.append(transition['value'])
        
        return board_states, scalar_states, actions, rewards, terminals, log_probs, values
    
    def _compute_advantages_returns(self, rewards, terminals, values, gamma=0.99, lam=0.95):
        """
        Compute advantages and returns using GAE (Generalized Advantage Estimation)
        
        Args:
            rewards: Tensor of rewards
            terminals: Tensor of terminal flags
            values: Tensor of value estimates
            gamma: Discount factor
            lam: GAE lambda parameter
            
        Returns:
            advantages, returns: Computed advantages and returns
        """
        import torch
        
        advantages = []
        returns = []
        
        # Convert to numpy for easier manipulation (move to CPU first if on GPU)
        rewards = rewards.cpu().numpy()
        terminals = terminals.cpu().numpy()
        values = values.cpu().numpy()
        
        # Scaling to reduce the range of Value Loss
        rewards = rewards / 10.0
        
        # Process each episode separately
        episode_start = 0
        for i in range(len(rewards)):
            if terminals[i] or i == len(rewards) - 1:
                # End of episode
                episode_rewards = rewards[episode_start:i+1]
                episode_terminals = terminals[episode_start:i+1]
                episode_values = values[episode_start:i+1]
                
                # Compute TD errors: δ_t = r_t + γ * V(s_{t+1}) * (1-done) - V(s_t)
                td_errors = []
                for t in range(len(episode_rewards)):
                    if t == len(episode_rewards) - 1:
                        # Last step: bootstrap with 0 if terminal, otherwise use current value
                        next_value = 0.0 if episode_terminals[t] else episode_values[t]
                    else:
                        next_value = episode_values[t + 1]
                    
                    td_error = episode_rewards[t] + gamma * next_value * (1 - episode_terminals[t]) - episode_values[t]
                    td_errors.append(td_error)
                
                # Compute GAE advantages: A_t = δ_t + γλ(1-done)*A_{t+1}
                episode_advantages = []
                running_advantage = 0
                
                # Backward pass for GAE
                for t in reversed(range(len(td_errors))):
                    if t == len(td_errors) - 1:
                        # Last step
                        running_advantage = td_errors[t]
                    else:
                        # GAE: A_t = δ_t + γλ(1-done)*A_{t+1}
                        running_advantage = td_errors[t] + gamma * lam * (1 - episode_terminals[t]) * running_advantage
                    
                    episode_advantages.append(running_advantage)
                
                # Reverse to get correct order
                episode_advantages.reverse()
                
                # Compute returns: R_t = A_t + V(s_t)
                episode_returns = [adv + val for adv, val in zip(episode_advantages, episode_values)]
                
                advantages.extend(episode_advantages)
                returns.extend(episode_returns)
                
                episode_start = i + 1
        
        return torch.FloatTensor(advantages), torch.FloatTensor(returns)
    
    def save_model(self, filepath):
        """
        Save the neural network model using PyTorch's standard method
        
        Args:
            filepath: Path to save the model
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Save the model state dict and additional info
            save_data = {
                'model_state_dict': self.network.state_dict(),
                'board_channels': self.board_channels,
                'scalar_dim': self.scalar_dim,
                'action_dim': self.action_dim,
                'hidden_dim': self.hidden_dim,
                'steps': self.steps,
                'episodes': self.episodes
            }
            
            torch.save(save_data, filepath)
            import sys
            print(f"Model saved to {filepath}", file=sys.stderr)
            return True
            
        except Exception as e:
            import sys
            print(f"Error saving model: {e}", file=sys.stderr)
            return False
    
    def load_model(self, filepath):
        """
        Load the neural network model using PyTorch's standard method
        
        Args:
            filepath: Path to load the model from
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not os.path.exists(filepath):
                import sys
                print(f"Model file {filepath} not found", file=sys.stderr)
                return False
            
            # Load the saved data
            save_data = torch.load(filepath, map_location='cpu', weights_only=False)
            
            # Load model state dict
            self.network.load_state_dict(save_data['model_state_dict'])
            
            # Move network to correct device after loading
            self.network.to(self.device)
            
            # Load additional info
            self.board_channels = save_data.get('board_channels', self.board_channels)
            self.scalar_dim = save_data.get('scalar_dim', self.scalar_dim)
            self.action_dim = save_data.get('action_dim', self.action_dim)
            self.hidden_dim = save_data.get('hidden_dim', self.hidden_dim)
            self.steps = save_data.get('steps', 0)
            self.episodes = save_data.get('episodes', 0)
            
            import sys
            print(f"Model loaded from {filepath}", file=sys.stderr)
            return True
            
        except Exception as e:
            import sys
            print(f"Error loading model: {e}", file=sys.stderr)
            return False
    
    def set_training_mode(self, training=True):
        """
        Set the network to training or evaluation mode
        
        Args:
            training: If True, set to training mode; if False, set to evaluation mode
        """
        if training:
            self.network.train()
        else:
            self.network.eval()
    
    def get_action_probabilities(self, state):
        """
        Get action probabilities for the current state
        
        Args:
            state: Current observation state (board_tensor, scalar_tensor)
            
        Returns:
            torch.Tensor: Action probabilities (converted from logits)
        """
        # Extract board and scalar tensors from state
        board_tensor, scalar_tensor = state
        
        # Add batch dimension if needed
        if board_tensor.dim() == 3:  # [channels, height, width]
            board_tensor = board_tensor.unsqueeze(0)  # Add batch dimension
        if scalar_tensor.dim() == 1:  # [features]
            scalar_tensor = scalar_tensor.unsqueeze(0)  # Add batch dimension
        
        with torch.no_grad():
            action_logits, value = self.network(board_tensor, scalar_tensor)
            action_probs = torch.softmax(action_logits, dim=-1)
        
        return action_probs.squeeze(0)  # Remove batch dimension
