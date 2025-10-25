#!/usr/bin/env python3
"""
Self-Play RL Training Script for Terminal Game
Implements self-play training mechanism where the agent learns by playing against itself
"""

import os
import subprocess
import sys
import json
import time
import pickle
import shutil
from pathlib import Path
from train_rl import RLTrainer

class SelfPlayTrainer(RLTrainer):
    def __init__(self, algo_path, opponent_path=None, epochs=10, save_interval=10, batch_size=5, 
                 opponent_update_interval=5, selfplay_start_epoch=3, log_prefix=None):
        """
        Initialize Self-Play RL Trainer
        
        Args:
            algo_path: Path to our RL algorithm
            opponent_path: Path to initial opponent algorithm (default: python-algo-weak)
            epochs: Number of epochs to train
            save_interval: Save model every N games
            batch_size: Number of games to train in each batch
            opponent_update_interval: Update opponent every N epochs
            selfplay_start_epoch: Start self-play from this epoch (to allow initial learning)
            log_prefix: Prefix for log files to avoid conflicts (default: "selfplay")
        """
        super().__init__(algo_path, opponent_path, epochs, save_interval, batch_size, log_prefix)
        
        # Self-play specific parameters
        self.opponent_update_interval = opponent_update_interval
        self.selfplay_start_epoch = selfplay_start_epoch
        self.current_opponent_epoch = 0
        
        # Track self-play statistics
        self.selfplay_stats = {
            'opponent_updates': 0,
            'last_opponent_epoch': 0,
            'win_rate_vs_self': [],
            'recent_win_rates': []  # Track recent win rates for dynamic updates
        }
        
        print(f"Self-Play Trainer initialized:")
        print(f"  - Dynamic opponent update: When last 3 epochs all have win rate ≥ 80%")
        print(f"  - Fallback update interval: {self.opponent_update_interval} epochs")
        print(f"  - Self-play starts at epoch: {self.selfplay_start_epoch}")
        print(f"  - Opponent directory: opponent-strategy")
    
    def _should_update_opponent(self, epoch):
        """
        Determine if opponent should be updated this epoch based on win rate
        
        Args:
            epoch: Current epoch number
            
        Returns:
            bool: True if opponent should be updated
        """
        # Don't update before self-play starts
        if epoch < self.selfplay_start_epoch:
            return False
            
        # Update on the first self-play epoch
        if epoch == self.selfplay_start_epoch:
            return True
            
        # Dynamic update based on recent win rate
        # Need at least 3 epochs of self-play data
        if len(self.selfplay_stats['recent_win_rates']) >= 3:
            # Check if last 3 epochs all have win rate ≥ 80%
            last_3_win_rates = self.selfplay_stats['recent_win_rates'][-3:]
            all_above_80 = all(win_rate >= 0.8 for win_rate in last_3_win_rates)
            
            if all_above_80:
                print(f"Dynamic opponent update triggered!")
                print(f"  - Last 3 epochs win rates: {[f'{wr:.1%}' for wr in last_3_win_rates]}")
                print(f"  - All ≥ 80% threshold: ✓")
                return True
        
        # Fallback: Update at specified intervals if no dynamic update
        return epoch % self.opponent_update_interval == 0
    
    def _update_opponent_to_current_model(self, epoch):
        """
        Update opponent to use current model
        
        Args:
            epoch: Current epoch number
        """
        try:
            print(f"\n{'='*50}")
            print(f"UPDATING OPPONENT TO CURRENT MODEL (Epoch {epoch})")
            print(f"{'='*50}")
            
            # Copy current model to opponent-strategy directory
            opponent_model_path = os.path.join(self.parent_dir, "opponent-strategy", "opponent_model.pkl")
            
            if os.path.exists(self.model_save_path):
                shutil.copy2(self.model_save_path, opponent_model_path)
                print(f"Copied current model to: {opponent_model_path}")
            else:
                print(f"Warning: Current model not found at {self.model_save_path}")
                return False
            
            # Update opponent path to use opponent-strategy directory
            self.opponent_path = os.path.join(self.parent_dir, "opponent-strategy")
            self.current_opponent_epoch = epoch
            
            # Update statistics
            self.selfplay_stats['opponent_updates'] += 1
            self.selfplay_stats['last_opponent_epoch'] = epoch
            
            print(f"Opponent updated successfully!")
            print(f"  - Opponent model: {opponent_model_path}")
            print(f"  - Opponent directory: {self.opponent_path}")
            print(f"  - Total opponent updates: {self.selfplay_stats['opponent_updates']}")
            print(f"{'='*50}\n")
            
            return True
            
        except Exception as e:
            print(f"Error updating opponent: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    
    def train(self):
        """Main self-play training loop"""
        print("=" * 60)
        print("STARTING SELF-PLAY RL TRAINING")
        print("=" * 60)
        
        for epoch in range(1, self.epochs + 1):
            print(f"\\nStarting epoch {epoch}/{self.epochs}")
            
            # Check if we should update opponent
            if self._should_update_opponent(epoch):
                success = self._update_opponent_to_current_model(epoch)
                if not success:
                    print(f"Warning: Failed to update opponent at epoch {epoch}")
            
            # Run normal training epoch
            self._run_training_epoch(epoch)
        
        # Print final self-play summary
        self._print_selfplay_summary()
    
    def _run_training_epoch(self, epoch):
        """
        Run a single training epoch (extracted from parent train method)
        
        Args:
            epoch: Current epoch number
        """
        # 1. Rollout: Run batch of games and collect rollout data
        rollout_data = []
        self.wins = 0
        self.losses = 0
        self.draws = 0
        enemy_hp_sum = 0.0
        total_action_counts = {}
        start_time = time.time()
        
        for game_num in range(1, self.batch_size + 1):
            print(f"Starting game {game_num}/{self.batch_size}")
            result = self.run_single_game(epoch, game_num)
            game_rollout = result.get('rollout_data', [])
            if 'rollout_data' not in result:
                print(f"Warning: rollout data missing for game {game_num}, defaulting to empty dataset")
            rollout_data.append(game_rollout)
            
            # Collect action counts from this game
            game_action_counts = result.get('action_counts', {})
            for action, count in game_action_counts.items():
                total_action_counts[action] = total_action_counts.get(action, 0) + count
            
            if result['result'] == 'win':
                self.wins += 1
            elif result['result'] == 'loss':
                self.losses += 1
            else:
                self.draws += 1
            
            # Collect enemy HP at game end
            if 'enemy_hp' in result:
                enemy_hp_sum += result['enemy_hp']

        total_games = self.wins + self.losses + self.draws
        win_rate = self.wins / max(1, total_games)
        
        # Count total transitions for logging
        total_transitions = sum(len(episode) for episode in rollout_data)
        
        # Calculate average enemy HP
        avg_enemy_hp = enemy_hp_sum / max(1, total_games)
        
        # 2. Train    
        # 2.1 Load Current Agent
        agent = self._load_current_agent()
        if agent is not None:
            # Count parameters manually
            total_params = sum(p.numel() for p in agent.network.parameters())
            print(f"Loaded agent with {total_params} parameters")
        else:
            print("Failed to load agent")
            return
        
        # 2.2 Update Agent with Rollout Data
        loss_info = None
        if rollout_data:
            print(f"Updating agent with {len(rollout_data)} episodes")
            # Calculate current step and total steps for entropy annealing
            current_step = epoch * self.batch_size * 50  # Approximate steps per epoch
            total_steps = self.epochs * self.batch_size * 50  # Total training steps
            loss_info = agent.update(rollout_data, current_step=current_step, total_steps=total_steps)
        else:
            print("No rollout data available for training")
        
        # 2.3 Save Agent
        agent.save_model(self.model_save_path)
        print(f"Agent saved to {self.model_save_path}")

        # 2.4 Save checkpoint at specified intervals
        if epoch % self.save_interval == 0:
            checkpoint_name = f"{self.log_prefix}_checkpoint_epoch_{epoch}.pkl"
            self.save_model(epoch, checkpoint_name)
    
        end_time = time.time()
        training_time = end_time - start_time
        
        # Log training epoch statistics
        self._log_training_epoch(epoch, self.wins, self.losses, self.draws, win_rate, training_time, total_transitions, avg_enemy_hp, loss_info, total_action_counts)
        
        # Track self-play statistics
        if epoch >= self.selfplay_start_epoch:
            self.selfplay_stats['win_rate_vs_self'].append(win_rate)
            self.selfplay_stats['recent_win_rates'].append(win_rate)
            
            # Keep only recent win rates (last 10 epochs for analysis)
            if len(self.selfplay_stats['recent_win_rates']) > 10:
                self.selfplay_stats['recent_win_rates'] = self.selfplay_stats['recent_win_rates'][-10:]
    
    def _print_selfplay_summary(self):
        """Print self-play training summary"""
        print("\\n" + "=" * 60)
        print("SELF-PLAY TRAINING SUMMARY")
        print("=" * 60)
        print(f"Total Opponent Updates: {self.selfplay_stats['opponent_updates']}")
        print(f"Last Opponent Epoch: {self.selfplay_stats['last_opponent_epoch']}")
        print(f"Self-Play Started at Epoch: {self.selfplay_start_epoch}")
        print(f"Update Strategy: Dynamic (last 3 epochs all ≥ 80%) + Fallback ({self.opponent_update_interval} epochs)")
        
        if self.selfplay_stats['win_rate_vs_self']:
            avg_win_rate_vs_self = sum(self.selfplay_stats['win_rate_vs_self']) / len(self.selfplay_stats['win_rate_vs_self'])
            print(f"Average Win Rate vs Self: {avg_win_rate_vs_self:.2%}")
            print(f"Latest Win Rate vs Self: {self.selfplay_stats['win_rate_vs_self'][-1]:.2%}")
            
            # Show recent win rates
            if len(self.selfplay_stats['recent_win_rates']) >= 3:
                last_3_rates = self.selfplay_stats['recent_win_rates'][-3:]
                all_above_80 = all(rate >= 0.8 for rate in last_3_rates)
                print(f"Last 3 Epochs: {[f'{rate:.1%}' for rate in last_3_rates]}")
                print(f"All ≥ 80%: {'✓' if all_above_80 else '✗'}")
        
        print("=" * 60)

if __name__ == "__main__":
    if len(sys.argv) < 2 or '--help' in sys.argv or '-h' in sys.argv:
        print("Usage: python train_selfplay.py <epochs> [opponent_path] [--log-prefix <prefix>]")
        print("Examples:")
        print("  python train_selfplay.py 100                           # Default 'selfplay' prefix")
        print("  python train_selfplay.py 50 ../python-algo             # vs_algo prefix")
        print("  python train_selfplay.py 20 --log-prefix custom        # Custom prefix")
        print("  python train_selfplay.py 30 -p my_experiment           # Short form")
        sys.exit(0)
    
    try:
        epochs = int(sys.argv[1])
    except ValueError:
        print("Error: epochs must be an integer")
        sys.exit(1)
    
    # Parse arguments
    opponent_path = None
    log_prefix = None
    
    i = 2
    while i < len(sys.argv):
        if sys.argv[i] in ['--log-prefix', '-p']:
            if i + 1 < len(sys.argv):
                log_prefix = sys.argv[i + 1]
                i += 2
            else:
                print("Error: --log-prefix requires a value")
                sys.exit(1)
        else:
            if opponent_path is None:
                opponent_path = sys.argv[i]
            i += 1
    
    # Get current directory (where our RL algo is)
    current_dir = os.path.dirname(os.path.realpath(__file__))
    
    # Create self-play trainer
    trainer = SelfPlayTrainer(
        algo_path=current_dir,
        opponent_path=opponent_path,
        epochs=epochs,
        save_interval=50,
        batch_size=5,                 # Correct batch size: 5 rollouts per epoch
        opponent_update_interval=50,  # Update opponent every 50 epochs (fallback)
        selfplay_start_epoch=1,       # Start self-play from epoch 1
        log_prefix=log_prefix
    )
    
    # Start self-play training
    trainer.train()
