#!/usr/bin/env python3
"""
RL Training Script for Terminal Game
Runs multiple games to train the reinforcement learning agent
"""

import os
import subprocess
import sys
import json
import time
import pickle
from pathlib import Path

class RLTrainer:
    def __init__(self, algo_path, opponent_path=None, epochs=10, save_interval=10, batch_size=5):
        """
        Initialize RL Trainer
        
        Args:
            algo_path: Path to our RL algorithm
            opponent_path: Path to opponent algorithm (default: python-algo)
            epochs: Number of epochs to train
            save_interval: Save model every N games
            batch_size: Number of games to train in each batch
        """
        self.algo_path = algo_path
        self.epochs = epochs
        self.save_interval = save_interval
        self.batch_size = batch_size
        
        # Get paths first
        self.file_dir = os.path.dirname(os.path.realpath(__file__))
        self.parent_dir = os.path.join(self.file_dir, os.pardir)
        self.parent_dir = os.path.abspath(self.parent_dir)
        self.is_windows = sys.platform.startswith('win')
        
        # Set opponent path after is_windows is defined
        self.opponent_path = opponent_path or self._get_default_opponent()
        
        # Training statistics
        self.wins = 0
        self.losses = 0
        self.draws = 0
        self.game_results = []
        
        # Model save path
        self.model_save_path = os.path.join(self.file_dir, "rl_model.pkl")
        
        # Training log path
        self.training_log_path = os.path.join(self.file_dir, "logs", "training_log.jsonl")
        
        # Ensure logs and checkpoints directories exist
        os.makedirs(os.path.join(self.file_dir, "logs"), exist_ok=True)
        os.makedirs(os.path.join(self.file_dir, "checkpoints"), exist_ok=True)
        
    def _get_default_opponent(self):
        """Get default opponent path"""
        if self.is_windows:
            return os.path.join(self.parent_dir, "python-algo-weak", "run.ps1")
        else:
            return os.path.join(self.parent_dir, "python-algo-weak", "run.sh")
    
    def _prepare_algo_paths(self):
        """Prepare algorithm paths for execution"""
        algo1 = self.algo_path
        algo2 = self.opponent_path
        
        # Add run files if needed
        if self.is_windows:
            if "run.ps1" not in algo1:
                trailing_char = "" if algo1.endswith("\\") else "\\"
                algo1 = algo1 + trailing_char + "run.ps1"
            if "run.ps1" not in algo2:
                trailing_char = "" if algo2.endswith("\\") else "\\"
                algo2 = algo2 + trailing_char + "run.ps1"
        else:
            if "run.sh" not in algo1:
                trailing_char = "" if algo1.endswith('/') else "/"
                algo1 = algo1 + trailing_char + "run.sh"
            if "run.sh" not in algo2:
                trailing_char = "" if algo2.endswith('/') else "/"
                algo2 = algo2 + trailing_char + "run.sh"
        
        return algo1, algo2
    
    def run_single_game(self, epoch, game_num):
        """
        Run a single game and return the result
        
        Args:
            epoch: Current epoch number
            game_num: Current game number within the batch
        
        Returns:
            dict: Game result with win/loss information
        """
        algo1, algo2 = self._prepare_algo_paths()
        
        print(f"Starting game {game_num}/{self.batch_size} (epoch {epoch})")
        print(f"RL Agent: {algo1}")
        print(f"Opponent: {algo2}")
        
        # Clear previous rollout data before starting new game
        rollout_file = os.path.join(self.file_dir, "logs", "current_turns.jsonl")
        if os.path.exists(rollout_file):
            os.remove(rollout_file)
            print(f"Cleared previous rollout data: {rollout_file}")
        
        # Run the game with game number as environment variable
        cmd = f"cd {self.parent_dir} && GAME_NUM={game_num} java -jar engine.jar work {algo1} {algo2}"
        
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            # Print the game output for debugging
            if result.stdout:
                print("Game Output:")
                print(result.stdout)
                print("End Game Output")
            
            # Parse result (this is simplified - you might need to parse the actual game output)
            game_result = self._parse_game_result(result.stdout, "")
            game_result['game_num'] = game_num
            game_result['timestamp'] = time.time()
            
            # Parse rollout data and calculate rewards
            rollout_data, action_counts = self._parse_rollout_data(game_result)
            game_result['rollout_data'] = rollout_data
            game_result['action_counts'] = action_counts

            return game_result
            
        except subprocess.TimeoutExpired:
            print(f"Game {game_num} timed out")
            return {
                'game_num': game_num,
                'result': 'timeout',
                'winner': 'none',
                'rollout_data': [],
                'action_counts': {}
            }
        except Exception as e:
            print(f"Error in game {game_num}: {e}")
            return {
                'game_num': game_num,
                'result': 'error',
                'winner': 'none',
                'rollout_data': [],
                'action_counts': {},
                'error': str(e)
            }
    
    def _parse_game_result(self, stdout, stderr):
        """
        Parse game result from output
        This is a simplified parser - you may need to adjust based on actual game output
        """
        # Look for win/loss indicators in the output
        output = stdout + stderr
        
        # Parse enemy HP from the last turn data
        enemy_hp = 0.0
        try:
            rollout_file = os.path.join(self.file_dir, "logs", "current_turns.jsonl")
            if os.path.exists(rollout_file):
                with open(rollout_file, 'r') as f:
                    lines = f.readlines()
                    if lines:
                        last_line = lines[-1].strip()
                        if last_line:
                            last_turn = json.loads(last_line)
                            enemy_hp = last_turn.get('enemy_hp', 0.0)
        except Exception as e:
            print(f"Error parsing enemy HP: {e}")
            enemy_hp = 0.0
        
        if "Player 1 wins" in output or "Player 0 wins" in output:
            if "Player 1 wins" in output:
                return {'result': 'win', 'winner': 'player1', 'score': 1, 'enemy_hp': enemy_hp}
            else:
                return {'result': 'loss', 'winner': 'player0', 'score': -1, 'enemy_hp': enemy_hp}
        elif "Draw" in output or "Tie" in output:
            return {'result': 'draw', 'winner': 'none', 'score': 0, 'enemy_hp': enemy_hp}
        else:
            # Default to loss if we can't parse
            return {'result': 'loss', 'winner': 'player0', 'score': -1, 'enemy_hp': enemy_hp}
    
    def _parse_rollout_data(self, game_result):
        """
        Parse rollout data from current_turns.jsonl and calculate rewards
        
        Args:
            game_result: Game result dictionary containing win/loss information
            
        Returns:
            list: Processed rollout data with calculated rewards
        """
        try:
            # Import RL components
            sys.path.append(self.file_dir)
            from rl.reward_tracker import RewardTracker
            
            # Path to rollout data file
            rollout_file = os.path.join(self.file_dir, "logs", "current_turns.jsonl")
            
            if not os.path.exists(rollout_file):
                print(f"Warning: Rollout file not found: {rollout_file}")
                return []
            
            # Read rollout data
            rollout_data = []
            with open(rollout_file, 'r') as f:
                for line in f:
                    if line.strip():
                        try:
                            turn_data = json.loads(line.strip())
                            rollout_data.append(turn_data)
                        except json.JSONDecodeError as e:
                            print(f"Error parsing rollout line: {e}")
                            continue
            
            if not rollout_data:
                print("Warning: No rollout data found")
                return []
            
            # Initialize reward tracker
            reward_tracker = RewardTracker()
            
            # Calculate rewards for each turn
            processed_data = []
            for i, turn_data in enumerate(rollout_data):
                # Create a mock game state for reward calculation
                class MockGameState:
                    def __init__(self, hp, enemy_hp):
                        self.my_health = hp
                        self.enemy_health = enemy_hp
                
                # Initialize reward tracker with first turn data
                if i == 0:
                    mock_state = MockGameState(turn_data['hp'], turn_data['enemy_hp'])
                    reward_tracker.reset(mock_state)
                
                # Calculate reward for this turn
                if i < len(rollout_data) - 1:
                    # Non-terminal turn: calculate reward using new combined formula
                    current_hp = turn_data['hp']
                    current_enemy_hp = turn_data['enemy_hp']
                    
                    # Get next turn data for HP comparison
                    next_turn = rollout_data[i + 1]
                    next_hp = next_turn['hp']
                    next_enemy_hp = next_turn['enemy_hp']
                    
                    # Create mock states for reward calculation
                    current_state = MockGameState(current_hp, current_enemy_hp)
                    next_state = MockGameState(next_hp, next_enemy_hp)
                    
                    # Calculate reward using new combined formula
                    action = turn_data.get('action')
                    reward = reward_tracker.compute_reward(next_state, action, turn_data)
                   
                else:
                    # Last turn: treat as terminal and use game result
                    if game_result['result'] == 'win':
                        reward = 100.0  # Large positive reward for winning
                    elif game_result['result'] == 'loss':
                        reward = -100.0  # Large negative reward for losing
                    else:  # draw
                        reward = 0.0
                    
                    # Mark the last turn as terminal
                    turn_data['terminal'] = True
                
                # Update turn data with calculated reward
                processed_turn = turn_data.copy()
                processed_turn['reward'] = reward
                processed_data.append(processed_turn)
                
                #print(f"Turn {turn_data['turn_num']}: action={turn_data['action']}, reward={reward:.2f}")
            
            # Calculate action statistics
            action_counts = {}
            for turn_data in rollout_data:
                action = turn_data.get('action', -1)
                action_counts[action] = action_counts.get(action, 0) + 1
            
            #print(f"Processed {len(processed_data)} turns with calculated rewards")
            return processed_data, action_counts
            
        except Exception as e:
            print(f"Error parsing rollout data: {e}")
            import traceback
            traceback.print_exc()
            return [], {}
    
    def save_model(self, game_num, checkpoint_name=None):
        """Save the RL model"""
        try:
            # Import the RL components
            sys.path.append(self.file_dir)
            from rl.agent import Agent
            
            # Load the existing trained agent
            agent = Agent()
            if os.path.exists(self.model_save_path):
                agent.load_model(self.model_save_path)
            
            # Determine save path
            if checkpoint_name:
                save_path = os.path.join(self.file_dir, 'checkpoints', checkpoint_name)
            else:
                save_path = self.model_save_path
            
            # Save the trained agent's Q-table
            agent.save_model(save_path)
            
            if checkpoint_name:
                print(f"Checkpoint saved: {checkpoint_name} at game {game_num}")
            else:
                print(f"Model saved at game {game_num}")
            
        except Exception as e:
            print(f"Error saving model: {e}")
    
    
    def _get_model_stats(self):
        """Get model statistics for logging"""
        try:
            # Import the RL components
            sys.path.append(self.file_dir)
            from rl.agent import Agent
            
            # Create a dummy agent to test saving
            agent = Agent()
            
            return agent.get_model_stats() if hasattr(agent, 'get_model_stats') else {
                'total_states': 0,
                'total_q_values': 0,
                'steps': 0,
                'epsilon': 0.1
            }
        except Exception as e:
            print(f"Error getting model stats: {e}")
            return {
                'total_states': 0,
                'total_q_values': 0,
                'steps': 0,
                'epsilon': 0.1
            }
    
    def _load_current_agent(self):
        """
        Load the current agent for training
        
        Returns:
            Agent: Loaded agent instance
        """
        try:
            # Import the RL components
            sys.path.append(self.file_dir)
            from rl.agent import Agent
            
            # Create agent with model path and CNN parameters
            agent = Agent(model_path=self.model_save_path, board_channels=4, scalar_dim=7, action_dim=8, hidden_dim=128)
            
            # Set to training mode
            agent.set_training_mode(True)
            
            return agent
            
        except Exception as e:
            print(f"Error loading agent: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _log_training_epoch(self, epoch, wins, losses, draws, win_rate, training_time, total_transitions, avg_enemy_hp, loss_info=None, action_counts=None):
        """
        Log training epoch statistics
        
        Args:
            epoch: Current epoch number
            wins: Number of wins in this epoch
            losses: Number of losses in this epoch
            draws: Number of draws in this epoch
            win_rate: Win rate for this epoch
            training_time: Time taken for this epoch
            total_transitions: Total number of transitions used for training
            avg_enemy_hp: Average enemy HP at game end
            loss_info: Dictionary containing loss information from agent.update()
        """
        log_entry = {
            "epoch": epoch,
            "total_time_seconds": training_time,
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "total_games": wins + losses + draws,
            "win_rate": win_rate,
            "training_time_seconds": training_time,
            "total_transitions": total_transitions,
            "avg_enemy_hp": avg_enemy_hp
        }
        
        # Add loss information if available
        if loss_info is not None:
            log_entry.update({
                "avg_policy_loss": loss_info['avg_policy_loss'],
                "avg_value_loss": loss_info['avg_value_loss'],
                "avg_entropy_loss": loss_info['avg_entropy_loss'],
                "avg_total_loss": loss_info['avg_total_loss']
            })
        
        # Add action statistics if available
        if action_counts is not None:
            total_actions = sum(action_counts.values())
            if total_actions > 0:
                action_stats = {}
                for action, count in action_counts.items():
                    action_stats[f"action_{action}_count"] = count
                    action_stats[f"action_{action}_percentage"] = (count / total_actions) * 100
                log_entry.update(action_stats)
        
        # Append to training log file
        with open(self.training_log_path, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
        
        print(f"Epoch {epoch} Win Rate: {win_rate:.2%}")
        print(f"Epoch {epoch} Results: {wins}W-{losses}L-{draws}D")
        print(f"Epoch {epoch} Training Time: {training_time:.2f}s")
        print(f"Epoch {epoch} Total Transitions: {total_transitions}")
        
        # Print loss information if available
        if loss_info is not None:
            print(f"Epoch {epoch} Avg Losses - Policy: {loss_info['avg_policy_loss']:.4f}, Value: {loss_info['avg_value_loss']:.4f}, Entropy: {loss_info['avg_entropy_loss']:.4f}, Total: {loss_info['avg_total_loss']:.4f}")
        
        # Print action statistics if available
        if action_counts is not None and action_counts:
            total_actions = sum(action_counts.values())
            print(f"Epoch {epoch} Action Usage:")
            for action in sorted(action_counts.keys()):
                count = action_counts[action]
                percentage = (count / total_actions) * 100
                print(f"  Action {action}: {count} times ({percentage:.1f}%)")
    
    def get_training_summary(self):
        """
        Get training summary from log file
        
        Returns:
            dict: Summary statistics
        """
        if not os.path.exists(self.training_log_path):
            return {"error": "No training log found"}
        
        epochs = []
        win_rates = []
        training_times = []
        avg_enemy_hps = []
        avg_policy_losses = []
        avg_value_losses = []
        avg_entropy_losses = []
        avg_total_losses = []
        
        with open(self.training_log_path, 'r') as f:
            for line in f:
                try:
                    log_entry = json.loads(line.strip())
                    epochs.append(log_entry['epoch'])
                    win_rates.append(log_entry['win_rate'])
                    training_times.append(log_entry['training_time_seconds'])
                    avg_enemy_hps.append(log_entry.get('avg_enemy_hp', 0.0))
                    
                    # Add loss information if available
                    avg_policy_losses.append(log_entry.get('avg_policy_loss', 0.0))
                    avg_value_losses.append(log_entry.get('avg_value_loss', 0.0))
                    avg_entropy_losses.append(log_entry.get('avg_entropy_loss', 0.0))
                    avg_total_losses.append(log_entry.get('avg_total_loss', 0.0))
                except (json.JSONDecodeError, KeyError):
                    continue
        
        if not epochs:
            return {"error": "No valid log entries found"}
        
        summary = {
            "total_epochs": len(epochs),
            "latest_epoch": max(epochs),
            "best_win_rate": max(win_rates),
            "latest_win_rate": win_rates[-1],
            "average_win_rate": sum(win_rates) / len(win_rates),
            "total_training_time": sum(training_times),
            "average_training_time": sum(training_times) / len(training_times),
            "latest_avg_enemy_hp": avg_enemy_hps[-1],
            "average_enemy_hp": sum(avg_enemy_hps) / len(avg_enemy_hps),
            "win_rate_trend": "improving" if len(win_rates) > 1 and win_rates[-1] > win_rates[0] else "declining" if len(win_rates) > 1 and win_rates[-1] < win_rates[0] else "stable",
            # Loss statistics
            "latest_avg_policy_loss": avg_policy_losses[-1] if avg_policy_losses else 0.0,
            "latest_avg_value_loss": avg_value_losses[-1] if avg_value_losses else 0.0,
            "latest_avg_entropy_loss": avg_entropy_losses[-1] if avg_entropy_losses else 0.0,
            "latest_avg_total_loss": avg_total_losses[-1] if avg_total_losses else 0.0,
            "average_policy_loss": sum(avg_policy_losses) / len(avg_policy_losses) if avg_policy_losses else 0.0,
            "average_value_loss": sum(avg_value_losses) / len(avg_value_losses) if avg_value_losses else 0.0,
            "average_entropy_loss": sum(avg_entropy_losses) / len(avg_entropy_losses) if avg_entropy_losses else 0.0,
            "average_total_loss": sum(avg_total_losses) / len(avg_total_losses) if avg_total_losses else 0.0,
            "best_policy_loss": min(avg_policy_losses) if avg_policy_losses else 0.0,
            "best_value_loss": min(avg_value_losses) if avg_value_losses else 0.0,
            "best_total_loss": min(avg_total_losses) if avg_total_losses else 0.0
        }
        
        return summary
    
    def print_training_summary(self):
        """Print training summary to console"""
        summary = self.get_training_summary()
        
        if "error" in summary:
            print(f"Training Summary Error: {summary['error']}")
            return
        
        print("\n" + "=" * 50)
        print("TRAINING SUMMARY")
        print("=" * 50)
        print(f"Total Epochs: {summary['total_epochs']}")
        print(f"Latest Epoch: {summary['latest_epoch']}")
        print(f"Best Win Rate: {summary['best_win_rate']:.2%}")
        print(f"Latest Win Rate: {summary['latest_win_rate']:.2%}")
        print(f"Average Win Rate: {summary['average_win_rate']:.2%}")
        print(f"Total Training Time: {summary['total_training_time']:.2f}s ({summary['total_training_time']/60:.2f}min)")
        print(f"Average Training Time per Epoch: {summary['average_training_time']:.2f}s")
        print(f"Latest Avg Enemy HP: {summary['latest_avg_enemy_hp']:.1f}")
        print(f"Average Enemy HP: {summary['average_enemy_hp']:.1f}")
        print(f"Win Rate Trend: {summary['win_rate_trend']}")
        
        # Print loss information if available
        if summary['latest_avg_total_loss'] > 0:
            print("\n" + "-" * 30)
            print("LOSS STATISTICS")
            print("-" * 30)
            print(f"Latest Policy Loss: {summary['latest_avg_policy_loss']:.4f}")
            print(f"Latest Value Loss: {summary['latest_avg_value_loss']:.4f}")
            print(f"Latest Entropy Loss: {summary['latest_avg_entropy_loss']:.4f}")
            print(f"Latest Total Loss: {summary['latest_avg_total_loss']:.4f}")
            print(f"Average Policy Loss: {summary['average_policy_loss']:.4f}")
            print(f"Average Value Loss: {summary['average_value_loss']:.4f}")
            print(f"Average Entropy Loss: {summary['average_entropy_loss']:.4f}")
            print(f"Average Total Loss: {summary['average_total_loss']:.4f}")
            print(f"Best Policy Loss: {summary['best_policy_loss']:.4f}")
            print(f"Best Value Loss: {summary['best_value_loss']:.4f}")
            print(f"Best Total Loss: {summary['best_total_loss']:.4f}")
        
        print("=" * 50)
    
    def train(self):
        """Main training loop"""
        print("=" * 50)
        print("Starting RL Training")
        print("=" * 50)
        
        for epoch in range(1, self.epochs + 1):
            print(f"Starting epoch {epoch}/{self.epochs}")
            
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
                continue
            
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
                checkpoint_name = f"checkpoint_epoch_{epoch}.pkl"
                self.save_model(epoch, checkpoint_name)
        
            end_time = time.time()
            training_time = end_time - start_time
            
            # Log training epoch statistics
            self._log_training_epoch(epoch, self.wins, self.losses, self.draws, win_rate, training_time, total_transitions, avg_enemy_hp, loss_info, total_action_counts)
        
        # Print final training summary
        self.print_training_summary()

if __name__ == "__main__":
    if len(sys.argv) < 2 or '--help' in sys.argv or '-h' in sys.argv:
        print("Usage: python train_rl.py <epochs> [opponent_path]")
        print("Example: python train_rl.py 100")
        print("Example: python train_rl.py 50 ../python-algo")
        print("Example: python train_rl.py 10 python-algo")
        sys.exit(0)
    
    try:
        epochs = int(sys.argv[1])
    except ValueError:
        print("Error: epochs must be an integer")
        sys.exit(1)
    
    opponent_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Get current directory (where our RL algo is)
    current_dir = os.path.dirname(os.path.realpath(__file__))
    
    # Create trainer
    trainer = RLTrainer(
        algo_path=current_dir,
        opponent_path=opponent_path,
        epochs=epochs,
        save_interval=100
    )
    
    # Start training
    trainer.train()
