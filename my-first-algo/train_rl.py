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
from rl.simple_logger import init_simple_logger

class RLTrainer:
    def __init__(self, algo_path, opponent_path=None, num_games=100, save_interval=10):
        """
        Initialize RL Trainer
        
        Args:
            algo_path: Path to our RL algorithm
            opponent_path: Path to opponent algorithm (default: python-algo)
            num_games: Number of games to train
            save_interval: Save model every N games
        """
        self.algo_path = algo_path
        self.num_games = num_games
        self.save_interval = save_interval
        
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
        
        # Initialize simple logger
        self.logger = init_simple_logger(log_dir=os.path.join(self.file_dir, "logs"))
        
    def _get_default_opponent(self):
        """Get default opponent path"""
        if self.is_windows:
            return os.path.join(self.parent_dir, "python-algo", "run.ps1")
        else:
            return os.path.join(self.parent_dir, "python-algo", "run.sh")
    
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
    
    def run_single_game(self, game_num):
        """
        Run a single game and return the result
        
        Returns:
            dict: Game result with win/loss information
        """
        algo1, algo2 = self._prepare_algo_paths()
        
        print(f"Starting game {game_num}/{self.num_games}")
        print(f"RL Agent: {algo1}")
        print(f"Opponent: {algo2}")
        
        # Start logging this game
        self.logger.start_game(game_num, opponent_info=algo2)
        
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
            
            # End logging for this game
            self.logger.end_game(
                result=game_result['result'],
                final_state=None,
                terminal_reward=None
            )
            
            return game_result
            
        except subprocess.TimeoutExpired:
            print(f"Game {game_num} timed out")
            self.logger.end_game('timeout')
            return {'game_num': game_num, 'result': 'timeout', 'winner': 'none'}
        except Exception as e:
            print(f"Error in game {game_num}: {e}")
            self.logger.end_game('error')
            return {'game_num': game_num, 'result': 'error', 'winner': 'none'}
    
    def _parse_game_result(self, stdout, stderr):
        """
        Parse game result from output
        This is a simplified parser - you may need to adjust based on actual game output
        """
        # Look for win/loss indicators in the output
        output = stdout + stderr
        
        if "Player 1 wins" in output or "Player 0 wins" in output:
            if "Player 1 wins" in output:
                return {'result': 'win', 'winner': 'player1', 'score': 1}
            else:
                return {'result': 'loss', 'winner': 'player0', 'score': -1}
        elif "Draw" in output or "Tie" in output:
            return {'result': 'draw', 'winner': 'none', 'score': 0}
        else:
            # Default to loss if we can't parse
            return {'result': 'loss', 'winner': 'player0', 'score': -1}
    
    def save_model(self, game_num, checkpoint_name=None):
        """Save the RL model"""
        try:
            # Import the RL components
            sys.path.append(self.file_dir)
            from rl.qlearn import Agent
            
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
            from rl.qlearn import Agent
            
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
    
    def load_model(self):
        """Load the RL model if it exists"""
        try:
            if os.path.exists(self.model_save_path):
                with open(self.model_save_path, 'rb') as f:
                    model_data = pickle.load(f)
                
                print(f"Loaded model from game {model_data.get('game_num', 0)}")
                print(f"Previous win rate: {model_data.get('win_rate', 0):.2%}")
                return model_data
        except Exception as e:
            print(f"Error loading model: {e}")
        
        return None
    
    def train(self):
        """Main training loop"""
        print("=" * 50)
        print("Starting RL Training")
        print("=" * 50)
        
        # Load existing model if available
        model_data = self.load_model()
        if model_data:
            self.wins = model_data.get('wins', 0)
            self.losses = model_data.get('losses', 0)
            self.draws = model_data.get('draws', 0)
        
        start_time = time.time()
        
        for game_num in range(1, self.num_games + 1):
            # Run single game
            result = self.run_single_game(game_num)
            self.game_results.append(result)
            
            # Update statistics
            if result['result'] == 'win':
                self.wins += 1
            elif result['result'] == 'loss':
                self.losses += 1
            else:
                self.draws += 1
            
            # Print progress
            total_games = self.wins + self.losses + self.draws
            win_rate = self.wins / max(1, total_games)
            
            print(f"Game {game_num}: {result['result']} | "
                  f"Win Rate: {win_rate:.2%} | "
                  f"Wins: {self.wins} | Losses: {self.losses} | Draws: {self.draws}")
            
            # Save checkpoint every 10 games
            if game_num % 10 == 0:
                checkpoint_name = f"checkpoint_game_{game_num}.pkl"
                self.save_model(game_num, checkpoint_name)
                print(f"Checkpoint saved: {checkpoint_name}")
            
            # Save model periodically (every save_interval games)
            if game_num % self.save_interval == 0:
                self.save_model(game_num)
            
            # Print current stats periodically
            if game_num % 5 == 0:
                print(f"Progress: {game_num}/{self.num_games} games completed")
            
            # Small delay between games
            time.sleep(1)
        
        # Final save
        self.save_model(self.num_games)
        
        # Training summary
        total_time = time.time() - start_time
        final_win_rate = self.wins / max(1, self.wins + self.losses + self.draws)
        
        # Save final Q-table to logger
        try:
            sys.path.append(self.file_dir)
            from rl.qlearn import Agent
            agent = Agent()
            if os.path.exists(self.model_save_path):
                agent.load_model(self.model_save_path)
                # Convert defaultdict to regular dict for JSON serialization
                # Convert tuple keys to strings for JSON compatibility
                q_dict = {}
                for state_key, q_values in agent.Q.items():
                    # Convert tuple key to string representation
                    state_str = str(state_key) if isinstance(state_key, tuple) else str(state_key)
                    q_dict[state_str] = q_values
                self.logger.save_final_qtable(q_dict)
        except Exception as e:
            print(f"Error saving final Q-table: {e}")
        
        # Save complete log
        self.logger.save_log()
        
        print("=" * 50)
        print("Training Complete!")
        print("=" * 50)
        print(f"Total games: {self.num_games}")
        print(f"Wins: {self.wins}")
        print(f"Losses: {self.losses}")
        print(f"Draws: {self.draws}")
        print(f"Final win rate: {final_win_rate:.2%}")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Average time per game: {total_time/self.num_games:.2f} seconds")
        
        # Save final results
        results_path = os.path.join(self.file_dir, "training_results.json")
        with open(results_path, 'w') as f:
            json.dump({
                'total_games': self.num_games,
                'wins': self.wins,
                'losses': self.losses,
                'draws': self.draws,
                'win_rate': final_win_rate,
                'total_time': total_time,
                'game_results': self.game_results,
                'log_file': self.logger.log_file
            }, f, indent=2)
        
        print(f"Results saved to: {results_path}")
        print(f"Training log saved to: {self.logger.log_file}")

def main():
    """Main function"""
    if len(sys.argv) < 2 or '--help' in sys.argv or '-h' in sys.argv:
        print("Usage: python train_rl.py <num_games> [opponent_path]")
        print("Example: python train_rl.py 100")
        print("Example: python train_rl.py 50 ../python-algo")
        print("Example: python train_rl.py 10 python-algo")
        sys.exit(0)
    
    try:
        num_games = int(sys.argv[1])
    except ValueError:
        print("Error: num_games must be an integer")
        sys.exit(1)
    
    opponent_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Get current directory (where our RL algo is)
    current_dir = os.path.dirname(os.path.realpath(__file__))
    
    # Create trainer
    trainer = RLTrainer(
        algo_path=current_dir,
        opponent_path=opponent_path,
        num_games=num_games,
        save_interval=10
    )
    
    # Start training
    trainer.train()

if __name__ == "__main__":
    main()
