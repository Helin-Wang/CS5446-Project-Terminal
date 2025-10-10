"""
Logging system for RL training
Logs macro actions, game states, rewards, and training metrics
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path

class RLLogger:
    def __init__(self, log_dir="logs"):
        """
        Initialize RL Logger
        
        Args:
            log_dir: Directory to save log files
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Create timestamp for this session
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Log files
        self.turn_log_file = self.log_dir / f"turns_{self.session_id}.jsonl"
        self.game_log_file = self.log_dir / f"games_{self.session_id}.jsonl"
        self.training_log_file = self.log_dir / f"training_{self.session_id}.jsonl"
        self.summary_file = self.log_dir / f"summary_{self.session_id}.json"
        
        # Session statistics
        self.session_stats = {
            'session_id': self.session_id,
            'start_time': datetime.now().isoformat(),
            'total_games': 0,
            'total_turns': 0,
            'total_wins': 0,
            'total_losses': 0,
            'total_draws': 0,
            'macro_action_counts': {},
            'reward_stats': {
                'total_reward': 0,
                'avg_reward_per_turn': 0,
                'max_reward': float('-inf'),
                'min_reward': float('inf')
            }
        }
        
        # Current game tracking
        self.current_game = None
        self.current_game_turns = []
        
        print(f"RL Logger initialized - Session ID: {self.session_id}")
        print(f"Log directory: {self.log_dir}")
    
    def start_game(self, game_num, opponent_info=None):
        """
        Start logging a new game
        
        Args:
            game_num: Game number in the training session
            opponent_info: Information about the opponent
        """
        self.current_game = {
            'game_num': game_num,
            'start_time': datetime.now().isoformat(),
            'opponent_info': opponent_info or "default",
            'turns': [],
            'final_result': None,
            'total_reward': 0,
            'macro_actions': []
        }
        self.current_game_turns = []
        
        print(f"Started logging game {game_num}")
    
    def log_turn(self, turn_num, state, action, reward, next_state, game_state_info=None):
        """
        Log a single turn
        
        Args:
            turn_num: Turn number in the current game
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            game_state_info: Additional game state information
        """
        turn_data = {
            'timestamp': datetime.now().isoformat(),
            'game_num': self.current_game['game_num'] if self.current_game else 0,
            'turn_num': turn_num,
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'game_state_info': game_state_info or {}
        }
        
        # Add to current game
        if self.current_game:
            self.current_game['turns'].append(turn_data)
            self.current_game['total_reward'] += reward
            self.current_game['macro_actions'].append(action)
        
        # Write to turn log file
        with open(self.turn_log_file, 'a') as f:
            f.write(json.dumps(turn_data) + '\n')
        
        # Update session statistics
        self.session_stats['total_turns'] += 1
        self.session_stats['reward_stats']['total_reward'] += reward
        self.session_stats['reward_stats']['max_reward'] = max(
            self.session_stats['reward_stats']['max_reward'], reward
        )
        self.session_stats['reward_stats']['min_reward'] = min(
            self.session_stats['reward_stats']['min_reward'], reward
        )
        
        # Update macro action counts
        action_str = str(action)
        self.session_stats['macro_action_counts'][action_str] = \
            self.session_stats['macro_action_counts'].get(action_str, 0) + 1
        
        # Update average reward
        if self.session_stats['total_turns'] > 0:
            self.session_stats['reward_stats']['avg_reward_per_turn'] = \
                self.session_stats['reward_stats']['total_reward'] / self.session_stats['total_turns']
    
    def end_game(self, result, final_state=None, terminal_reward=None):
        """
        End logging for the current game
        
        Args:
            result: Game result ('win', 'loss', 'draw')
            final_state: Final game state
            terminal_reward: Terminal reward if any
        """
        if not self.current_game:
            return
        
        self.current_game['end_time'] = datetime.now().isoformat()
        self.current_game['final_result'] = result
        self.current_game['final_state'] = final_state
        self.current_game['terminal_reward'] = terminal_reward
        
        # Add terminal reward to total if present
        if terminal_reward:
            self.current_game['total_reward'] += terminal_reward
        
        # Write to game log file
        with open(self.game_log_file, 'a') as f:
            f.write(json.dumps(self.current_game) + '\n')
        
        # Update session statistics
        self.session_stats['total_games'] += 1
        if result == 'win':
            self.session_stats['total_wins'] += 1
        elif result == 'loss':
            self.session_stats['total_losses'] += 1
        else:
            self.session_stats['total_draws'] += 1
        
        # Calculate win rate
        total_games = self.session_stats['total_games']
        if total_games > 0:
            self.session_stats['win_rate'] = self.session_stats['total_wins'] / total_games
        
        print(f"Game {self.current_game['game_num']} ended: {result}")
        print(f"  Total reward: {self.current_game['total_reward']:.2f}")
        print(f"  Turns: {len(self.current_game['turns'])}")
        print(f"  Macro actions: {self.current_game['macro_actions']}")
        
        self.current_game = None
    
    def log_training_epoch(self, epoch_num, model_stats, training_metrics=None):
        """
        Log training epoch information
        
        Args:
            epoch_num: Epoch number
            model_stats: Model statistics
            training_metrics: Additional training metrics
        """
        epoch_data = {
            'timestamp': datetime.now().isoformat(),
            'epoch_num': epoch_num,
            'model_stats': model_stats,
            'training_metrics': training_metrics or {},
            'session_stats': self.session_stats.copy()
        }
        
        # Write to training log file
        with open(self.training_log_file, 'a') as f:
            f.write(json.dumps(epoch_data) + '\n')
        
        print(f"Training epoch {epoch_num} logged")
        print(f"  Model states: {model_stats.get('total_states', 0)}")
        print(f"  Win rate: {self.session_stats.get('win_rate', 0):.2%}")
    
    def save_summary(self):
        """
        Save session summary
        """
        self.session_stats['end_time'] = datetime.now().isoformat()
        self.session_stats['duration'] = (
            datetime.now() - datetime.fromisoformat(self.session_stats['start_time'])
        ).total_seconds()
        
        # Calculate additional statistics
        if self.session_stats['total_games'] > 0:
            self.session_stats['avg_turns_per_game'] = \
                self.session_stats['total_turns'] / self.session_stats['total_games']
        
        # Save to file
        with open(self.summary_file, 'w') as f:
            json.dump(self.session_stats, f, indent=2)
        
        print(f"Session summary saved to: {self.summary_file}")
        return self.session_stats
    
    def get_session_stats(self):
        """Get current session statistics"""
        return self.session_stats.copy()
    
    def print_current_stats(self):
        """Print current session statistics"""
        stats = self.session_stats
        print("\n" + "="*50)
        print("Current Session Statistics")
        print("="*50)
        print(f"Games: {stats['total_games']}")
        print(f"Turns: {stats['total_turns']}")
        print(f"Wins: {stats['total_wins']}")
        print(f"Losses: {stats['total_losses']}")
        print(f"Draws: {stats['total_draws']}")
        print(f"Win Rate: {stats.get('win_rate', 0):.2%}")
        print(f"Total Reward: {stats['reward_stats']['total_reward']:.2f}")
        print(f"Avg Reward/Turn: {stats['reward_stats']['avg_reward_per_turn']:.2f}")
        print(f"Max Reward: {stats['reward_stats']['max_reward']:.2f}")
        print(f"Min Reward: {stats['reward_stats']['min_reward']:.2f}")
        print("\nMacro Action Distribution:")
        for action, count in stats['macro_action_counts'].items():
            percentage = (count / stats['total_turns'] * 100) if stats['total_turns'] > 0 else 0
            print(f"  Action {action}: {count} ({percentage:.1f}%)")
        print("="*50)

# Global logger instance
_global_logger = None

def get_logger():
    """Get the global logger instance"""
    global _global_logger
    if _global_logger is None:
        _global_logger = RLLogger()
    return _global_logger

def init_logger(log_dir="logs"):
    """Initialize the global logger"""
    global _global_logger
    _global_logger = RLLogger(log_dir)
    return _global_logger
