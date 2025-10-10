"""
Simplified logging system for RL training.

Creates a single comprehensive log file containing:
1. Overall win rate through the whole training process
2. Final Q-table
3. For each single game, the HP/MP/SP/selected action/reward of each turn
"""
import json
import os
import time
from datetime import datetime

class SimpleRLLogger:
    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Create single log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"rl_training_{timestamp}.json")
        
        # Initialize log structure
        self.log_data = {
            "training_session": {
                "start_time": datetime.now().isoformat(),
                "total_games": 0,
                "wins": 0,
                "losses": 0,
                "draws": 0,
                "overall_win_rate": 0.0
            },
            "final_qtable": {},
            "games": []
        }
        
        # Current game data
        self.current_game = None
        
        # Save initial log structure
        self._save_log_data()
        
    def start_game(self, game_num, opponent_info="default"):
        """Start logging a new game"""
        self.current_game = {
            "game_num": game_num,
            "opponent": opponent_info,
            "start_time": datetime.now().isoformat(),
            "turns": []
        }
        
    def log_turn(self, turn_num, state, action, reward, game_state_info):
        """Log a single turn"""
        if self.current_game is None:
            return
            
        turn_data = {
            "turn_num": turn_num,
            "state": state,
            "action": action,
            "reward": reward,
            "hp": game_state_info.get("my_hp", 0),
            "mp": game_state_info.get("my_mp", 0),
            "sp": game_state_info.get("my_sp", 0),
            "enemy_hp": game_state_info.get("enemy_hp", 0)
        }
        
        self.current_game["turns"].append(turn_data)
        
        # Save turn data immediately to file
        self._save_turn_to_file(turn_data)
        
    def end_game(self, result, final_state=None, terminal_reward=None):
        """End logging for the current game"""
        if self.current_game is None:
            return
            
        self.current_game["end_time"] = datetime.now().isoformat()
        self.current_game["result"] = result
        self.current_game["final_state"] = final_state
        self.current_game["terminal_reward"] = terminal_reward
        
        # Add to games list
        self.log_data["games"].append(self.current_game)
        
        # Update training session stats
        self.log_data["training_session"]["total_games"] += 1
        if result == "win":
            self.log_data["training_session"]["wins"] += 1
        elif result == "loss":
            self.log_data["training_session"]["losses"] += 1
        else:
            self.log_data["training_session"]["draws"] += 1
            
        # Update overall win rate
        total_games = self.log_data["training_session"]["total_games"]
        wins = self.log_data["training_session"]["wins"]
        self.log_data["training_session"]["overall_win_rate"] = wins / max(1, total_games)
        
        # Reset current game
        self.current_game = None
        
    def save_final_qtable(self, qtable):
        """Save the final Q-table"""
        self.log_data["final_qtable"] = qtable
        
    def _save_log_data(self):
        """Save log data to file"""
        with open(self.log_file, 'w') as f:
            json.dump(self.log_data, f, indent=2)
    
    def _save_turn_to_file(self, turn_data):
        """Save turn data to a separate file for immediate access"""
        turn_file = os.path.join(self.log_dir, "current_turns.jsonl")
        with open(turn_file, 'a') as f:
            f.write(json.dumps(turn_data) + '\n')
    
    def _load_turns_from_file(self):
        """Load turn data from file"""
        turn_file = os.path.join(self.log_dir, "current_turns.jsonl")
        turns = []
        if os.path.exists(turn_file):
            with open(turn_file, 'r') as f:
                for line in f:
                    if line.strip():
                        turns.append(json.loads(line.strip()))
        return turns
    
    def save_log(self):
        """Save the complete log to file"""
        self.log_data["training_session"]["end_time"] = datetime.now().isoformat()
        
        # Load turns from file and add to current game
        if self.current_game:
            self.current_game["turns"] = self._load_turns_from_file()
        
        # Also load turns for all games in the log_data
        for game in self.log_data["games"]:
            if not game.get("turns"):
                game["turns"] = self._load_turns_from_file()
        
        with open(self.log_file, 'w') as f:
            json.dump(self.log_data, f, indent=2)
            
        print(f"Training log saved to: {self.log_file}")

# Global logger instance
_logger = None

def get_simple_logger():
    """Get the global simple logger instance"""
    global _logger
    if _logger is None:
        _logger = SimpleRLLogger()
    return _logger

def init_simple_logger(log_dir="logs"):
    """Initialize the global simple logger"""
    global _logger
    _logger = SimpleRLLogger(log_dir)
    return _logger
