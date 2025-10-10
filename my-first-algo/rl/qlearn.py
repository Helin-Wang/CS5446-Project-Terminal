"""
This module implements the Q-learning algorithm.

ε-greedy tabular agent (dict-based Q)
"""
from rl.config import MACROS, EPS_START, EPS_END, EPS_DECAY_STEPS, ALPHA, GAMMA
from collections import defaultdict
import random
import pickle
import os

def _default_q_values():
    """Default Q-values for new states"""
    return [0.0] * len(MACROS)

class Agent:
    def __init__(self):
        # 1. Q-table: {state: {action: Q-value}}
        self.Q = defaultdict(_default_q_values)
        self.epsilon = 0.1
        self.steps = 0
        
    def act(self, state):
        self.steps += 1
        
        # Explore
        if random.random() < self.eps():
            return random.randrange(len(MACROS))
        
        # Exploit: choose the action with the highest Q-value
        q_values = self.Q[state]
        best_action = q_values.index(max(q_values))
        return best_action

    def eps(self):
        frac = min(1.0, self.steps / EPS_DECAY_STEPS)
        return EPS_START + (EPS_END - EPS_START) * frac
    
    def update(self, state, action, reward, next_state):
        current_q = self.Q[state][action]
        if next_state is None:
            best_next_q = 0.0
        else:
            best_next_q = max(self.Q[next_state])
        new_q = current_q + ALPHA * (reward + GAMMA * best_next_q - current_q)
        self.Q[state][action] = new_q
    
    def save_model(self, filepath):
        """Save the Q-table to a file"""
        try:
            # Convert defaultdict to regular dict for saving
            q_dict = dict(self.Q)
            model_data = {
                'Q': q_dict,
                'steps': self.steps,
                'epsilon': self.epsilon
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            print(f"Model saved to {filepath}")
            return True
        except Exception as e:
            print(f"Error saving model: {e}")
            return False
    
    def load_model(self, filepath):
        """Load the Q-table from a file"""
        try:
            if not os.path.exists(filepath):
                print(f"Model file {filepath} not found")
                return False
            
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            # Restore Q-table
            self.Q = defaultdict(_default_q_values)
            self.Q.update(model_data['Q'])
            
            # Restore training state
            self.steps = model_data.get('steps', 0)
            self.epsilon = model_data.get('epsilon', self.epsilon)
            
            print(f"Model loaded from {filepath}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def get_model_stats(self):
        """Get statistics about the current model"""
        total_states = len(self.Q)
        total_q_values = sum(len(q_values) for q_values in self.Q.values())
        
        return {
            'total_states': total_states,
            'total_q_values': total_q_values,
            'steps': self.steps,
            'epsilon': self.epsilon
        }
