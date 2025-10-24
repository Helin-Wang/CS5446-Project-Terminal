"""
This module implements the reward tracker for the bot.

Updated reward formula:
- HP component: r_hp = Δ(opp_hp) - Δ(my_hp) 
- Efficiency component: r_eff = (damage_dealt / mp_consumed) * action_bonus
- Combined: r_total = α * r_hp + β * r_eff (α=0.7, β=0.3)
- Episode end bonus: +1000 win, -1000 loss

"""

class RewardTracker:
    def __init__(self):
        self.my_last_hp = None
        self.opp_last_hp = None
        
        # Reward weights
        self.hp_weight = 0.7
        self.efficiency_weight = 0.3

    def reset(self, game_state=None):
        """
        Reset stored HP for a new episode.
        """
        if game_state is None:
            self.my_last_hp = None
            self.opp_last_hp = None
        else:
            self.my_last_hp = game_state.my_health
            self.opp_last_hp = game_state.enemy_health
        
    def compute_reward(self, game_state, action=None, turn_data=None):
        """
        Compute reward based on HP changes and combat efficiency.
        
        Args:
            game_state: Current game state
            action: Action taken this turn
            turn_data: Turn data containing combat information
            
        Returns:
            Combined reward value
        """
        # HP-based reward (original logic)
        hp_reward = self._compute_hp_reward(game_state)
        
        # Efficiency-based reward (new logic)
        efficiency_reward = self._compute_efficiency_reward(turn_data, action)
        
        # Combine rewards
        total_reward = self.hp_weight * hp_reward + self.efficiency_weight * efficiency_reward
        
        return total_reward
    
    def _compute_hp_reward(self, game_state):
        """
        Compute HP-based reward (original logic)
        """
        my_hp = game_state.my_health
        opp_hp = game_state.enemy_health
        
        reward = 0.0
        if self.my_last_hp is not None and self.opp_last_hp is not None:
            reward = (self.opp_last_hp - opp_hp) - (self.my_last_hp - my_hp)
        
        self.my_last_hp, self.opp_last_hp = my_hp, opp_hp
        return reward
    
    def _compute_efficiency_reward(self, turn_data, action):
        """
        Compute efficiency-based reward
        
        Args:
            turn_data: Dictionary containing combat data
            action: Action taken this turn
            
        Returns:
            Efficiency reward value
        """
        if turn_data is None:
            return 0.0
            
        damage_dealt = turn_data.get('damage_dealt', 0.0)
        
        # Return damage dealt as efficiency reward (no MP normalization)
        return damage_dealt
    
    def terminal_bonus(self, game_state):
        """
        Compute terminal reward based on final HP comparison.
        +1000 for win, -1000 for loss, 0 for tie
        """
        my_hp = game_state.my_health
        opp_hp = game_state.enemy_health
        
        if my_hp > opp_hp:
            return 100.0
        elif my_hp < opp_hp:
            return -100.0
        return 0.0
