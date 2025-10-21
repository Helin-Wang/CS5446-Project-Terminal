"""
This module implements the reward tracker for the bot.


- Per turn: r_t = Δ(opp_hp) - Δ(my_hp)
- Episode end bonus: +500 win, -500 loss

"""

class RewardTracker:
    def __init__(self):
        self.my_last_hp = None
        self.opp_last_hp = None

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
        
    def compute_reward(self, game_state):
        """
        Compute reward based on HP changes from GameState.
        Reward = opponent HP loss - my HP loss
        """
        my_hp = game_state.my_health
        opp_hp = game_state.enemy_health
        
        reward = 0.0
        if self.my_last_hp is not None and self.opp_last_hp is not None:
            reward = (self.opp_last_hp - opp_hp) - (self.my_last_hp - my_hp)
        
        self.my_last_hp, self.opp_last_hp = my_hp, opp_hp
        return reward 
    
    def terminal_bonus(self, game_state):
        """
        Compute terminal reward based on final HP comparison.
        +500 for win, -500 for loss, 0 for tie
        """
        my_hp = game_state.my_health
        opp_hp = game_state.enemy_health
        
        if my_hp > opp_hp:
            return 500.0
        elif my_hp < opp_hp:
            return -500.0
        return 0.0
