"""
This module defines the observation space for the bot.

It provides:
- Convert the GameState to the observation space with continuous normalized values
- mp_norm ∈ [0,1] normalized from your current MP (max 150)
- sp_norm ∈ [0,1] normalized from your current SP (max 150)  
- hp_norm ∈ [0,1] normalized from your current HP (max 40)
- enemy_mp_norm ∈ [0,1] normalized from enemy MP (max 150)
- enemy_sp_norm ∈ [0,1] normalized from enemy SP (max 150)
- enemy_hp_norm ∈ [0,1] normalized from enemy HP (max 40)
- Represent state as a tuple of normalized floats, e.g. (mp_norm, sp_norm, hp_norm, enemy_mp_norm, enemy_sp_norm, enemy_hp_norm).
"""

class State:
    def __init__(self):
        # Maximum values for normalization
        self.mp_max = 150
        self.sp_max = 150
        self.hp_max = 40


    def build_state(self, game_state):
        """
        Build the observation space from the GameState with normalized continuous values.
        """
        # Get raw values
        mp = game_state.get_resource(1)
        sp = game_state.get_resource(0)
        hp = game_state.my_health
        
        # Get enemy values
        enemy_mp = game_state.get_resource(1, 1)
        enemy_sp = game_state.get_resource(0, 1)
        enemy_hp = game_state.enemy_health
        
        # Normalize to [0, 1] range
        mp_norm = self._normalize(mp, self.mp_max)
        sp_norm = self._normalize(sp, self.sp_max)
        hp_norm = self._normalize(hp, self.hp_max)
        
        enemy_mp_norm = self._normalize(enemy_mp, self.mp_max)
        enemy_sp_norm = self._normalize(enemy_sp, self.sp_max)
        enemy_hp_norm = self._normalize(enemy_hp, self.hp_max)
        
        return (mp_norm, sp_norm, hp_norm, enemy_mp_norm, enemy_sp_norm, enemy_hp_norm)
    
    def _normalize(self, value, max_value):
        """
        Normalize a value to [0, 1] range.
        Clips the value to [0, max_value] and then normalizes.
        """
        # Clip value to valid range
        clipped_value = max(0, min(value, max_value))
        # Normalize to [0, 1]
        return clipped_value / max_value
    
    def is_terminal_state(self, game_state):
        """
        Check if the current game state is terminal (game ended).
        Returns True if either player's HP reaches 0.
        """
        return game_state.my_health <= 0 or game_state.enemy_health <= 0 or game_state.turn_number >= 100