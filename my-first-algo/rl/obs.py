"""
This module defines the observation space for the bot.

It provides:
- Convert the GameState to the observation space
- mp_bin ∈ {0,1,2,3} from your current MP (e.g., [0–5], [6–10], [11–15], [16+])
- sp_bin ∈ {0,1,2} from your current SP (e.g., [0–5], [6–12], [13+])
- hp_rel_bin ∈ {0,1,2} from sign of (my_hp − opp_hp): behind / even / ahead
- lane_pressure ∈ {L, C, R} estimated coarsely: which of (left, mid, right) has fewest enemy turrets in first 4 rows from their side (count via GameState.contains_stationary_unit + GameUnit.unit_type)
- Represent state as a tuple of small ints, e.g. (bits_bin, cores_bin, hp_rel_bin, lane_idx).
"""

class State:
    def __init__(self):
        self.mp_bins = [10, 20, 30, 40]
        self.sp_bins = [10, 20, 30, 40]
        # HP bins: 0 (terminal), 1-9, 10-19, 20-29, 30-39, 40+
        # HP=0 gets its own bin for terminal state detection
        self.hp_bins = [1, 10, 20, 30, 40]
        # self.hp_rel_bins = [-1, 0, 1]
        # self.lane_pressure_bins = ["L", "C", "R"]
        # self.lane_idx_bins = [0, 1, 2]


    def build_state(self, game_state):
        """
        Build the observation space from the GameState.
        """
        # 1. mp&sp
        mp = game_state.get_resource(1)
        sp = game_state.get_resource(0)
        hp = game_state.my_health
        mp_bin = self._map_to_bin(mp, self.mp_bins)
        sp_bin = self._map_to_bin(sp, self.sp_bins)
        hp_bin = self._map_to_bin(hp, self.hp_bins)
        
        # mp&sp of enemy
        enemy_mp = game_state.get_resource(1, 1)
        enemy_sp = game_state.get_resource(0, 1)
        enemy_hp = game_state.enemy_health
        enemy_mp_bin = self._map_to_bin(enemy_mp, self.mp_bins)
        enemy_sp_bin = self._map_to_bin(enemy_sp, self.sp_bins)
        enemy_hp_bin = self._map_to_bin(enemy_hp, self.hp_bins)
        
        # 2. hp: TODO
        
        # 3. lane pressure: TODO    
        
        # 4. lane idx: TODO
        
        return (mp_bin, sp_bin, hp_bin, enemy_mp_bin, enemy_sp_bin, enemy_hp_bin)
    
    def _map_to_bin(self, value, bins):
        """
        Map a value to a bin.
        """
        for index in range(len(bins)):
            if value < bins[index]:
                return index
        return len(bins)
    
    def is_terminal_state(self, game_state):
        """
        Check if the current game state is terminal (game ended).
        Returns True if either player's HP reaches 0.
        """
        return game_state.my_health <= 0 or game_state.enemy_health <= 0 or game_state.turn_number >= 100