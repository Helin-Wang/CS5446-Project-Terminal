"""
CNN-based observation space for the bot.

Board representation: [4, 28, 28]
- Channel 0: WALL occupancy    [-1, 0, +1]
- Channel 1: SUPPORT occupancy [-1, 0, +1]  
- Channel 2: TURRET occupancy  [-1, 0, +1]
- Channel 3: HP values         [-1, +1] (normalized HP)

Global vector: [7]
- My MP, SP, HP normalized
- Enemy MP, SP, HP normalized  
- Turn normalized
"""

import torch
import numpy as np

class State:
    def __init__(self, config=None):
        # Maximum values for normalization
        self.mp_max = 150
        self.sp_max = 150
        self.hp_max = 40
        self.turn_max = 100
        
        # Store unit type constants from config
        if config is not None:
            self.WALL = config["unitInformation"][0]["shorthand"]
            self.SUPPORT = config["unitInformation"][1]["shorthand"]
            self.TURRET = config["unitInformation"][2]["shorthand"]
        else:
            # Fallback to default values if config not provided
            self.WALL = "FF"
            self.SUPPORT = "EF"
            self.TURRET = "DF"

    def build_state(self, game_state):
        """
        Build CNN state representation
        Returns: (board_tensor, scalar_tensor)
        """
        # Create board representation
        board_tensor = self._create_board_tensor(game_state)
        
        # Create scalar features
        scalar_tensor = self._create_scalar_tensor(game_state)
        
        return board_tensor, scalar_tensor
    
    def _create_board_tensor(self, game_state):
        """
        Create 4-channel board tensor [4, 28, 28]
        Channels: WALL, SUPPORT, TURRET occupancy + HP values
        """
        board = torch.zeros(4, 28, 28)
        
        # Iterate through all valid board positions
        for x in range(28):
            for y in range(28):
                if game_state.game_map.in_arena_bounds([x, y]):
                    location = [x, y]
                    units = game_state.game_map[location]
                    
                    for unit in units:
                        # Only process stationary units (walls, supports, turrets)
                        if unit.stationary:
                            # Determine ownership and channel
                            if unit.player_index == 0:  # My units
                                ownership = 1.0
                            else:  # Enemy units
                                ownership = -1.0
                            
                            # Set occupancy in appropriate channel based on unit type
                            if unit.unit_type == self.WALL:
                                board[0, x, y] = ownership
                            elif unit.unit_type == self.SUPPORT:
                                board[1, x, y] = ownership
                            elif unit.unit_type == self.TURRET:
                                board[2, x, y] = ownership
                            
                            # Set HP value in channel 3
                            # Get max HP from unit's max_health property
                            max_hp = unit.max_health
                            
                            # Calculate HP ratio (ensure it's between 0 and 1)
                            hp_ratio = max(0.0, min(1.0, unit.health / max_hp))
                            board[3, x, y] = ownership * hp_ratio
                            
                            # Break after first stationary unit (should only be one)
                            break
        
        return board
    
    def _create_scalar_tensor(self, game_state):
        """
        Create scalar features tensor [7]
        """
        mp = game_state.get_resource(1)
        sp = game_state.get_resource(0)
        hp = game_state.my_health
        enemy_mp = game_state.get_resource(1, 1)
        enemy_sp = game_state.get_resource(0, 1)
        enemy_hp = game_state.enemy_health
        turn = game_state.turn_number
        
        return torch.tensor([
            self._normalize(mp, self.mp_max),
            self._normalize(sp, self.sp_max),
            self._normalize(hp, self.hp_max),
            self._normalize(enemy_mp, self.mp_max),
            self._normalize(enemy_sp, self.sp_max),
            self._normalize(enemy_hp, self.hp_max),
            self._normalize(turn, self.turn_max)
        ], dtype=torch.float32)
    
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