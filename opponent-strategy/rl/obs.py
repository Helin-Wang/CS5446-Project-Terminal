"""
CNN-based observation space for the bot.

Board representation: [5, 28, 28]
- Channel 0: WALL occupancy    [-1, 0, +1]
- Channel 1: SUPPORT occupancy [-1, 0, +1]  
- Channel 2: TURRET occupancy  [-1, 0, +1]
- Channel 3: HP values         [-1, +1] (normalized HP)
- Channel 4: Valid mask        [0, 1] (1 for valid octagonal area, 0 for invalid)

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
        Create 5-channel board tensor [5, 28, 28]
        Channels: WALL, SUPPORT, TURRET occupancy + HP values + Valid mask
        """
        board = torch.zeros(5, 28, 28)
        
        # Create valid mask for octagonal board area
        valid_mask = self._create_valid_mask()
        board[4, :, :] = valid_mask
        
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
    
    def _create_valid_mask(self):
        """
        Create a valid mask for the octagonal board area.
        Returns a 28x28 tensor where 1 indicates valid positions and 0 indicates invalid positions.
        
        Octagonal vertices: [0,13], [0,14], [13,27], [14,27], [27,13], [27,14], [14,0], [13,0]
        """
        mask = torch.zeros(28, 28)
        
        # Define the octagonal vertices
        vertices = [
            [0, 13], [0, 14], [13, 27], [14, 27], 
            [27, 13], [27, 14], [14, 0], [13, 0]
        ]
        
        # Convert to numpy for easier manipulation
        vertices_np = np.array(vertices)
        
        # Create a grid of all positions
        y_coords, x_coords = np.mgrid[0:28, 0:28]
        points = np.stack([x_coords.flatten(), y_coords.flatten()], axis=1)
        
        # Check if each point is inside the octagon
        for i, (x, y) in enumerate(points):
            if self._point_in_octagon(x, y, vertices_np):
                mask[y, x] = 1.0
        
        # Explicitly mark the octagonal vertices as valid
        for x, y in vertices:
            mask[y, x] = 1.0
        
        return mask
    
    def _point_in_octagon(self, x, y, vertices):
        """
        Check if a point (x, y) is inside the octagon defined by vertices.
        Uses ray casting algorithm.
        """
        n = len(vertices)
        inside = False
        
        p1x, p1y = vertices[0]
        for i in range(1, n + 1):
            p2x, p2y = vertices[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
    
    def is_terminal_state(self, game_state):
        """
        Check if the current game state is terminal (game ended).
        Returns True if either player's HP reaches 0.
        """
        return game_state.my_health <= 0 or game_state.enemy_health <= 0 or game_state.turn_number >= 100