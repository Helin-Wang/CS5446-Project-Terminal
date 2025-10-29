"""
Expert Collector Strategy

This is a wrapper around python-algo's expert strategy that collects
(state, action) demonstrations for behavior cloning.

Usage: This strategy will be used in place of one of the python-algo strategies
during expert demonstrations collection.
"""

import sys
import os
import random
import json
from sys import maxsize
import pickle

# Add paths for imports
BASE_DIR = os.path.dirname(__file__)
PARENT_DIR = os.path.join(BASE_DIR, '..')

# Add paths for gamelib
# for gamelib_dir in ['python-algo', 'my-first-algo-ppo', 'opponent-strategy']:
#     sys.path.insert(0, os.path.join(PARENT_DIR, gamelib_dir))
# sys.path.insert(0, PARENT_DIR)

import gamelib

# Add my-first-algo-ppo for RL modules
State = None
try:
    ppo_path = os.path.join(PARENT_DIR, 'my-first-algo-ppo')
    if ppo_path not in sys.path:
        sys.path.insert(0, ppo_path)
    from rl.obs import State
except ImportError as e:
    gamelib.debug_write(f"Warning: Could not import State from rl.obs: {e}")
    State = None


class ExpertCollectorStrategy(gamelib.AlgoCore):
    """
    Expert strategy that mimics python-algo's behavior and records demonstrations.
    """
    
    def __init__(self):
        super().__init__()
        seed = random.randrange(maxsize)
        random.seed(seed)
        
        # Collector state
        self.current_episode = []
        self.episode_num = 0
        self.output_dir = os.environ.get('BC_OUTPUT_DIR', os.path.join(os.path.dirname(__file__), 'bc_data'))
        os.makedirs(self.output_dir, exist_ok=True)
    
    def on_game_start(self, config):
        """ Initialize strategy with config """
        self.config = config
        
        # Set episode number from environment variable
        self.episode_num = int(os.environ.get('GAME_NUM', 0))
        gamelib.debug_write('Expert Collector: Configuring expert strategy...')
        gamelib.debug_write(f'Episode number: {self.episode_num}, Output dir: {self.output_dir}')
        
        global WALL, SUPPORT, TURRET, SCOUT, DEMOLISHER, INTERCEPTOR, MP, SP
        WALL = config["unitInformation"][0]["shorthand"]
        SUPPORT = config["unitInformation"][1]["shorthand"]
        TURRET = config["unitInformation"][2]["shorthand"]
        SCOUT = config["unitInformation"][3]["shorthand"]
        DEMOLISHER = config["unitInformation"][4]["shorthand"]
        INTERCEPTOR = config["unitInformation"][5]["shorthand"]
        MP = 1
        SP = 0
        
        self.scored_on_locations = []
        
        # Initialize state builder
        self.state_builder = State(config)
        
        # Reset episode data
        self.current_episode = []

    def on_turn(self, turn_state):
        """ Execute expert strategy and record (state, action) """
        game_state = gamelib.GameState(self.config, turn_state)
        gamelib.debug_write(f'Expert Collector: Turn {game_state.turn_number}')
        game_state.suppress_warnings(True)
    
        board_tensor, scalar_tensor = self.state_builder.build_state(game_state)
        expert_action = self._get_expert_action(game_state)
            
        transition = {
            'state': {
                'board': board_tensor.numpy(),
                'scalar': scalar_tensor.numpy()
            },
            'action': expert_action,
            'turn': game_state.turn_number,
            'hp': game_state.my_health,
            'enemy_hp': game_state.enemy_health,
            'mp': game_state.get_resource(MP),
            'sp': game_state.get_resource(SP)
        }
        
        # Add to current episode
        self.current_episode.append(transition)
        
        # Save transition immediately to prevent data loss
        self._save_transition(transition, game_state.turn_number)
        
        self._execute_expert_action(game_state, expert_action)
        gamelib.debug_write(f'Expert Collector: Action={expert_action}, Turn={game_state.turn_number}')
        game_state.submit_turn()
    
    def on_action_frame(self, turn_string):
        """ Track breaches for reactive defense """
        try:
            state = json.loads(turn_string)
            events = state.get("events", {})
            breaches = events.get("breach", [])
            
            for breach in breaches:
                location = breach[0]
                unit_owner_self = True if breach[4] == 1 else False
                # When parsing the frame data directly, 
                # 1 is integer for yourself, 2 is opponent (StarterKit code uses 0, 1 as player_index instead)
                if not unit_owner_self:
                    gamelib.debug_write("Got scored on at: {}".format(location))
                    self.scored_on_locations.append(location)
                    gamelib.debug_write("All locations: {}".format(self.scored_on_locations))
        except Exception as e:
            gamelib.debug_write(f"Error in on_action_frame: {e}")
    
    def _save_transition(self, transition, turn_number):
        """ Save a single transition to a file immediately """
        try:
            # Create a unique filename for this transition
            filename = f"episode_{self.episode_num}_turn_{turn_number}.pkl"
            filepath = os.path.join(self.output_dir, filename)
            
            # Save the transition
            with open(filepath, 'wb') as f:
                pickle.dump(transition, f)
            
            gamelib.debug_write(f"Saved transition: episode_{self.episode_num}, turn {turn_number}")
        except Exception as e:
            gamelib.debug_write(f"Error saving transition: {e}")
    
    def save_episode(self):
        """ Save current episode data to file """
        if len(self.current_episode) == 0:
            gamelib.debug_write("No data to save for this episode (episode_num={})".format(self.episode_num))
            return
        
        episode_file = os.path.join(self.output_dir, f"episode_{self.episode_num}.pkl")
        gamelib.debug_write(f"Saving episode {self.episode_num} to {episode_file}")
        
        # Prepare episode data in format suitable for training
        episode_data = {
            'states': [transition['state'] for transition in self.current_episode],
            'actions': [transition['action'] for transition in self.current_episode],
            'turns': [transition['turn'] for transition in self.current_episode],
            'hp': [transition['hp'] for transition in self.current_episode],
            'enemy_hp': [transition['enemy_hp'] for transition in self.current_episode],
        }
        
        with open(episode_file, 'wb') as f:
            pickle.dump(episode_data, f)
        
        gamelib.debug_write(f"Saved episode {self.episode_num} with {len(self.current_episode)} transitions to {episode_file}")
        
        # Reset for next episode
        self.current_episode = []
        self.scored_on_locations = []
    
    def on_game_end(self):
        """ Called when the game ends - save the episode data """
        gamelib.debug_write(f"Game ended - saving episode {self.episode_num}")
        self.save_episode()
    
    def _get_expert_action(self, game_state):
        """
        Determine expert action based on python-algo's starter_strategy logic.
        
        Returns action index (0-7):
        0: SAVE_BITS
        1: SCOUT_FLOOD_LEFT  
        2: SCOUT_FLOOD_RIGHT
        3: DEMOLISHER_LINE_RIGHT
        4: DEMOLISHER_LINE_LEFT
        5: INTERCEPTOR_NET_GLOBAL
        6: INTERCEPTOR_NET_LEFT
        7: INTERCEPTOR_NET_RIGHT
        """
        # Strategy decision logic from python-algo starter_strategy
        if game_state.turn_number < 5:
            # Turn < 5: stall with interceptors (global deployment)
            return 5  # INTERCEPTOR_NET_GLOBAL
        else:
            # Check if enemy has many units in front rows
            front_units = self._detect_enemy_unit(game_state, valid_y=[14, 15])
            
            if front_units > 10:
                # Many units in front -> use demolisher line strategy
                return 3  # DEMOLISHER_LINE_RIGHT
            else:
                # Scout flood strategy (alternating turns)
                if game_state.turn_number % 2 == 1:
                    # Check which side is safer for scout deployment
                    left_damage = self._estimate_path_damage(game_state, [13, 0])
                    right_damage = self._estimate_path_damage(game_state, [14, 0])
                    
                    if left_damage < right_damage:
                        return 1  # SCOUT_FLOOD_LEFT
                    else:
                        return 2  # SCOUT_FLOOD_RIGHT
                else:
                    # Even turns: save resources
                    return 0  # SAVE_BITS
    
    def _estimate_path_damage(self, game_state, location):
        """ Estimate damage along path to enemy base """
        path = game_state.find_path_to_edge(location)
        damage = 0
        for path_location in path:
            damage += len(game_state.get_attackers(path_location, 0))
        return damage
    
    def _detect_enemy_unit(self, game_state, unit_type=None, valid_x=None, valid_y=None):
        """ Count enemy units in specified area """
        total_units = 0
        for location in game_state.game_map:
            if game_state.contains_stationary_unit(location):
                for unit in game_state.game_map[location]:
                    if unit.player_index == 1:
                        if unit_type is None or unit.unit_type == unit_type:
                            if valid_x is None or location[0] in valid_x:
                                if valid_y is None or location[1] in valid_y:
                                    total_units += 1
        return total_units
    
    def _execute_expert_action(self, game_state, action):
        """
        Execute the expert action based on macro actions.
        This mirrors the execute_macro logic from macro.py
        
        Args:
            game_state: Current game state
            action: Action index (0-7) matching macro.py definitions
        """
        # Always build basic defenses first
        self._build_defences(game_state)
        self._build_reactive_defense(game_state)
        
        # Execute the specific macro action based on macro.py mapping
        # Action 0: SAVE_BITS - do nothing offensive
        if action == 0:
            pass  # Already handled by building supports at the end
        
        # Action 1: SCOUT_FLOOD_LEFT
        elif action == 1:
            self._scout_flood_left(game_state)
        
        # Action 2: SCOUT_FLOOD_RIGHT
        elif action == 2:
            self._scout_flood_right(game_state)
        
        # Action 3: DEMOLISHER_LINE_RIGHT
        elif action == 3:
            self._demolisher_line_right(game_state)
        
        # Action 4: DEMOLISHER_LINE_LEFT
        elif action == 4:
            self._demolisher_line_left(game_state)
        
        # Action 5: INTERCEPTOR_NET_GLOBAL
        elif action == 5:
            self._interceptor_net_global(game_state)
        
        # Action 6: INTERCEPTOR_NET_LEFT
        elif action == 6:
            self._interceptor_net_left(game_state)
        
        # Action 7: INTERCEPTOR_NET_RIGHT
        elif action == 7:
            self._interceptor_net_right(game_state)
        
        # Always try to build supports if possible
        support_locations = [[13, 2], [14, 2], [13, 3], [14, 3]]
        game_state.attempt_spawn(SUPPORT, support_locations)
    
    def _build_defences(self, game_state):
        """ Build basic defenses """
        turret_locations = [[0, 13], [27, 13], [8, 11], [19, 11], [13, 11], [14, 11]]
        game_state.attempt_spawn(TURRET, turret_locations)
        
        wall_locations = [[8, 12], [19, 12]]
        game_state.attempt_spawn(WALL, wall_locations)
        game_state.attempt_upgrade(wall_locations)
    
    def _build_reactive_defense(self, game_state):
        """ Build reactive defenses """
        for location in self.scored_on_locations:
            build_location = [location[0], location[1]+1]
            game_state.attempt_spawn(TURRET, build_location)
    
    def _stall_with_interceptors(self, game_state):
        """ Deploy interceptors globally """
        friendly_edges = (
            game_state.game_map.get_edge_locations(game_state.game_map.BOTTOM_LEFT) + 
            game_state.game_map.get_edge_locations(game_state.game_map.BOTTOM_RIGHT)
        )
        deploy_locations = self._filter_blocked_locations(friendly_edges, game_state)
        
        while game_state.get_resource(MP) >= game_state.type_cost(INTERCEPTOR)[MP] and len(deploy_locations) > 0:
            deploy_index = random.randint(0, len(deploy_locations) - 1)
            deploy_location = deploy_locations[deploy_index]
            game_state.attempt_spawn(INTERCEPTOR, deploy_location)
    
    def _scout_flood_left(self, game_state):
        """ Scout flood from left side """
        scout_spawn_location_options = [[11, 2], [12, 1], [13, 0]]
        best_location = self._least_damage_spawn_location(game_state, scout_spawn_location_options)
        game_state.attempt_spawn(SCOUT, best_location, 1000)
    
    def _scout_flood_right(self, game_state):
        """ Scout flood from right side """
        scout_spawn_location_options = [[16, 2], [15, 1], [14, 0]]
        best_location = self._least_damage_spawn_location(game_state, scout_spawn_location_options)
        game_state.attempt_spawn(SCOUT, best_location, 1000)
    
    def _demolisher_line_right(self, game_state):
        """ Build demolisher line on right side """
        stationary_units = [WALL, TURRET, SUPPORT]
        cheapest_unit = WALL
        for unit in stationary_units:
            unit_class = gamelib.GameUnit(unit, game_state.config)
            if unit_class.cost[game_state.MP] < gamelib.GameUnit(cheapest_unit, game_state.config).cost[game_state.MP]:
                cheapest_unit = unit
        
        for x in range(27, 5, -1):
            game_state.attempt_spawn(cheapest_unit, [x, 11])
        
        game_state.attempt_spawn(DEMOLISHER, [24, 10], 1000)
    
    def _demolisher_line_left(self, game_state):
        """ Build demolisher line on left side """
        stationary_units = [WALL, TURRET, SUPPORT]
        cheapest_unit = WALL
        for unit in stationary_units:
            unit_class = gamelib.GameUnit(unit, game_state.config)
            if unit_class.cost[game_state.MP] < gamelib.GameUnit(cheapest_unit, game_state.config).cost[game_state.MP]:
                cheapest_unit = unit
        
        for x in range(0, 22):
            game_state.attempt_spawn(cheapest_unit, [x, 11])
        
        game_state.attempt_spawn(DEMOLISHER, [3, 10], 1000)
    
    def _interceptor_net_global(self, game_state):
        """ Deploy interceptors globally """
        friendly_edges = (
            game_state.game_map.get_edge_locations(game_state.game_map.BOTTOM_LEFT) + 
            game_state.game_map.get_edge_locations(game_state.game_map.BOTTOM_RIGHT)
        )
        deploy_locations = self._filter_blocked_locations(friendly_edges, game_state)
        
        while game_state.get_resource(MP) >= game_state.type_cost(INTERCEPTOR)[MP] and len(deploy_locations) > 0:
            deploy_index = random.randint(0, len(deploy_locations) - 1)
            deploy_location = deploy_locations[deploy_index]
            game_state.attempt_spawn(INTERCEPTOR, deploy_location)
    
    def _interceptor_net_left(self, game_state):
        """ Deploy interceptors on left side """
        friendly_edges = (
            game_state.game_map.get_edge_locations(game_state.game_map.BOTTOM_LEFT) + 
            game_state.game_map.get_edge_locations(game_state.game_map.BOTTOM_RIGHT)
        )
        deploy_locations = self._filter_blocked_locations(friendly_edges, game_state)
        
        left_locations = [loc for loc in deploy_locations if loc[0] < 14]
        
        while game_state.get_resource(MP) >= game_state.type_cost(INTERCEPTOR)[MP] and len(left_locations) > 0:
            deploy_index = random.randint(0, len(left_locations) - 1)
            deploy_location = left_locations[deploy_index]
            game_state.attempt_spawn(INTERCEPTOR, deploy_location)
    
    def _interceptor_net_right(self, game_state):
        """ Deploy interceptors on right side """
        friendly_edges = (
            game_state.game_map.get_edge_locations(game_state.game_map.BOTTOM_LEFT) + 
            game_state.game_map.get_edge_locations(game_state.game_map.BOTTOM_RIGHT)
        )
        deploy_locations = self._filter_blocked_locations(friendly_edges, game_state)
        
        right_locations = [loc for loc in deploy_locations if loc[0] > 13]
        
        while game_state.get_resource(MP) >= game_state.type_cost(INTERCEPTOR)[MP] and len(right_locations) > 0:
            deploy_index = random.randint(0, len(right_locations) - 1)
            deploy_location = right_locations[deploy_index]
            game_state.attempt_spawn(INTERCEPTOR, deploy_location)
    
    def _least_damage_spawn_location(self, game_state, location_options):
        """ Find safest spawn location """
        damages = []
        for location in location_options:
            path = game_state.find_path_to_edge(location)
            damage = 0
            for path_location in path:
                damage += len(game_state.get_attackers(path_location, 0)) * gamelib.GameUnit(TURRET, game_state.config).damage_i
            damages.append(damage)
        return location_options[damages.index(min(damages))]
    
    def _filter_blocked_locations(self, locations, game_state):
        """ Filter out blocked locations """
        filtered = []
        for location in locations:
            if not game_state.contains_stationary_unit(location):
                filtered.append(location)
        return filtered


if __name__ == "__main__":
    algo = ExpertCollectorStrategy()
    algo.start()
