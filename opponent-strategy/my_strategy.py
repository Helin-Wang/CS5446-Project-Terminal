import gamelib
import random
import math
import warnings
import os
from sys import maxsize
import json
from rl.obs import State
from rl.reward_tracker import RewardTracker
from rl.agent import Agent
from rl.macro import MacroActions
"""
Self-Play Opponent Strategy

This is identical to my_strategy.py but loads opponent_model.pkl instead of rl_model.pkl.
The opponent model is updated periodically during self-play training.
"""

class AlgoStrategy(gamelib.AlgoCore):
    def __init__(self):
        super().__init__()
        seed = random.randrange(maxsize)
        random.seed(seed)
        gamelib.debug_write('Self-Play Opponent Random seed: {}'.format(seed))

    def on_game_start(self, config):
        """ 
        Read in config and perform any initial setup here 
        """
        gamelib.debug_write('Configuring self-play opponent strategy...')
        self.config = config
        global WALL, SUPPORT, TURRET, SCOUT, DEMOLISHER, INTERCEPTOR, MP, SP
        WALL = config["unitInformation"][0]["shorthand"]
        SUPPORT = config["unitInformation"][1]["shorthand"]
        TURRET = config["unitInformation"][2]["shorthand"]
        SCOUT = config["unitInformation"][3]["shorthand"]
        DEMOLISHER = config["unitInformation"][4]["shorthand"]
        INTERCEPTOR = config["unitInformation"][5]["shorthand"]
        MP = 1
        SP = 0
        
        # Initialize RL components for self-play opponent
        try:
            # Load the opponent model (updated periodically during training)
            model_path = os.path.join(os.path.dirname(__file__), "opponent_model.pkl")
            self.agent = Agent(model_path=model_path, board_channels=5, scalar_dim=7, action_dim=8, hidden_dim=128)
            self.state = State(config)
            self.macro_actions = MacroActions(config)
            self.reward_tracker = RewardTracker()
            
            # Set to evaluation mode (no training for opponent)
            self.agent.set_training_mode(False)
            
            gamelib.debug_write('Self-play opponent initialized with RL model')
        except Exception as e:
            gamelib.debug_write(f'Error initializing opponent RL model: {e}')
            # Fallback to simple strategy
            self.agent = None
            self.scored_on_locations = []
            gamelib.debug_write('Self-play opponent initialized (fallback mode)')

        # Track locations where the enemy scored on us
        self.scored_on_locations = []
        
        # Initialize rollout data caching
        self.current_turn_data = None
        self.turn_data_logged = False
        
        # Initialize combat data tracking
        self.current_turn_combat_data = {
            'damage_dealt': 0.0,
            'damage_received': 0.0,
            'events': {}
        }
        
        # Game state tracking
        self.game_ended = False
        self.reward_tracker.reset()
        
        gamelib.debug_write('Self-play opponent strategy configured')

    def on_turn(self, turn_state):
        """
        This function is called every turn with the game state wrapper as
        an argument. The wrapper stores the state of the arena and has methods
        for querying its state, allocating your current resources as planned
        unit deployments, and transmitting your intended deployments to the
        game engine.
        """
        game_state = gamelib.GameState(self.config, turn_state)
        gamelib.debug_write('Performing turn {} of self-play opponent strategy'.format(game_state.turn_number))
        game_state.suppress_warnings(True)

        self.opponent_strategy(game_state)

        game_state.submit_turn()

    def opponent_strategy(self, game_state):
        # Place basic defenses
        self.build_defences(game_state)
        
        # Build reactive defenses based on where the enemy scored
        self.build_reactive_defense(game_state)
        
        try:
            # Debug: Log turn info
            gamelib.debug_write(f"Opponent Turn {game_state.turn_number}: HP={game_state.my_health}, MP={game_state.get_resource(1)}, SP={game_state.get_resource(0)}")
            
            # 0. Initialize combat data for new turn (if not exists)
            if not hasattr(self, 'current_turn_combat_data'):
                self.current_turn_combat_data = {
                    'damage_dealt': 0.0,      # 我方对敌方造成的总伤害
                    'damage_received': 0.0,    # 我方受到的伤害
                    'events': {}               # 保存events用于后续处理
                }
            
            # Use RL model if available
            if self.agent is not None:
                # 1. Build state from GameState
                current_state = self.state.build_state(game_state)
                # Convert tensors to lists for JSON serialization
                board_tensor, scalar_tensor = current_state
                state_for_logging = {
                    "board": board_tensor.tolist(),  # Convert to list for JSON serialization
                    "scalar": scalar_tensor.tolist()  # Convert to list for JSON serialization
                }
                gamelib.debug_write(f"Opponent state shape: board={board_tensor.shape}, scalar={scalar_tensor.shape}")
                if self.reward_tracker.my_last_hp is None or self.reward_tracker.opp_last_hp is None:
                    self.reward_tracker.reset(game_state)
                
                # 2. Log previous turn data if available (delayed logging)
                if self.current_turn_data is not None and not self.turn_data_logged:
                    self._log_turn_data(self.current_turn_data)
                    self.turn_data_logged = True
                
                # 3. Choose Attack Action
                action, log_prob, value = self.agent.act(current_state)
                gamelib.debug_write(f"Opponent selected action: {action}, log_prob: {log_prob:.4f}, value: {value:.4f}")
                
                # 4. Execute action
                self.macro_actions.execute_macro(action, game_state)

                # 5. Cache current turn data for next turn logging
                self.current_turn_data = {
                    "turn_num": game_state.turn_number,
                    "state": state_for_logging,  # Use serializable state format
                    "action": action,
                    "log_prob": log_prob,  # Store log probability for PPO
                    "value": value,  # Store value for GAE calculation
                    "hp": game_state.my_health,
                    "enemy_hp": game_state.enemy_health,
                    "mp": game_state.get_resource(MP),
                    "sp": game_state.get_resource(SP),
                    "terminal": False  # Will be updated if this is the last turn
                }
                self.turn_data_logged = False
                
            else:
                # Fallback to simple strategy
                action = self._choose_simple_action(game_state)
                gamelib.debug_write(f'Opponent fallback action: {action}')
                self._execute_simple_action(action, game_state)
            
        except Exception as e:
            gamelib.debug_write(f'Error in opponent strategy: {e}')
            # Fallback to random action
            action = random.choice([0, 1, 2, 3, 4, 5, 6, 7])
            self._execute_simple_action(action, game_state)

    def _choose_simple_action(self, game_state):
        """
        Choose action using simple heuristics (fallback)
        """
        # Simple strategy: prioritize different actions based on game state
        if game_state.turn_number < 5:
            return 2  # Early game: spawn scouts
        elif game_state.my_health < 20:
            return 1  # Low health: defensive strategy
        else:
            return random.choice([0, 2, 3, 4, 5, 6, 7])  # Random offensive action

    def _execute_simple_action(self, action, game_state):
        """
        Execute action using simple logic (fallback)
        """
        if action == 0:  # Scout rush
            self.scout_rush(game_state)
        elif action == 1:  # Defensive strategy
            self.defensive_strategy(game_state)
        elif action == 2:  # Scout flood
            self.scout_flood(game_state)
        elif action == 3:  # Demolisher line
            self.demolisher_line_strategy(game_state)
        elif action == 4:  # Interceptor stall
            self.stall_with_interceptors(game_state)
        elif action == 5:  # Mixed strategy
            self.mixed_strategy(game_state)
        elif action == 6:  # Aggressive strategy
            self.aggressive_strategy(game_state)
        else:  # action == 7, Skip
            pass

    def scout_rush(self, game_state):
        """Simple scout rush strategy"""
        if game_state.get_resource(MP) >= 1:
            game_state.attempt_spawn(SCOUT, [[13, 0], [14, 0]])

    def scout_flood(self, game_state):
        """Simple scout flood strategy"""
        if game_state.get_resource(MP) >= 1:
            game_state.attempt_spawn(SCOUT, [[13, 0], [14, 0], [15, 0]])

    def demolisher_line_strategy(self, game_state):
        """Simple demolisher strategy"""
        if game_state.get_resource(MP) >= 3:
            game_state.attempt_spawn(DEMOLISHER, [[13, 0], [14, 0]])

    def stall_with_interceptors(self, game_state):
        """Simple interceptor strategy"""
        if game_state.get_resource(MP) >= 1:
            game_state.attempt_spawn(INTERCEPTOR, [[13, 0], [14, 0]])

    def defensive_strategy(self, game_state):
        """Simple defensive strategy"""
        if game_state.get_resource(SP) >= 1:
            game_state.attempt_spawn(TURRET, [[13, 12], [14, 12]])

    def mixed_strategy(self, game_state):
        """Simple mixed strategy"""
        if game_state.get_resource(MP) >= 2:
            game_state.attempt_spawn(SCOUT, [[13, 0]])
            game_state.attempt_spawn(INTERCEPTOR, [[14, 0]])

    def aggressive_strategy(self, game_state):
        """Simple aggressive strategy"""
        if game_state.get_resource(MP) >= 2:
            game_state.attempt_spawn(SCOUT, [[13, 0], [14, 0], [15, 0]])

    def _log_turn_data(self, turn_data):
        """
        Log turn data to the rollout file
        
        Args:
            turn_data: Dictionary containing turn information
        """
        try:
            log_dir = os.path.join(os.path.dirname(__file__), "logs")
            os.makedirs(log_dir, exist_ok=True)
            turn_log_file = os.path.join(log_dir, "current_turns.jsonl")
            
            with open(turn_log_file, 'a') as f:
                f.write(json.dumps(turn_data) + '\n')
            
            gamelib.debug_write(f"Opponent logged turn: action={turn_data['action']}, hp={turn_data['hp']}")
            
        except Exception as e:
            gamelib.debug_write(f"Error logging opponent turn data: {e}")

    def build_defences(self, game_state):
        """
        Build basic defenses using hardcoded locations.
        """
        # Place turrets that attack enemy units
        turret_locations = [[0, 13], [27, 13], [8, 11], [19, 11], [13, 11], [14, 11]]
        game_state.attempt_spawn(TURRET, turret_locations)
        
        # Place walls in front of turrets to soak up damage for them
        wall_locations = [[8, 12], [19, 12]]
        game_state.attempt_spawn(WALL, wall_locations)
        # upgrade walls so they soak more damage
        game_state.attempt_upgrade(wall_locations)

    def build_reactive_defense(self, game_state):
        """
        Build reactive defenses based on where the enemy scored on us from.
        """
        for location in self.scored_on_locations:
            # Build turret one space above so that it doesn't block our own edge spawn locations
            build_location = [location[0], location[1]+1]
            game_state.attempt_spawn(TURRET, build_location)

    def on_action_frame(self, turn_string):
        """
        This is the action frame of the game. This function is called once every 20 game ticks
        and allows for your bot to have a more complex decision making process.
        """
        state = json.loads(turn_string)
        events = state["events"]
        
        breaches = events["breach"]
        for breach in breaches:
            location = breach[0]
            unit_owner_self = True if breach[4] == 1 else False
            if not unit_owner_self:
                gamelib.debug_write("Opponent got scored on at: {}".format(location))
                self.scored_on_locations.append(location)

    def on_game_end(self, result):
        """
        This function is called at the end of the game and allows you to do any cleanup of
        your bot.
        """
        gamelib.debug_write(f"Opponent game ended with result: {result}")
        
        # Log final turn data if available
        if self.current_turn_data is not None and not self.turn_data_logged:
            self.current_turn_data['terminal'] = True
            self._log_turn_data(self.current_turn_data)
        
        # Mark game as ended
        self.game_ended = True

if __name__ == "__main__":
    algo = AlgoStrategy()
    algo.start()
