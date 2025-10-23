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
Wire-up (how it runs in a match)

During on_game_start: init agent, reward tracker, and any static placements you want for round 0–1.

During each on_turn:

- Build obs_t from GameState (mp/sp/hp/lane).
- If have_last_obs_action: compute reward_{t-1} from HP deltas → agent.update(last_obs, last_action, reward, obs_t).

action_t = agent.act(obs_t) → dispatch to macros.py to execute.

Cache (last_obs, last_action) ← (obs_t, action_t); submit_turn().

During on_action_frame: update breach counters if you want shaping.

At game end: do one last update with terminal bonus.

This is fully self-contained and will run with run_match.py.
"""

class AlgoStrategy(gamelib.AlgoCore):
    def __init__(self):
        super().__init__()
        seed = random.randrange(maxsize)
        random.seed(seed)
        gamelib.debug_write('Random seed: {}'.format(seed))

    def on_game_start(self, config):
        """ 
        Read in config and perform any initial setup here 
        """
        gamelib.debug_write('Configuring your custom algo strategy...')
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
        # This is a good place to do initial setup
        self.scored_on_locations = []
        
        # My-First-Algo
        self.last_state = None
        self.last_action = None
        self.reward_tracker = RewardTracker()
        
        # Initialize agent with model path and CNN parameters
        model_path = os.path.join(os.path.dirname(__file__), "rl_model.pkl")
        self.agent = Agent(model_path=model_path, board_channels=4, scalar_dim=7, action_dim=4, hidden_dim=128)
        
        self.state = State(config)
        self.macro_actions = MacroActions(config)
        self.game_ended = False
        
        # Rollout data caching for proper PPO data collection
        self.current_turn_data = None  # Cache for current turn data
        self.turn_data_logged = False  # Flag to track if current turn data is logged
        
        # Get game number from environment variable
        self.game_num = int(os.environ.get('GAME_NUM', 1))
        
        # Reset for new game
        self.reset_for_new_game()
        
    def on_turn(self, turn_state):
        """
        This function is called every turn with the game state wrapper as
        an argument. The wrapper stores the state of the arena and has methods
        for querying its state, allocating your current resources as planned
        unit deployments, and transmitting your intended deployments to the
        game engine.
        """
        game_state = gamelib.GameState(self.config, turn_state)
        gamelib.debug_write('Performing turn {} of your custom algo strategy'.format(game_state.turn_number))
        game_state.suppress_warnings(True)  #Comment or remove this line to enable warnings.
        
        self.my_strategy(game_state)

        game_state.submit_turn()
    
    def my_strategy(self, game_state):
        try:
            # Debug: Log turn info
            gamelib.debug_write(f"RL Turn {game_state.turn_number}: HP={game_state.my_health}, MP={game_state.get_resource(1)}, SP={game_state.get_resource(0)}")
            
            # 1. Build state from GameState
            current_state = self.state.build_state(game_state)
            # Convert tensors to lists for JSON serialization
            board_tensor, scalar_tensor = current_state
            state_for_logging = {
                "board": board_tensor.tolist(),  # Convert to list for JSON serialization
                "scalar": scalar_tensor.tolist()  # Convert to list for JSON serialization
            }
            gamelib.debug_write(f"Current state shape: board={board_tensor.shape}, scalar={scalar_tensor.shape}")
            if self.reward_tracker.my_last_hp is None or self.reward_tracker.opp_last_hp is None:
                self.reward_tracker.reset(game_state)
            
            # 2. Log previous turn data if available (delayed logging)
            if self.current_turn_data is not None and not self.turn_data_logged:
                self._log_turn_data(self.current_turn_data)
                self.turn_data_logged = True
            
            # 3. Choose Attack Action
            action, log_prob, value = self.agent.act(current_state)
            gamelib.debug_write(f"Selected action: {action}, log_prob: {log_prob:.4f}, value: {value:.4f}")
            
            # 4. Execute action
            self.macro_actions.execute_macro(action, game_state)

            # 5. Cache current turn data for next turn logging
            self.current_turn_data = {
                "turn_num": game_state.turn_number,
                "state": state_for_logging,  # Use serializable state format
                "action": action,
                "log_prob": log_prob,  # Store log probability for PPO
                "value": value,  # Store value for GAE calculation
                "reward": 0.0,  # Will be calculated in train_rl.py
                "hp": game_state.my_health,
                "enemy_hp": game_state.enemy_health,
                "terminal": False
            }
            self.turn_data_logged = False  # Reset flag for next turn
            
            # 6. Update last_state, last_action, last_log_prob, and last_value
            self.last_state = current_state
            self.last_action = action
            self.last_log_prob = log_prob
            self.last_value = value
            
            # 7. Save model periodically (every 5 turns) to ensure we don't lose progress
            # if game_state.turn_number % 5 == 0:
            #     self.save_model(model_name="rl_model.pkl")
            #     gamelib.debug_write(f"Periodic model save at turn {game_state.turn_number}")
            
        except Exception as e:
            gamelib.debug_write(f"Error in my_strategy: {e}")
            import traceback
            traceback.print_exc()
        
       
            
        # Place basic defenses
        self.build_defences(game_state)
        
        # Build reactive defenses based on where the enemy scored
        self.build_reactive_defense(game_state)
    
    def reset_for_new_game(self):
        """
        Reset RL state for a new game/episode.
        """
        self.last_state = None
        self.last_action = None
        self.last_log_prob = None
        self.last_value = None
        self.game_ended = False
        self.reward_tracker.reset()
        
        # Reset rollout data caching
        self.current_turn_data = None
        self.turn_data_logged = False
        
        # Note: We keep the same agent and reward_tracker to preserve learning
    
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
            
            gamelib.debug_write(f"Logged turn: action={turn_data['action']}, reward={turn_data['reward']}, hp={turn_data['hp']}")
            
        except Exception as e:
            gamelib.debug_write(f"Error logging turn data: {e}")
    
    def save_model(self, model_name='rl_model.pkl'):
        """Save the RL model"""
        try:
            model_path = os.path.join(os.path.dirname(__file__), model_name)
            # Get model stats before saving
            stats = self.agent.get_model_stats()
            gamelib.debug_write(f"Saving model with stats: {stats}")
            self.agent.save_model(model_path)
            gamelib.debug_write("Saved RL model")
        except Exception as e:
            gamelib.debug_write(f"Error saving RL model: {e}")
        
    """
    Defensive actions
    """
    
    def build_defences(self, game_state):
        """
        Build basic defenses using hardcoded locations.
        Remember to defend corners and avoid placing units in the front where enemy demolishers can attack them.
        """
        # Useful tool for setting up your base locations: https://www.kevinbai.design/terminal-map-maker
        # More community tools available at: https://terminal.c1games.com/rules#Download

        # Place turrets that attack enemy units
        turret_locations = [[0, 13], [27, 13], [8, 11], [19, 11], [13, 11], [14, 11]]
        # attempt_spawn will try to spawn units if we have resources, and will check if a blocking unit is already there
        game_state.attempt_spawn(TURRET, turret_locations)
        
        # Place walls in front of turrets to soak up damage for them
        wall_locations = [[8, 12], [19, 12]]
        game_state.attempt_spawn(WALL, wall_locations)
        # upgrade walls so they soak more damage
        game_state.attempt_upgrade(wall_locations)
    
    def build_reactive_defense(self, game_state):
        """
        This function builds reactive defenses based on where the enemy scored on us from.
        We can track where the opponent scored by looking at events in action frames 
        as shown in the on_action_frame function
        """
        for location in self.scored_on_locations:
            # Build turret one space above so that it doesn't block our own edge spawn locations
            build_location = [location[0], location[1]+1]
            game_state.attempt_spawn(TURRET, build_location)
    
    def on_action_frame(self, turn_string):
        """
        This is the action frame of the game. This function could be called 
        hundreds of times per turn and could slow the algo down so avoid putting slow code here.
        Processing the action frames is complicated so we only suggest it if you have time and experience.
        Full doc on format of a game frame at in json-docs.html in the root of the Starterkit.
        """
        # Let's record at what position we get scored on
        state = json.loads(turn_string)
        events = state["events"]
        breaches = events["breach"]
        for breach in breaches:
            location = breach[0]
            unit_owner_self = True if breach[4] == 1 else False
            # When parsing the frame data directly, 
            # 1 is integer for yourself, 2 is opponent (StarterKit code uses 0, 1 as player_index instead)
            if not unit_owner_self:
                gamelib.debug_write("Got scored on at: {}".format(location))
                self.scored_on_locations.append(location)
                gamelib.debug_write("All locations: {}".format(self.scored_on_locations))


if __name__ == "__main__":
    algo = AlgoStrategy()
    algo.start()
