# Macro actions
MACROS = {
    0: "SAVE_BITS",
    1: "SCOUT_FLOOD_LEFT",
    2: "SCOUT_FLOOD_RIGHT",
    3: "DEMOLISHER_LINE_RIGHT",
    4: "DEMOLISHER_LINE_LEFT",
    5: "INTERCEPTOR_NET_GLOBAL",
    6: "INTERCEPTOR_NET_LEFT",
    7: "INTERCEPTOR_NET_RIGHT",
}


# Q-learning hyperparameters
ALPHA = 0.2 # Learning rate
GAMMA = 0.95 # Discount factor
EPS_START, EPS_END, EPS_DECAY_STEPS = 0.2, 0.05, 5000 # Exploration rate: start from 0.2, end at 0.05, decay over 5000 steps

# State discretization
BITS_BINS  = [6, 11, 16]   # -> 0:[0-5],1:[6-10],2:[11-15],3:[16+]
CORES_BINS = [6, 13]       # -> 0:[0-5],1:[6-12],2:[13+]
