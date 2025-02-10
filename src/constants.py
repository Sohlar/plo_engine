# Game constants
STARTING_STACK = 200
MINIMUM_BET_INCREMENT = 2
SMALL_BLIND = 1
BIG_BLIND = 2
INITIAL_POT = 3

# State encoding
STATE_SIZE = 8 + (5*2) + 2*4*2
ACTION_SIZE = 4

# Action mappings
ACTION_MAP = {
    "fold": 0,
    "check": 1, 
    "call": 2,
    "bet": 3
} 