import neat
from pathlib import Path

RAND_SEED = 81
SEEDS = [1692, 214, 67]
GENERATIONS = 200
CHECKPOINT = 50
CONFIG = 'config-neat-mod'

AVERAGED = True
DYN_THRESHOLD = True
N_SPECIES = 20
ADJUST_RATE = 0.02
MIN_THRES = 2.0
MAX_THRES = 3.0

class Penalties:
    knee_coef = 0.2
    splay_coef = 0.2
    tilt_coef = 0.2
    min_gen = 50

def get_config():
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         Path(__file__).parent / CONFIG)
    
    return config


