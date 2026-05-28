import neat
from pathlib import Path

SEEDS = [81, 42, 67]
GENERATIONS = 100
CONFIG = 'config-neat-mod'

AVERAGED = False
DYN_THRESHOLD = True
N_SPECIES = 15
ADJUST_RATE = 0.1
MIN_THRES = 0.6
MAX_THRES = 3.0

def get_config():
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         Path(__file__).parent / CONFIG)
    
    return config


