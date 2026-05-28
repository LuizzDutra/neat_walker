import neat
from pathlib import Path

SEEDS = [81, 42, 67]
GENERATIONS = 50
CHECKPOINT = 50
CONFIG = 'config-neat-mod'

AVERAGED = True
DYN_THRESHOLD = True
N_SPECIES = 15
ADJUST_RATE = 0.02
MIN_THRES = 0.8
MAX_THRES = 2.5

def get_config():
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         Path(__file__).parent / CONFIG)
    
    return config


