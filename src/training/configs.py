import neat
from pathlib import Path

SEED = 81
GENERATIONS = 50
CONFIG = 'config-neat-mod'

AVERAGED = False
RUNS = 3

def get_config():
    return neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         Path(__file__).parent / CONFIG)


