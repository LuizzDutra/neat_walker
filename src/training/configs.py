import neat
from pathlib import Path

SEED = 81
GENERATIONS = 80
CONFIG = 'config-neat-mod'

AVERAGED = True
RUNS = 3

def get_config():
    return neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         Path(__file__).parent / CONFIG)


