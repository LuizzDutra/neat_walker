import neat
from pathlib import Path

SEEDS = [81, 42, 67]
GENERATIONS = 50
CONFIG = 'config-neat-mod'

AVERAGED = True

def get_config():
    return neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         Path(__file__).parent / CONFIG)


