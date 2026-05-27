import neat
from neat.nn import RecurrentNetwork
from time import time
import multiprocessing
import random
from src.simulation.simulate import run_episode, create_run_net
from src.training.configs import SEED, GENERATIONS, get_config, AVERAGED, RUNS
from src.results.manager import save_net

def eval_genome(genome, config):
    net = RecurrentNetwork.create(genome, config)
    
    if AVERAGED:
        total_fitness = 0.0
        for _ in range(RUNS):
            total_fitness += run_episode(net)
        fitness = total_fitness/RUNS

    else:
        fitness = run_episode(net)

    return fitness

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config)

if __name__ == "__main__":
    
    random.seed(SEED)

    # Load configuration
    config = get_config()

    # Create population
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))


    with neat.ParallelEvaluator(
            multiprocessing.cpu_count(), 
            eval_genome, 
            seed=SEED
            ) as evaluator:
        #winner = p.run(eval_genomes, 300)
        winner = p.run(evaluator.evaluate, GENERATIONS)

    print('\nBest genome:\n{!s}'.format(winner))

    save_net(winner, f"best_winner_{GENERATIONS}_{SEED}_{int(time())}.pkl")

    create_run_net(winner, config)
