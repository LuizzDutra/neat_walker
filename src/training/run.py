import neat
from neat.nn import RecurrentNetwork
from time import time
import multiprocessing
import random
from src.simulation.simulate import run_episode, create_run_net
from src.training.configs import SEEDS, GENERATIONS, get_config, AVERAGED
from src.results.manager import save_net
from src.simulation.model import SimResult

def calc_fitness(result: SimResult):
    fitness = result.reward
    if result.has_fallen:
        fitness -= 100
    if result.has_stopped:
        fitness -= 100
    return fitness

def eval_genome(genome, config):
    net = RecurrentNetwork.create(genome, config)
    fitness_list: list[float] = [0.0]*len(SEEDS)
    if AVERAGED:
        for i, seed in enumerate(SEEDS):
            fitness_list[i] = calc_fitness(run_episode(net, seed=seed))
        fitness = min(fitness_list)

    else:
        fitness = calc_fitness(run_episode(net))

    return fitness

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config)

if __name__ == "__main__":
    
    random.seed(SEEDS[0])

    # Load configuration
    config = get_config()

    # Create population
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))


    with neat.ParallelEvaluator(
            multiprocessing.cpu_count(), 
            eval_genome, 
            seed=SEEDS[0]
            ) as evaluator:
        #winner = p.run(eval_genomes, 300)
        winner = p.run(evaluator.evaluate, GENERATIONS)

    print('\nBest genome:\n{!s}'.format(winner))

    save_net(winner, f"best_winner_{GENERATIONS}_{SEEDS[0]}_{AVERAGED}_{int(time())}.pkl")

    create_run_net(winner, config)
