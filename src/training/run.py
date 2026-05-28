import neat
from neat.nn import RecurrentNetwork
from time import time
import multiprocessing
import random
from src.simulation.simulate import run_episode, create_run_net
from src.training.configs import SEEDS, GENERATIONS, get_config, AVERAGED, DYN_THRESHOLD, N_SPECIES, MAX_THRES, MIN_THRES, ADJUST_RATE, CHECKPOINT
from src.results.manager import save_net
from src.simulation.model import SimResult
from src.dynamic.threshold import DynamicThresholdReporter
import numpy as np


def calc_fitness(result: SimResult):
    fitness = result.reward

    if result.has_fallen:
        early_fall_factor = max(0.0, 1.0 - result.steps / 800.0)
        fitness -= 30.0 * (0.5 + early_fall_factor)

    if result.has_stopped:
        fitness -= 60.0 * max(0.1, 1.0 - result.steps / 1600.0)

    return fitness

def eval_genome(genome, config):
    net = RecurrentNetwork.create(genome, config)
    random.seed(SEEDS[0] + genome.key)
    np.random.seed(SEEDS[0] + genome.key)
    if AVERAGED:
        fitness_list = [calc_fitness(run_episode(net, seed=seed))
                        for seed in SEEDS]

        fitness = sum(fitness_list) / len(fitness_list)
    else:
        fitness = calc_fitness(run_episode(net))

    return float(fitness)

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config)

if __name__ == "__main__":
    
    random.seed(SEEDS[0])
    np.random.seed(SEEDS[0])

    # Load configuration
    config = get_config()

    # Create population
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    
    if DYN_THRESHOLD:
        p.add_reporter(
                DynamicThresholdReporter(N_SPECIES, 
                                         ADJUST_RATE, 
                                         MIN_THRES, 
                                         MAX_THRES)
                )
    
    
    start_time = int(time())

    #Checkpoint  
    p.add_reporter(neat.Checkpointer(
        generation_interval=CHECKPOINT,
        time_interval_seconds=None,
        filename_prefix=f'{start_time}-checkpoint-'
    ))
    # ------------------------------
        


    with neat.ParallelEvaluator(
            multiprocessing.cpu_count(), 
            eval_genome, 
            seed=SEEDS[0]
            ) as evaluator:
        #winner = p.run(eval_genomes, 300)
        winner = p.run(evaluator.evaluate, GENERATIONS)

    print('\nBest genome:\n{!s}'.format(winner))

    save_net(winner, f"best_winner_{GENERATIONS}_{SEEDS[0]}_{AVERAGED}_{start_time}.pkl")

    create_run_net(winner, config)
