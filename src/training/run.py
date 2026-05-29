import neat
from neat.nn import RecurrentNetwork
from time import time
import multiprocessing
import random
from src.simulation.simulate import run_episode, create_run_net
from src.training.configs import SEEDS, GENERATIONS, get_config, AVERAGED, DYN_THRESHOLD, N_SPECIES, MAX_THRES, MIN_THRES, ADJUST_RATE, CHECKPOINT, RAND_SEED
from src.results.manager import save_net
from src.simulation.model import SimResult
from src.dynamic.threshold import DynamicThresholdReporter
import numpy as np

random.seed(RAND_SEED)
np.random.seed(RAND_SEED)


def calc_fitness(result: SimResult):
    fitness = result.reward

    if result.has_fallen:
        early_fall_factor = max(0.0, 1.0 - result.steps / 1600.0)
        fitness -= 30.0 * (0.5 + early_fall_factor)
        #fitness -= 30

    if result.has_stopped:
        fitness -= 60.0 * max(0.1, 1.0 - result.steps / 1600.0)
        #fitness -= 60

    return fitness

def eval_genome(genome, config):
    net = RecurrentNetwork.create(genome, config)
    if AVERAGED:
        fitness_list = []
        for seed in SEEDS:
            net.reset()
            fitness_list.append(
                    calc_fitness(run_episode(net, seed=seed))
                    )

        fitness = sum(fitness_list) / len(fitness_list)
    else:
        net.reset()
        fitness = calc_fitness(run_episode(net))

    return float(fitness)

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config)

if __name__ == "__main__":
    

    # Load configuration
    config = get_config()

    # Create population
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))

    pop_size = len(p.population)
    print(f"Population size: {pop_size}")
    
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
        filename_prefix=f'{pop_size}-{start_time}-checkpoint-'
    ))
        


    with neat.ParallelEvaluator(
            multiprocessing.cpu_count(), 
            eval_genome, 
            seed=RAND_SEED
            ) as evaluator:
        #winner = p.run(eval_genomes, 300)
        winner = p.run(evaluator.evaluate, GENERATIONS)

    print('\nBest genome:\n{!s}'.format(winner))

    save_net(winner, f"best_winner_{pop_size}_{GENERATIONS}_{SEEDS[0]}_{AVERAGED}_{start_time}.pkl")

    create_run_net(winner, config)
