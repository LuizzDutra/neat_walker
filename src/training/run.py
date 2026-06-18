import neat
from neat.nn import RecurrentNetwork
from time import time
import multiprocessing
import random
from src.simulation.simulate import run_episode, create_run_net
from src.training.configs import SEEDS, GENERATIONS, get_config, AVERAGED, DYN_THRESHOLD, N_SPECIES, MAX_THRES, MIN_THRES, ADJUST_RATE, CHECKPOINT, RAND_SEED, Penalties
from src.results.manager import save_net
from src.simulation.model import SimResult
from src.dynamic.threshold import DynamicThresholdReporter
from src.dynamic.generation import ParallelGenerationTracker
from src.logging.logger import Tee
import numpy as np
import sys

random.seed(RAND_SEED)
np.random.seed(RAND_SEED)


def calc_fitness(result: SimResult, gen):
    fitness = result.reward
    penalties = 0.0
    penalties += result.total_knee * Penalties.knee_coef
    penalties += result.total_splay * Penalties.splay_coef
    penalties += result.total_tilt * Penalties.tilt_coef
    

    fitness -= penalties * min(1.0, gen/Penalties.min_gen)
    

    if result.has_fallen:
        #early_fall_factor = max(0.5, 1.0 - result.steps / 1600.0)
        #fitness -= 80.0 * early_fall_factor
        fitness -= 80

    if result.has_stopped:
        #fitness -= 100.0 * max(0.5, 1.0 - result.steps / 1600.0)
        fitness -= 100

    return fitness

def eval_genome(genome, config):
    net = RecurrentNetwork.create(genome, config)
    gen = config.current_generation
    if AVERAGED:
        fitness_list = []
        for seed in SEEDS:
            net.reset()
            fitness_list.append(
                    calc_fitness(run_episode(net, seed=seed), gen)
                    )

        fitness = sum(fitness_list) / len(fitness_list)
    else:
        net.reset()
        fitness = calc_fitness(run_episode(net), gen)

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

    start_time = int(time())
    pop_size = len(p.population)
    run_string = f"best_winner_{pop_size}_{GENERATIONS}_{SEEDS[0]}_{AVERAGED}_{start_time}"
    
    log_file = open(run_string+".log", "w")
    sys.stdout = Tee(sys.stdout, log_file)


    if DYN_THRESHOLD:
        p.add_reporter(
                DynamicThresholdReporter(N_SPECIES, 
                                         ADJUST_RATE, 
                                         MIN_THRES, 
                                         MAX_THRES)
                )
    
    #Checkpoint  
    p.add_reporter(neat.Checkpointer(
        generation_interval=CHECKPOINT,
        time_interval_seconds=None,
        filename_prefix=f'{pop_size}-{start_time}-checkpoint-'
    ))

    print(f"Population size: {pop_size}")
    with neat.ParallelEvaluator(
            multiprocessing.cpu_count(), 
            eval_genome, 
            seed=RAND_SEED
            ) as evaluator:
        
        tracker = ParallelGenerationTracker(evaluator, p)

        winner = p.run(tracker.evaluate, GENERATIONS)

    print('\nBest genome:\n{!s}'.format(winner))

    save_net(winner, f"{run_string}.pkl")

    create_run_net(winner, config)
    sys.stdout =  sys.__stdout__
    log_file.close()
