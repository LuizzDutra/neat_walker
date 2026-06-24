import neat
from neat.nn import FeedForwardNetwork
from time import time
import multiprocessing
import random
import numpy as np
import sys
import os

from src.simulation.simulate import run_episode, create_run_net
from src.training.configs import (
    SEEDS, GENERATIONS, get_config, AVERAGED,
    DYN, N_SPECIES, MAX_THRES, MIN_THRES, ADJUST_RATE,
    CHECKPOINT, RAND_SEED, Penalties,
    USE_SPECIES_SCHEDULE, USE_WEIGHT_DECAY,
    get_species_schedule, get_weight_decay,
)
from src.results.manager import save_net
from src.simulation.model import SimResult
from src.dynamic.threshold import DynamicReporter 
from src.dynamic.generation import ParallelGenerationTracker
from src.logging.logger import Tee

random.seed(RAND_SEED)
np.random.seed(RAND_SEED)


def calc_fitness(result: SimResult, gen: int) -> float:
    fitness = result.reward
    penalties = 0.0
    penalties += result.total_knee * Penalties.knee_coef
    penalties += result.total_splay * Penalties.splay_coef
    penalties += result.total_tilt * Penalties.tilt_coef

    fitness -= penalties * min(1.0, gen / Penalties.min_gen)

    if result.has_fallen:
        fitness -= 80

    if result.has_stopped:
        fitness -= 100

    return fitness


def eval_genome(genome, config):
    net = FeedForwardNetwork.create(genome, config)
    gen = config.current_generation

    if AVERAGED:
        fitness_list = [
            calc_fitness(run_episode(net, seed=seed), gen)
            for seed in SEEDS
        ]
        fitness = sum(fitness_list) / len(fitness_list)
    else:
        fitness = calc_fitness(run_episode(net), gen)

    return float(fitness)


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config)


if __name__ == "__main__":

    config = get_config()

    species_schedule = get_species_schedule()
    weight_decay = get_weight_decay()

    condition_parts = []
    if USE_SPECIES_SCHEDULE:
        condition_parts.append("dyn-species")
    if USE_WEIGHT_DECAY:
        condition_parts.append("weight-decay")
    condition_label = "_".join(condition_parts) if condition_parts else "baseline"

    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))

    start_time = int(time())
    pop_size = len(p.population)
    run_string = (
        f"best_winner_{pop_size}_{GENERATIONS}_{SEEDS[0]}"
        f"_{AVERAGED}_{condition_label}_{start_time}"
    )

    os.makedirs('logs', exist_ok=True)
    log_file = open('logs' + os.sep + run_string + ".log", "w")
    sys.stdout = Tee(sys.stdout, log_file)

    print(f"Condition: {condition_label}")
    print(f"USE_SPECIES_SCHEDULE={USE_SPECIES_SCHEDULE}  USE_WEIGHT_DECAY={USE_WEIGHT_DECAY}")

    if DYN:
        p.add_reporter(
            DynamicReporter(
                target_species=N_SPECIES,
                adjust_rate=ADJUST_RATE,
                min_thresh=MIN_THRES,
                max_thresh=MAX_THRES,
                species_schedule=species_schedule,
                weight_decay=weight_decay,
            )
        )

    os.makedirs('checkpoints', exist_ok=True)
    p.add_reporter(neat.Checkpointer(
        generation_interval=CHECKPOINT,
        time_interval_seconds=None,
        filename_prefix='checkpoints' + os.sep + f'{pop_size}-{start_time}-{condition_label}-checkpoint-'
    ))

    print(f"Population size: {pop_size}")

    with neat.ParallelEvaluator(
        multiprocessing.cpu_count(),
        eval_genome,
        seed=RAND_SEED,
    ) as evaluator:
        tracker = ParallelGenerationTracker(evaluator, p)
        winner = p.run(tracker.evaluate, GENERATIONS)

    print('\nBest genome:\n{!s}'.format(winner))

    save_net(winner, f"{run_string}.pkl")
    create_run_net(winner, config)

    sys.stdout = sys.__stdout__
    log_file.close()
