import neat
from pathlib import Path
from src.dynamic.species_schedule import SpeciesSchedule, ScheduleMode
from src.dynamic.weight_decay import WeightDecaySchedule, DecayMode

RAND_SEED = 81
SEEDS = [1692, 214, 67]
GENERATIONS = 300
CHECKPOINT = 50
CONFIG = 'config-neat-mod'

AVERAGED = True
DYN = True
N_SPECIES = 20
ADJUST_RATE = 0.03
MIN_THRES = 1.0
MAX_THRES = 4.0

USE_SPECIES_SCHEDULE = False
USE_WEIGHT_DECAY = False

SPECIES_SCHEDULE_MODE = ScheduleMode.STAGED
SPECIES_S_START = N_SPECIES
SPECIES_S_MIN = 4

WEIGHT_DECAY_MODE = DecayMode.EXPONENTIAL
WEIGHT_SIGMA_START = 0.5
WEIGHT_SIGMA_MIN = 0.02
WEIGHT_DECAY_RATE = 2.5


class Penalties:
    knee_coef = 0.25
    splay_coef = 0.25
    tilt_coef = 0.2
    min_gen = 100


def get_config():
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        Path(__file__).parent / CONFIG,
    )
    return config


def get_species_schedule() -> SpeciesSchedule | None:
    if not USE_SPECIES_SCHEDULE:
        return None
    schedule = SpeciesSchedule(
        mode=SPECIES_SCHEDULE_MODE,
        s_start=SPECIES_S_START,
        s_min=SPECIES_S_MIN,
        gen_max=GENERATIONS,
    )
    print(schedule.summary())
    return schedule


def get_weight_decay() -> WeightDecaySchedule | None:
    if not USE_WEIGHT_DECAY:
        return None
    schedule = WeightDecaySchedule(
        mode=WEIGHT_DECAY_MODE,
        sigma_start=WEIGHT_SIGMA_START,
        sigma_min=WEIGHT_SIGMA_MIN,
        gen_max=GENERATIONS,
        decay_rate=WEIGHT_DECAY_RATE,
    )
    print(schedule.summary())
    return schedule
