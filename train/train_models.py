import gymnasium as gym
import neat
from neat.nn import RecurrentNetwork
import pickle
from time import time
import multiprocessing
import random

SEED = 81
GENERATIONS = 20

def run_episode(net, render=False) -> float:
    if render:
        env = gym.make("BipedalWalker-v3", render_mode="human")
    else:
        env = gym.make("BipedalWalker-v3")
    env.reset(seed=SEED)
    observation, info = env.reset()
    episode_over: bool = False
    t_reward: float = 0

    steps = 0
    while not episode_over:

        action = net.activate(observation)

        observation, reward, terminated, truncated, info = env.step(action)

        t_reward += float(reward)

        episode_over = terminated or truncated
        
        if render:
            print(f"\r {steps}: {t_reward}                   ", end="")
            #env.render()
        steps += 1
    env.close()
    return t_reward

def eval_genome(genome, config):
    net = RecurrentNetwork.create(genome, config)
    fitness = run_episode(net)
    #fitness = max(0.001, fitness)
    return fitness

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config)

def create_run_net(net, config):
    n_net = RecurrentNetwork.create(net, config)
    run_episode(n_net, render=True)

def get_config():
    return neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         'config-neat-mod')

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

    with open(f"winners/best_winner_{int(time())}.pkl", "wb") as f:
        pickle.dump(winner, f)

    create_run_net(winner, config)
