import gymnasium as gym
from neat.nn import RecurrentNetwork
from src.simulation.model import SimResult

env_default = gym.make("BipedalWalker-v3")
env_render = gym.make("BipedalWalker-v3", render_mode="human")

def run_episode(net, render=False, seed: int | None = None) -> SimResult:
    if render:
        env = env_render
    else:
        env = env_default

    env.reset(seed=seed)
    observation, info = env.reset()
    episode_over: bool = False
    result = SimResult(reward=0, 
                       steps=0, 
                       has_fallen=False, 
                       has_stopped=False)

    steps_stuck = 0
    while not episode_over:

        action = net.activate(observation)

        observation, reward, terminated, truncated, info = env.step(action)
        episode_over = terminated or truncated
        reward = float(reward)
        if reward == -100.0:
            #Has fallen
            result.has_fallen = True

        result.reward += reward
        
        #X speed
        if observation[2] < 0.1:
            steps_stuck += 1
        else:
            steps_stuck = 0

        if steps_stuck > 100:
            result.has_stopped = True
            episode_over = True

        if render:
            print(f"\r {result.steps}: {result.reward}                   ", end="")

        result.steps += 1

    return result


def create_run_net(net, config):
    n_net = RecurrentNetwork.create(net, config)
    run_episode(n_net, render=True)

