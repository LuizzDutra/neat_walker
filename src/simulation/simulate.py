import gymnasium as gym
from src.training.configs import SEED
from neat.nn import RecurrentNetwork

env_default = gym.make("BipedalWalker-v3")
env_render = gym.make("BipedalWalker-v3", render_mode="human")

def run_episode(net, render=False) -> float:
    if render:
        env = env_render
    else:
        env = env_default
    #env.reset(seed=SEED)
    env.reset()
    observation, info = env.reset()
    episode_over: bool = False
    t_reward: float = 0

    steps = 0
    steps_stuck = 0
    while not episode_over:

        action = net.activate(observation)

        observation, reward, terminated, truncated, info = env.step(action)
        episode_over = terminated or truncated
        reward = float(reward)
        if reward == -100.0:
            #Dampens fall penalty
            reward = -50.0

        t_reward += reward
        
        #X speed
        if observation[2] < 0.1:
            steps_stuck += 1
        else:
            steps_stuck = 0

        if steps_stuck > 100:
            t_reward -= 100
            episode_over = True

        if render:
            print(f"\r {steps}: {t_reward}                   ", end="")

        steps += 1

    return t_reward


def create_run_net(net, config):
    n_net = RecurrentNetwork.create(net, config)
    run_episode(n_net, render=True)

