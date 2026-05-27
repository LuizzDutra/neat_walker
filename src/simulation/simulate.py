import gymnasium as gym
from src.training.configs import SEED
from neat.nn import RecurrentNetwork

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


def create_run_net(net, config):
    n_net = RecurrentNetwork.create(net, config)
    run_episode(n_net, render=True)

