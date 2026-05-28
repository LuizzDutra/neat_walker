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
    result = SimResult()
    
    steps_stuck = 0
    while not episode_over:

        action = net.activate(observation)

        observation, reward, terminated, truncated, info = env.step(action)
        episode_over = terminated or truncated

        hull_angle = observation[0]
        hip_1_angle = observation[4]
        hip_2_angle = observation[9]

        if abs(hip_1_angle) > 1.0 or abs(hip_2_angle) > 1.0:
            result.reward -= 0.2  # Heavy penalty for splaying legs
            
        # Penalize the network if the main body tilts too far forward or backward.
        if abs(hull_angle) > 0.5:
            result.reward -= 0.1
            

        reward = float(reward)
        if reward == -100.0:
            #Has fallen
            result.has_fallen = True

        result.reward += reward
        result.canon_reward += reward
        
        #X speed
        if observation[2] < 0.1:
            steps_stuck += 1
        else:
            steps_stuck = 0

        if steps_stuck > 100:
            result.has_stopped = True
            episode_over = True

        if render:
            print(f"\r {result.steps}: {result.canon_reward}                   ", end="")

        result.steps += 1

    return result


def create_run_net(net, config):
    n_net = RecurrentNetwork.create(net, config)
    run_episode(n_net, render=True)

