import gymnasium as gym
from neat.nn import RecurrentNetwork
from src.simulation.model import SimResult


def run_episode(net, render=False, seed: int | None = None) -> SimResult:
    if render:
        env = gym.make("BipedalWalker-v3", render_mode="human")
    else:
        env = gym.make("BipedalWalker-v3")

    observation, info = env.reset(seed=seed)
    episode_over: bool = False
    result = SimResult()
    
    steps_stuck = 0
    knee_penalty_thres = 0.35 # < 0 = folded; > 0 streched-ish

    while not episode_over:

        action = net.activate(observation)

        observation, reward, terminated, truncated, info = env.step(action)
        episode_over = terminated or truncated

        hull_angle = observation[0]
        hip_1_angle = observation[4]
        hip_2_angle = observation[9]

        knee_1_angle = observation[6]
        knee_2_angle = observation[11]
        leg_1_contact = observation[8]
        leg_2_contact = observation[13]

        knee_1_pen = (
                max(0.0, -knee_1_angle + knee_penalty_thres) * 
                leg_1_contact *
                int((hip_1_angle - abs(hull_angle)) < 0.3)
                )
        knee_2_pen = (
                max(0.0, -knee_2_angle + knee_penalty_thres) * 
                leg_2_contact *
                int((hip_2_angle - abs(hull_angle)) < 0.3)
                )
        knee_penalty = (knee_1_pen + knee_2_pen)
        result.total_knee += knee_penalty


        splay_1 = max(0.0, abs(hip_1_angle) - 1.0)
        splay_2 = max(0.0, abs(hip_2_angle) - 1.0)
        result.total_splay += (splay_1 + splay_2)

        tilt = max(0.0, abs(hull_angle) - 0.4)
        result.total_tilt += tilt

        reward = float(reward)
        if reward == -100.0:
            reward = 0
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
            print(f"\rSteps: {result.steps}: Canon reward: {result.canon_reward:.3f} {result.total_knee}  ", end="")

        result.steps += 1

    if render:
        print()
        print("Final reward: ", result.reward)
        print("Penalties")
        print("Knee: ", result.total_knee)
        print("Splay: ", result.total_splay)
        print("Tilt: ", result.total_tilt)
    env.close()
    return result


def create_run_net(net, config):
    n_net = RecurrentNetwork.create(net, config)
    run_episode(n_net, render=True, seed=101)

