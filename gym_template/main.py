import gymnasium as gym

env = gym.make("BipedalWalker-v3", render_mode="human")
observation, info = env.reset()

print(observation)

episode_over = False
t_reward = 0

steps = 0
while not episode_over:
    action = env.action_space.sample()

    observation, reward, terminated, truncated, info = env.step(action)

    t_reward += float(reward)

    episode_over = terminated or truncated
    
    print(f"\r {steps}: {t_reward}                   ", end="")
    env.render()
    steps += 1

env.close()
