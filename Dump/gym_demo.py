import gym
env = gym.make("PongDeterministic-v4", render_mode="human")
env.metadata.setdefault('render_fps', 10)

observation, info = env.reset(seed=42)
done = False
while not done:
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    print(reward, info)
    done = terminated or truncated
    if done:
        observation, info = env.reset()
env.close()