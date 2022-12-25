import gym
env = gym.make("PongDeterministic-v4")
env.metadata.setdefault('render_fps', 10)

observation = env.reset()
done = False
while not done:
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    print(reward, info)
    if done:
        observation = env.reset()
env.close()