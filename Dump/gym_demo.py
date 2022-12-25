import gym
# import cv2
env = gym.make("Pong-v4")
env.metadata.setdefault('render_fps', 10)

observation = env.reset()
done = False
while not done:
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    # gray = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)  
    # reshaped = cv2.resize(gray, (84,110))
    # cropped = reshaped[18:102,:]
    # cv2.imwrite('sample.png',cropped)
    print(reward, info)
    # break
    if done:
        observation = env.reset()
env.close()