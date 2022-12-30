import gym
import cv2
import numpy as np
env = gym.make("BreakoutDeterministic-v4")
env.metadata.setdefault('render_fps', 10)

observation = env.reset()
done = False
cv2.namedWindow('Agent', cv2.WINDOW_NORMAL)
while not done:
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    img = env.render(mode="rgb_array").astype(np.float32)/255
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imshow('Agent', img)
    cv2.waitKey(100)    
    # gray = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
    # reshaped = cv2.resize(gray, (84,110))
    # cropped = reshaped[18:102,:]/255
    # cv2.imshow('Agent', cropped)
    # cv2.waitKey(100)
    print(reward, info)
    if done:
        observation = env.reset()
env.close()