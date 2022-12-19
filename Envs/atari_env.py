import gym
import numpy as np
import cv2

class AtariEnv():
    def __init__(self, name):
        self.env = gym.make(name, render_mode="rgb_array")
        self.obs_dim = self.env.observation_space.shape[0]
        self.act_dim = self.env.action_space.n
        self.action_space = self.env.action_space
        self.n_buffer = 4
        self.buffer = None

    def pre_process(self, observation):
        '''
        State Preprocessing
        '''
        gray = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)  
        reshaped = cv2.resize(gray, (84,110))
        cropped = reshaped[18:102,:]/255
        return np.expand_dims(cropped, 0)

    def get_state(self):
        if self.buffer == None:
            self.reset()
        return np.expand_dims(np.vstack(self.buffer), 0)

    def get_reward(self, observation, reward):
        return reward

    def reset(self, seed=None):
        if seed:
            observation, _ = self.env.reset(seed)
        else:
            observation, _ = self.env.reset()
        observation = self.pre_process(observation)
        self.buffer = [observation,]*self.n_buffer

    def step(self, action):
        observation, reward, terminated, truncated, _ = self.env.step(action)
        observation = self.pre_process(observation)
        self.buffer.pop(0)
        self.buffer.append(observation)
        next_state = self.get_state()
        if terminated or truncated:
            self.buffer = None
        return next_state, self.get_reward(observation, reward), terminated or truncated
    
    def render(self):
        return self.env.render()/255
        