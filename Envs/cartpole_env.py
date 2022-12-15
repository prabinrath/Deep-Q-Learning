import gym
import numpy as np

class CartPoleEnv():
    def __init__(self, name):
        self.env = gym.make(name, render_mode="rgb_array")
        self.obs_dim = self.env.observation_space.shape[0]
        self.act_dim = self.env.action_space.n
        self.action_space = self.env.action_space
        self.n_buffer = 1
        self.buffer = None

    def get_state(self):
        '''
        State Preprocessing
        '''
        if self.buffer == None:
            self.reset()
        return np.vstack(self.buffer)

    def get_reward(self, observation, reward):
        return reward

    def reset(self, seed=None):
        if seed:
            observation, _ = self.env.reset(seed)
        else:
            observation, _ = self.env.reset()
        self.buffer = [observation,]*self.n_buffer

    def step(self, action):
        observation, reward, terminated, truncated, _ = self.env.step(action)
        self.buffer.pop(0)
        self.buffer.append(observation)
        next_state = self.get_state()
        if terminated or truncated:
            self.buffer = None
        return next_state, self.get_reward(observation, reward), terminated or truncated
    
    def render(self):
        return self.env.render()
        