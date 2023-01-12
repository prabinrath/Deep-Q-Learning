import gym
import numpy as np
import cv2

class PongEnv():
    def __init__(self, name):
        self.env = gym.make(name)
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
        cropped = reshaped[18:102,:]
        return np.expand_dims(cropped, 0)

    def get_state(self):
        if self.buffer == None:
            self.reset()
        return np.expand_dims(np.vstack(self.buffer), 0)

    def get_reward(self, observation, reward):
        return reward

    def reset(self, seed=None):
        if seed:
            observation = self.env.reset(seed)
        else:
            observation = self.env.reset()
        observation = self.pre_process(observation)
        self.buffer = [observation,]*self.n_buffer

    def step(self, action):
        observation, reward, terminated, _ = self.env.step(action)
        observation = self.pre_process(observation)
        self.buffer.pop(0)
        self.buffer.append(observation)
        next_state = self.get_state()
        if terminated:
            self.buffer = None
        return next_state, self.get_reward(observation, reward), terminated, terminated
    
    def render(self):
        return self.env.render(mode="rgb_array").astype(np.float32)/255
        
class BreakoutEnv():
    def __init__(self, name):
        self.env = gym.make(name)
        self.obs_dim = self.env.observation_space.shape[0]
        self.act_dim = self.env.action_space.n
        self.action_space = self.env.action_space
        self.n_buffer = 4
        self.buffer = None
        self.life = 5

    def pre_process(self, observation):
        '''
        State Preprocessing
        '''
        gray = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)  
        reshaped = cv2.resize(gray, (84,110))
        cropped = reshaped[18:102,:]
        return np.expand_dims(cropped, 0)

    def get_state(self):
        if self.buffer == None:
            self.reset()
        return np.expand_dims(np.vstack(self.buffer), 0)

    def get_reward(self, observation, reward, info):
        reward += info['ale.lives'] - self.life
        self.life = info['ale.lives']
        return reward

    def reset(self, seed=None):
        if seed:
            observation = self.env.reset(seed)
        else:
            observation = self.env.reset()
        observation = self.pre_process(observation)
        self.buffer = [observation,]*self.n_buffer
        self.life = 5

    def step(self, action):
        observation, reward, terminated, info = self.env.step(action)
        observation = self.pre_process(observation)
        self.buffer.pop(0)
        self.buffer.append(observation)
        next_state = self.get_state()
        reward = self.get_reward(observation, reward, info)
        terminal_life_lost = False
        if reward == -1:
            terminal_life_lost = True            
            reward = 0
            # terminal_life_lost = terminated
        if terminated:
            self.buffer = None
        
        return next_state, reward, terminated, terminal_life_lost
    
    def render(self):
        return self.env.render(mode="rgb_array").astype(np.float32)/255
