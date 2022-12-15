import numpy as np
import torch
import torch.optim as optim
from get_env_and_learner import GetEnvAndLearner
from collections import namedtuple
from dqn_utils import ReplayMemory
import cv2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

RENDER = False

GAMMA = 0.99 # Discount factor
UPDATE_INTERVAL = 20 # Interval for target update
LR = 1e-3 # AdamW learning rate
EPSILON_START = 1 # Annealing start
EPSILON_END = 0.05 # Annealing end
EXPLORATION_FRAMES = 10000 # Annealing frames
BATCH_SIZE = 64 # Sampling size from memory
MEMORY_BUFFER = 100 # Replay buffer size
EPISODES = 10 # Number of episodes for training

# environment, training policy, target policy, frames to consider for each observation
env, policy, target = GetEnvAndLearner(name = 'CartPole-v1')

# Named tuple for storing transitions
Transition = namedtuple('Transition', 'state action reward next_state')

memory = ReplayMemory(MEMORY_BUFFER)
glob_frame = 0

def select_action(state, act_dim):
    # Linear Annealing
    eps = EPSILON_END + (EXPLORATION_FRAMES-glob_frame)*(EPSILON_START-EPSILON_END)/EXPLORATION_FRAMES \
        if glob_frame < EXPLORATION_FRAMES else EPSILON_END
    if np.random.uniform() < eps:
        return np.random.choice(act_dim)
    else:
        with torch.no_grad():
            policy.eval()
            q_values = policy(torch.tensor(state, device=device))
        return torch.argmax(q_values[0]).item()

def optimize_dqn():
    pass

for episode in range(EPISODES):
    if episode%10==0:
        RENDER = True
    # Default max episode steps is defined in Gym environments
    done = False
    while not done:       
        state = env.get_state()
        if RENDER:
            rgb = env.render()
            cv2.imshow('Cart Pole', rgb)
            cv2.waitKey(100)
        action = select_action(state, env.act_dim)
        next_state, reward, done = env.step(action)

        memory.push(Transition(state, action, reward, next_state))
        if memory.length()<BATCH_SIZE:
            continue
        else:
            samples = memory.sample(BATCH_SIZE)
            

        glob_frame+=1
        if glob_frame%UPDATE_INTERVAL==0:
            pass

    RENDER = False
