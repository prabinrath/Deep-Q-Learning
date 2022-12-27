import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from get_env_and_learner import GetEnvAndLearner
from dqn_utils import ReplayMemory
from copy import deepcopy
import cv2
from collections import deque
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Constant Parameters
RENDER = False
GAMMA = 0.97 # Discount factor
UPDATE_INTERVAL = 1000 # Interval for target update
LR = 0.001 # Adam learning rate
EPSILON_START = 1 # Annealing start
EPSILON_END = 0.05 # Annealing end
EXPLORATION_FRAMES = 500000 # Annealing frames
BATCH_SIZE = 64 # Sampling size from memory
MEMORY_BUFFER = 50000 # Replay buffer size
EPISODES = 10000 # Number of episodes for training
LOAD_SAVED_MODEL = False
MODEL_PATH = ''

environment = 'Pong-v4'
env_folder = 'Pong'
# environment, training policy, target policy
env, policy, target = GetEnvAndLearner(name = environment, learner='dqn')
if LOAD_SAVED_MODEL:
    policy.load_state_dict(torch.load(MODEL_PATH))
    target.load_state_dict(policy.state_dict())
target.eval()
renv = deepcopy(env)
loss_fn = nn.MSELoss()
optimizer = optim.Adam(policy.parameters(), lr=LR)

# Memory for Experience Replay
memory = ReplayMemory(MEMORY_BUFFER)

glob_frame = 0

def get_epsilon():
    # Linear Annealing
    return EPSILON_END + (EXPLORATION_FRAMES-glob_frame)*(EPSILON_START-EPSILON_END)/EXPLORATION_FRAMES \
        if glob_frame < EXPLORATION_FRAMES else EPSILON_END

def select_action(state, act_dim, eps=None):    
    if eps==None:
        eps = get_epsilon()
    # Epsilon-greedy exploration
    if np.random.uniform() < eps:
        return np.random.choice(act_dim)
    else:
        with torch.no_grad():
            q_sa = policy(torch.tensor(state, device=device))
        return torch.argmax(q_sa[0]).item()

def optimize_policy(samples):
    states, actions, rewards, next_states, terminals = zip(*samples)
    states = torch.tensor(np.vstack(states), device=device)
    actions = torch.tensor(np.vstack(actions), device=device)
    rewards = torch.tensor(np.vstack(rewards), device=device)
    next_states = torch.tensor(np.vstack(next_states), device=device)
    terminals = torch.tensor(np.vstack(terminals), device=device)

    q_sa = policy(states).gather(1, actions).squeeze()      
    q_nsa_max = target(next_states).max(1).values
    rewards = rewards.squeeze()
    terminals = terminals.squeeze()
    q_sa_target = rewards + GAMMA * q_nsa_max * (1 - terminals.int())

    # Optimize on the TD loss
    loss = loss_fn(q_sa, q_sa_target)
    optimizer.zero_grad()
    loss.backward()
    # torch.nn.utils.clip_grad_norm_(policy.parameters(), 10)
    optimizer.step()            

def validate_policy():    
    renv.reset()
    done = False
    valid_reward = 0
    if RENDER:
        cv2.namedWindow(environment, cv2.WINDOW_NORMAL)
    # cv2.resizeWindow(environment, 300, 300)
    while not done:       
        state = renv.get_state()
        if RENDER:
            rgb = renv.render()
            cv2.imshow(environment, rgb)
            cv2.waitKey(10)
        action = select_action(state, renv.act_dim, EPSILON_END)
        _, reward, done = renv.step(action)
        valid_reward+=reward
    return valid_reward

max_possible_reward = 21
reward_increment = max_possible_reward/10
max_valid_reward = -21
reward_history = []
max_reward_target = max_valid_reward + reward_increment
train_reward_history = []
valid_reward_history = []
recent_train_reward = deque(maxlen=100)
recent_valid_history = deque(maxlen=100)

for episode in range(EPISODES):
    # Default max episode steps is defined in Gym environments
    done = False
    episode_reward = 0
    while not done:       
        state = env.get_state()
        action = select_action(state, env.act_dim)
        next_state, reward, done = env.step(action)  
        episode_reward+=reward      
        glob_frame+=1

        memory.push((state, action, reward, next_state, done))
        if memory.length()<MEMORY_BUFFER*0.008:
            continue
        else:
            optimize_policy(memory.sample(BATCH_SIZE))

        if glob_frame%UPDATE_INTERVAL==0:
            target.load_state_dict(policy.state_dict())

    train_reward_history.append(episode_reward)
    recent_train_reward.append(episode_reward)

    # if max_valid_reward > max_possible_reward*0.98:
    #     RENDER = True
    valid_reward = validate_policy()    
    max_valid_reward = max(valid_reward,max_valid_reward)
    valid_reward_history.append(valid_reward)
    recent_valid_history.append(valid_reward)

    # Save model when there is a performance improvement
    if max_valid_reward>max_reward_target:
        max_reward_target = min(max_possible_reward, max(max_reward_target,max_valid_reward)+reward_increment)-1        
        print('Episode: ', episode, ' | Max Validation Reward: ', max_valid_reward, ' | Epsilon: ', get_epsilon())
        torch.save(policy.state_dict(), 'Checkpoints/'+env_folder+'/'+environment+'(dqn'+str(int(max_valid_reward))+')'+'.dqn')
        if max_valid_reward==max_possible_reward:
            print('Best Model Achieved !!!')
            break

    print('Episode: ', episode, ' | Epsilon: ', get_epsilon(), ' | Train Reward:', episode_reward, ' | Avg Train Reward:', np.mean(recent_train_reward), ' | Valid Reward:', valid_reward, ' | Avg Valid Reward:', np.mean(recent_valid_history))

# RENDER = True
# validate_policy()

reward_history = np.array(train_reward_history)
smooth_reward_history = np.convolve(reward_history, np.ones(20)/20, mode='same')
import matplotlib.pyplot as plt
plt.plot(reward_history, label='Reward')
plt.plot(smooth_reward_history, label='Smooth Reward')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.legend(loc='upper left')
# plt.show()
plt.savefig('res_train_dqn.png')

reward_history = np.array(valid_reward_history)
smooth_reward_history = np.convolve(reward_history, np.ones(20)/20, mode='same')
import matplotlib.pyplot as plt
plt.plot(reward_history, label='Reward')
plt.plot(smooth_reward_history, label='Smooth Reward')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.legend(loc='upper left')
# plt.show()
plt.savefig('res_valid_dqn.png')