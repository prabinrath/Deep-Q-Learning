import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from get_env_and_learner import GetEnvAndLearner
from dqn_utils import BatchReplayMemory
from copy import deepcopy
import cv2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Constant Parameters
RENDER = False
GAMMA = 0.9 # Discount factor
UPDATE_INTERVAL = 100 # Interval for target update
LR = 0.00025 # AdamW learning rate
EPSILON_START = 0.9 # Annealing start
EPSILON_END = 0.05 # Annealing end
EXPLORATION_FRAMES = 1000000 # Annealing frames
BATCH_SIZE = 64 # Sampling size from memory
MEMORY_BUFFER = 50000 # Replay buffer size
EPISODES = 1000 # Number of episodes for training

environment = 'Pong-v4'
# environment, training policy, target policy
env, policy, target = GetEnvAndLearner(name = environment)
renv = deepcopy(env)
loss_fn = nn.SmoothL1Loss()
optimizer = optim.AdamW(policy.parameters(), lr=LR, amsgrad=True)

# Memory for Experience Replay
memory = BatchReplayMemory(env.n_buffer, MEMORY_BUFFER)
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
            policy.eval()
            q_sa = policy(torch.tensor(state, device=device))
        return torch.argmax(q_sa[0]).item()

def optimize_policy(samples):
    states, actions, rewards, next_states, terminals = zip(*samples)
    states = torch.tensor(np.vstack(states), device=device)
    actions = torch.tensor(np.vstack(actions), device=device)
    next_states = torch.tensor(np.vstack(next_states), device=device)
    policy.train()
    q_sa = policy(states).gather(1, actions).squeeze()
    with torch.no_grad():
        target.eval()
        q_nsa_max = target(next_states).max(1).values
    q_sa_target = [rewards[j]+GAMMA*q_nsa_max[j].item()*(1-terminals[j]) for j in range(len(rewards))]
    q_sa_target = torch.tensor(q_sa_target, device=device)
    # Optimize on the TD loss
    loss = loss_fn(q_sa, q_sa_target)
    optimizer.zero_grad()
    loss.backward()
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
for episode in range(EPISODES):
    if max_valid_reward > max_possible_reward*0.98:
        RENDER = True
    valid_reward = validate_policy()
    print('Episode: ', episode, ' | Validation Reward: ', valid_reward, ' | Epsilon: ', get_epsilon())
    max_valid_reward = max(valid_reward,max_valid_reward)
    reward_history.append(valid_reward)

    # Save model when there is a performance improvement
    if max_valid_reward>max_reward_target:
        max_reward_target = min(max_possible_reward, max(max_reward_target,max_valid_reward)+reward_increment)-1        
        print('Episode: ', episode, ' | Max Validation Reward: ', max_valid_reward, ' | Epsilon: ', get_epsilon())
        torch.save(policy.state_dict(), 'Checkpoints/'+environment+'/'+str(int(max_valid_reward))+'.dqn')
        if max_valid_reward==max_possible_reward:
            print('Best Model Achieved !!!')
            break
    
    # Default max episode steps is defined in Gym environments
    done = False
    while not done:       
        state = env.get_state()
        action = select_action(state, env.act_dim)
        next_state, reward, done = env.step(action)        
        glob_frame+=1

        memory.push((state[:,env.n_buffer-1,:,:], action, reward, next_state[:,env.n_buffer-1,:,:], done))
        if memory.length()<BATCH_SIZE:
            continue
        else:
            optimize_policy(memory.sample(BATCH_SIZE))

        if glob_frame%UPDATE_INTERVAL==0:
            target.load_state_dict(policy.state_dict())

RENDER = True
validate_policy()

reward_history = np.array(reward_history)
smooth_reward_history = np.convolve(reward_history, np.ones(20)/20, mode='same')
import matplotlib.pyplot as plt
plt.plot(reward_history, label='Real')
plt.plot(smooth_reward_history, label='Smooth')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.legend(loc='upper left')
plt.show()