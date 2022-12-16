import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from get_env_and_learner import GetEnvAndLearner
from collections import namedtuple
from dqn_utils import ReplayMemory
from tqdm import tqdm
import cv2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Constant Parameters
RENDER = False
GAMMA = 0.9 # Discount factor
UPDATE_INTERVAL = 100 # Interval for target update
LR = 0.01 # AdamW learning rate
EPSILON_START = 0.9 # Annealing start
EPSILON_END = 0.05 # Annealing end
EXPLORATION_FRAMES = 1000 # Annealing frames
BATCH_SIZE = 32 # Sampling size from memory
MEMORY_BUFFER = 2000 # Replay buffer size
EPISODES = 200 # Number of episodes for training

# environment, training policy, target policy
environment = 'CartPole-v1'
env, policy, target = GetEnvAndLearner(name = environment)
loss_fn = nn.SmoothL1Loss()
optimizer = optim.AdamW(policy.parameters(), lr=LR, amsgrad=True)

# Named tuple for storing transitions
Transition = namedtuple('Transition', 'state action reward next_state terminal')

# Memory for Experience Replay
memory = ReplayMemory(MEMORY_BUFFER)
glob_frame = 0

def get_epsilon():
    # Linear Annealing
    return EPSILON_END + (EXPLORATION_FRAMES-glob_frame)*(EPSILON_START-EPSILON_END)/EXPLORATION_FRAMES \
        if glob_frame < EXPLORATION_FRAMES else EPSILON_END

def select_action(state, act_dim):    
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
    loss = loss_fn(q_sa, q_sa_target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()            

max_possible_reward = 500
reward_increment = max_possible_reward/10
max_episode_reward = 0
max_reward_target = reward_increment
for episode in tqdm(range(EPISODES)):
    # Render when there is a performance improvement
    if max_episode_reward>max_reward_target:
        RENDER = True
        max_reward_target=min(max_possible_reward, max(max_reward_target,max_episode_reward)+reward_increment)-1
        print('Max Episode Reward: ', max_episode_reward, ' | Epsilon: ', get_epsilon())
        torch.save(policy.state_dict(), 'Checkpoints/'+environment+'/'+str(int(max_episode_reward))+'.dqn')
    
    episode_reward = 0
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
        episode_reward+=reward
        glob_frame+=1
        
        memory.push(Transition(state, action, reward, next_state, done))
        if memory.length()<BATCH_SIZE:
            continue
        else:
            optimize_policy(memory.sample(BATCH_SIZE))

        # if glob_frame%UPDATE_INTERVAL==0:
        #     target.load_state_dict(policy.state_dict())
        target_state_dict = target.state_dict()
        policy_state_dict = policy.state_dict()
        TAU = 0.005
        for key in policy_state_dict:
            target_state_dict[key] = policy_state_dict[key]*TAU + target_state_dict[key]*(1-TAU)
        target.load_state_dict(target_state_dict)

    max_episode_reward = max(episode_reward,max_episode_reward)
    RENDER = False
