import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from get_env_and_learner import GetEnvAndLearner
from dqn_utils import ReplayMemory
from copy import deepcopy
from collections import deque
import matplotlib.pyplot as plt
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Constant Parameters
GAMMA = 0.99 # Discount factor
UPDATE_INTERVAL = 1000 # Interval for target update
LR = 0.00025 # Adam learning rate
EPSILON_START = 1 # Annealing start
EPSILON_END = 0.05 # Annealing end
EXPLORATION_FRAMES = 1000000 # Annealing frames
BATCH_SIZE = 64 # Sampling size from memory
MEMORY_BUFFER = 1000000 # Replay buffer size
EPISODES = 20000 # Number of episodes for training

environment = 'BreakoutDeterministic-v4'
env_folder = 'Breakout'
# environment, training policy, target policy
env, policy, target = GetEnvAndLearner(name = environment, learner='dqn')
target.eval()
renv = deepcopy(env)
loss_fn = nn.SmoothL1Loss()
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
            q_sa = policy(torch.tensor(state, device=device, dtype=torch.float))
        return torch.argmax(q_sa[0]).item()

def optimize_policy(samples):
    states, actions, rewards, next_states, terminals = zip(*samples)
    states = torch.tensor(np.vstack(states), device=device, dtype=torch.float)
    actions = torch.tensor(np.vstack(actions), device=device)
    rewards = torch.tensor(np.vstack(rewards), device=device, dtype=torch.float)
    next_states = torch.tensor(np.vstack(next_states), device=device, dtype=torch.float)
    terminals = torch.tensor(np.vstack(terminals), device=device, dtype=torch.float)

    q_sa = policy(states).gather(1, actions).squeeze()     
    with torch.no_grad(): 
        q_nsa_max = target(next_states).max(1).values
        q_sa_target = rewards.squeeze() + GAMMA * q_nsa_max * (1.0 - terminals.squeeze())

    # Optimize on the TD loss
    loss = loss_fn(q_sa, q_sa_target)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 10)
    optimizer.step()            

def validate_policy():    
    renv.reset()
    done = False
    valid_reward = 0
    while not done:       
        state = renv.get_state()
        action = select_action(state, renv.act_dim, EPSILON_END)
        _, reward, done, _ = renv.step(action)
        valid_reward+=reward
    return valid_reward

def save_stats(train_reward_history, valid_reward_history, padding=10):
    reward_history = np.array(train_reward_history)
    smooth_reward_history = np.convolve(reward_history, np.ones(padding*2)/(padding*2), mode='valid')
    plt.plot(reward_history, label='Reward')
    plt.plot(smooth_reward_history, label='Smooth Reward')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend(loc='upper left')
    plt.title('Deep Q-Learning')
    # plt.show()
    plt.savefig('res_train_dqn.png')
    plt.clf()
    reward_history = np.array(valid_reward_history)
    smooth_reward_history = np.convolve(reward_history, np.ones(padding*2)/(padding*2), mode='valid')
    plt.plot(reward_history, label='Reward')
    plt.plot(smooth_reward_history, label='Smooth Reward')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend(loc='upper left')
    plt.title('Deep Q-Learning')
    # plt.show()
    plt.savefig('res_valid_dqn.png')
    plt.clf()

max_possible_reward = 300
reward_increment = max_possible_reward/50
max_valid_reward = 0
max_reward_target = max_valid_reward + reward_increment
train_reward_history = []
valid_reward_history = []
recent_train_reward = deque(maxlen=100)
recent_valid_reward = deque(maxlen=100)

for episode in range(EPISODES):
    # Default max episode steps is defined in Gym environments
    done = False
    episode_reward = 0
    while not done:       
        state = env.get_state()
        action = select_action(state, env.act_dim)
        next_state, reward, done, terminal_life_lost = env.step(action)  
        episode_reward+=reward      
        glob_frame+=1

        memory.push((state, action, reward, next_state, float(terminal_life_lost)))
        if memory.length()<MEMORY_BUFFER*0.05:
            glob_frame-=1
            continue
        else:
            optimize_policy(memory.sample(BATCH_SIZE))

        if glob_frame%UPDATE_INTERVAL==0:
            target.load_state_dict(policy.state_dict())

    train_reward_history.append(episode_reward)
    recent_train_reward.append(episode_reward)
    avg_train_reward = round(np.mean(recent_train_reward),3)

    valid_reward = validate_policy()    
    max_valid_reward = max(valid_reward,max_valid_reward)
    valid_reward_history.append(valid_reward)
    recent_valid_reward.append(valid_reward)
    avg_valid_reward = round(np.mean(recent_valid_reward),3)

    # Save model when there is a performance improvement
    if max_valid_reward>=max_reward_target:
        max_reward_target = min(max_possible_reward, max(max_reward_target,max_valid_reward)+reward_increment)-1        
        print('Episode: ', episode, ' | Max Validation Reward: ', max_valid_reward, ' | Epsilon: ', get_epsilon())
        save_stats(train_reward_history, valid_reward_history)
        torch.save(policy.state_dict(), 'Checkpoints/'+env_folder+'/'+environment+'(dqn'+str(int(max_valid_reward))+')'+'.dqn')
        if max_valid_reward>=max_possible_reward:
            print('Best Model Achieved !!!')
            break

    print('Episode: ', episode, ' | Epsilon: ', round(get_epsilon(),3) , ' | Train Reward:', episode_reward, ' | Avg Train Reward:', avg_train_reward, ' | Valid Reward:', valid_reward, ' | Avg Valid Reward:', avg_valid_reward)

save_stats(train_reward_history, valid_reward_history)