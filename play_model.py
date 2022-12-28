import numpy as np
import torch
from get_env_and_learner import GetEnvAndLearner
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

MODEL_PATH = 'Benchmarks/PongDeterministic-v4(dddqn21).dqn'
environment = 'PongDeterministic-v4'
env, policy, _ = GetEnvAndLearner(name = environment, learner='dddqn')
policy.load_state_dict(torch.load(MODEL_PATH))

def select_action(state, act_dim, eps=None):    
    # Epsilon-greedy exploration
    if np.random.uniform() < 0.05:
        return np.random.choice(act_dim)
    else:
        with torch.no_grad():
            q_sa = policy(torch.tensor(state, device=device))
        return torch.argmax(q_sa[0]).item()

observation = env.reset()
done = False
while not done:
    state = env.get_state()
    action = select_action(state, env.act_dim)
    next_state, reward, done = env.step(action)  
    env.render()
    if done:
        observation = env.reset()