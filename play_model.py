import numpy as np
import torch
import cv2  
from get_env_and_learner import GetEnvAndLearner
import random
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

env_folder = 'Breakout'
learner = 'dqn'
benchmark = '414'
environment = env_folder+'Deterministic-v4'
MODEL_PATH = 'Benchmarks/'+env_folder+'Deterministic-v4('+learner+benchmark+').dqn'
env, policy, _ = GetEnvAndLearner(name = environment, learner=learner)
policy.load_state_dict(torch.load(MODEL_PATH))

def select_action(state, act_dim, eps=None):    
    # Epsilon-greedy exploration
    if np.random.uniform() < 0.01:
        return np.random.choice(act_dim)
    else:
        with torch.no_grad():
            q_sa = policy(torch.tensor(state, device=device))
        return torch.argmax(q_sa[0]).item()

done = False
img_frames = []
cv2.namedWindow('Agent', cv2.WINDOW_NORMAL)
net_reward = 0

# Random FIRE to start episode
for _ in range(random.randint(1,10)):
    _ = env.get_state()
    _, _, _, _ = env.step(1)  

while not done:
    state = env.get_state()
    action = select_action(state, env.act_dim)
    next_state, reward, done, _ = env.step(action)  
    net_reward+=reward
    img = env.render()
    img_frames.append((img*255).astype(np.uint8))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imshow('Agent', img)
    cv2.waitKey(10)    
    if done:
        observation = env.reset()

print(net_reward)

GENERATE_GIF = False
if GENERATE_GIF:
    import imageio      
    print("Saving GIF file")
    imageio.mimsave("atari.gif", img_frames, format='GIF', fps=30)
