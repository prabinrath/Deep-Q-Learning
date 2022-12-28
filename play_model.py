import numpy as np
import torch
import cv2  
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
img_frames = []
cv2.namedWindow('Agent', cv2.WINDOW_NORMAL)
while not done:
    state = env.get_state()
    action = select_action(state, env.act_dim)
    next_state, reward, done = env.step(action)  
    img = env.render()
    img_frames.append((img*255).astype(np.uint8))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imshow('Agent', img)
    cv2.waitKey(5)    
    if done:
        observation = env.reset()

GENERATE_GIF = False
if GENERATE_GIF:
    import imageio      
    print("Saving GIF file")
    imageio.mimsave("atari.gif", img_frames, format='GIF', fps=30)
