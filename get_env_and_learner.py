from Envs.cartpole_env import CartPoleEnv
from Learners.cartpole_learner import CartPoleLearner
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def GetEnvAndLearner(name='CartPole-v1'):
    if name == 'CartPole-v1':
        env = CartPoleEnv(name)
        policy = CartPoleLearner(env.obs_dim, env.act_dim).to(device)
        target = CartPoleLearner(env.obs_dim, env.act_dim).to(device)
        target.load_state_dict(policy.state_dict())
        return env, policy, target
    else:
        raise Exception('Environment not defined')