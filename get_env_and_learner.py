from Envs.cartpole_env import CartPoleEnv
from Envs.atari_env import AtariEnv
from Learners.cartpole_learner import CartPoleLearner
from Learners.atari_learner import AtariLearner
from Learners.atari_learner_adv import AtariLearnerAdv
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def GetEnvAndLearner(name='CartPole-v1'):
    if name == 'CartPole-v1':
        env = CartPoleEnv(name)
        policy = CartPoleLearner(env.obs_dim, env.act_dim).to(device)
        target = CartPoleLearner(env.obs_dim, env.act_dim).to(device)
        target.load_state_dict(policy.state_dict())
        return env, policy, target
    elif name == 'PongDeterministic-v4' or name == 'Pong-v4':
        env = AtariEnv(name)
        policy = AtariLearnerAdv(env.n_buffer, env.act_dim).double().to(device)
        target = AtariLearnerAdv(env.n_buffer, env.act_dim).double().to(device)
        target.load_state_dict(policy.state_dict())
        return env, policy, target
    else:
        raise Exception('Environment Not Defined')
        