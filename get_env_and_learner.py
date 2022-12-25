from Envs.cartpole_env import CartPoleEnv
from Envs.atari_env import AtariEnv
from Learners.cartpole_learner import CartPoleLearner
from Learners.atari_learner import AtariLearner
from Learners.atari_learner_adv import AtariLearnerAdv
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def GetEnvAndLearner(name='CartPole-v1', learner='dqn'):
    if name == 'CartPole-v1' and learner=='dqn':
        print('DQN')
        env = CartPoleEnv(name)
        policy = CartPoleLearner(env.obs_dim, env.act_dim).to(device)
        target = CartPoleLearner(env.obs_dim, env.act_dim).to(device)
        target.load_state_dict(policy.state_dict())
        return env, policy, target
    elif (name == 'PongDeterministic-v4' or name == 'Pong-v4') and learner=='dqn':
        print('DQN')
        env = AtariEnv(name)
        policy = AtariLearner(env.n_buffer, env.act_dim).double().to(device)
        target = AtariLearner(env.n_buffer, env.act_dim).double().to(device)
        target.load_state_dict(policy.state_dict())
        return env, policy, target
    elif (name == 'PongDeterministic-v4' or name == 'Pong-v4') and learner=='dddqn':
        print('Duel Double DQN')
        env = AtariEnv(name)
        policy = AtariLearnerAdv(env.n_buffer, env.act_dim).double().to(device)
        target = AtariLearnerAdv(env.n_buffer, env.act_dim).double().to(device)
        target.load_state_dict(policy.state_dict())
        return env, policy, target
    else:
        raise Exception('Environment and/or Learner Not Defined')
        