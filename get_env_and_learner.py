from Envs.cartpole_env import CartPoleEnv
from Envs.atari_env import PongEnv, BreakoutEnv
from Learners.cartpole_learner import CartPoleLearner
from Learners.atari_learner import AtariLearner
from Learners.atari_learner_adv import AtariLearnerAdv
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def GetEnvAndLearner(name='CartPole-v1', learner='dqn'):
    print(name) 
    if name == 'CartPole-v1':
        env = CartPoleEnv(name)
        if learner=='dqn':
            print('DQN')
            policy = CartPoleLearner(env.obs_dim, env.act_dim).to(device)
            target = CartPoleLearner(env.obs_dim, env.act_dim).to(device)
        else:
            raise Exception('Learner Not Defined')
        target.load_state_dict(policy.state_dict())
        return env, policy, target
    elif name == 'PongDeterministic-v4':        
        env = PongEnv(name)
        if learner=='dqn':
            print('DQN')
            policy = AtariLearner(env.n_buffer, env.act_dim).to(device)
            target = AtariLearner(env.n_buffer, env.act_dim).to(device)
        elif learner=='dddqn':
            print('Duel Double DQN')
            policy = AtariLearnerAdv(env.n_buffer, env.act_dim).to(device)
            target = AtariLearnerAdv(env.n_buffer, env.act_dim).to(device)
        else:
            raise Exception('Learner Not Defined')
        target.load_state_dict(policy.state_dict())
        return env, policy, target
    elif name == 'BreakoutNoFrameskip-v4':
        env = BreakoutEnv(name)
        if learner=='dqn':
            print('DQN')
            policy = AtariLearner(env.n_buffer, env.act_dim).to(device)
            target = AtariLearner(env.n_buffer, env.act_dim).to(device)
        elif learner=='dddqn':
            print('Duel Double DQN')
            policy = AtariLearnerAdv(env.n_buffer, env.act_dim).to(device)
            target = AtariLearnerAdv(env.n_buffer, env.act_dim).to(device)
        else:
            raise Exception('Learner Not Defined')
        target.load_state_dict(policy.state_dict())
        return env, policy, target
    else:
        raise Exception('Environment Not Defined')
        