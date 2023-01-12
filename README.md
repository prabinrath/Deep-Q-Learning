# Deep-Q-Learning
 Reinforcement Learning with PyTorch and OpenAI-Gym. This repository has implementation for [Deep-Q-Learning Algorithm](https://arxiv.org/abs/1312.5602) and [Dueling Double Deep-Q-Learning Algorithm](https://arxiv.org/abs/1511.06581). The CNNs have batch normalization layers to expedite the training process. With the current settings, training completes in ~4 hours (Ryzen7 + RTX3060) for both DQN and DDDQN. Training time is determined by the specified exploration frames. Maximum runtime memory requirement is ~8.5 GB (100k Memory Buffer) for training. Runtime memory requirement depends on the size of experience replay buffer. <br/>

 `main` branch has a preliminary implementation for `Cartpole` environment.

![Python 3](https://img.shields.io/badge/Python-3-yellow.svg)
![Pytorch](https://img.shields.io/badge/Pytorch-1.13-orange.svg)
![GYM](https://img.shields.io/badge/GYM-0.19-turquoise.svg)

## Results
Before Training (Left) vs After Training (Right)
<p align="center">
  <img width="460" height="300" src="Results/atari_not_trained.gif">
  <img width="460" height="300" src="Results/atari_trained.gif">
</p>

<!-- Reward Graph
<p align="center">
  <img width="460" height="300" src="Results/res_valid_dqn.png">
  <img width="460" height="300" src="Results/res_valid_dddqn.png">
</p> -->

## Install Dependencies
```bash
pip install -r requirements.txt
```

## Train 
Create folder for checkpoints
```bash
mkdir -p Checkpoints/Pong
```
Run DQN
```bash
python deep_q_learning.py
```
Run DDDQN
```bash
python duel_double_deep_q_learning.py
```

## Test
Set `MODEL_PATH` and `environment` parameters in `play_model.py`
```bash
python play_model.py
```

## Notes
- DQN is tricky and it takes numerous attempts and patience to get things working from scratch. For quicker results one can start with OpenAI baselines. However, trying from scratch provides deeper insights about parameters and their effect on the algorithm's performance.
- Pytorch specific: The target needs to be detached from the computation graph or the target Q values need to be calculated with `no_grad` scope during model optimization. This ensures that the optimizer does not update the weights of the target network during backpropagation.
- Be aware of the data types of tensors. Replay memory should have `uint8` type but that needs to be converted to `float32` while training. Reward and terminal tensors should also be `float32` type.
- Gradient clipping and reward clipping both are extremely important for stability.
- Batch normalization is **not** a good choice for DQN and leads to increased training time.
- Huber loss is an essential alternative for MSE loss. Simpler problems such as Cartpole and Pong where the planning horizon is short and the focus is more on immediate rewards, MSE loss performs fairly well. 
- Parameters such as GAMMA, BATCH_SIZE, LEARNING_RATE, EXPLORATION_FRAMES and MEMORY_BUFFER determine the training time. Small deviations in these parameters should not affect the stability drastically.
- EPSILON scheduling is important during initial training phase. In the later phase of training most of the learning happens with EPSILON_END.
- Gym Deterministic environments are the ones Deepmind folks used for their [2013 paper](https://arxiv.org/pdf/1312.5602.pdf)
- Model optimization happens once in POLICY_UPDATE_INTERVAL steps and target update happens once in TARGET_UPDATE_INTERVAL steps. For Deterministic environments each step skips 4 frames. So if POLICY_UPDATE_INTERVAL=4 and TARGET_UPDATE_INTERVAL=100 then backpropagation happens once in 16 game frames and target update happens once in 400 game frames.
- Understanding the pre-processing is confusing. I would recommend [this](https://danieltakeshi.github.io/2016/11/25/frame-skipping-and-preprocessing-for-deep-q-networks-on-atari-2600-games/) article to get a clear idea about how the Atari environments work. 
- There are two terminal flags. One is true every time the agent loses a life and the other is true only at the end of an episode. Replay memory should be filled with the one indicating lives lost.
- Random FIRE actions at the start of an episode avoids learning a suboptimal policy.
- Start training when memory buffer has some good number of samples. This avoids learning a suboptimal policy.
- Don't give up. Deep RL is frustrating to train but it is also an engineering marvel.