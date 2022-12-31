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

Reward Graph
<p align="center">
  <img width="460" height="300" src="Results/res_valid_dqn.png">
  <img width="460" height="300" src="Results/res_valid_dddqn.png">
</p>

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