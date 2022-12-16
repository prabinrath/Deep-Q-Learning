
import torch.nn as nn
import torch.nn.functional as F

class CartPoleLearner(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.fcc1 = nn.Linear(obs_dim, 128, bias=True)
        self.fcc2 = nn.Linear(128, 64, bias=True)
        self.fcc3 = nn.Linear(64, act_dim, bias=True)

    def forward(self, x):
        x = F.relu(self.fcc1(x))
        x = F.relu(self.fcc2(x))
        return self.fcc3(x)        
