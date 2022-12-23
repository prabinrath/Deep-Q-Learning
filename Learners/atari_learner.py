
import torch.nn as nn
import torch.nn.functional as F

class AtariLearner(nn.Module):
    def __init__(self, in_channels, act_dim):
        super().__init__()
        self.cnn1 = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=8, stride=4)
        h, w = self.calc_conv2d_output_dim(in_dim = (84,84), kernel_size=(8,8), stride=(4,4))
        self.bn1 = nn.BatchNorm2d(16)
        self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2)
        h, w = self.calc_conv2d_output_dim(in_dim = (h,w), kernel_size=(4,4), stride=(2,2))
        self.bn2 = nn.BatchNorm2d(32)
        self.fcc1 = nn.Linear(h*w*32, 256)
        self.fcc2 = nn.Linear(256, act_dim)

    def calc_conv2d_output_dim(self, in_dim, kernel_size, padding=(0,0), dialation=(1,1), stride=(1,1)):
        h = (in_dim[0]+2*padding[0]-dialation[0]*(kernel_size[0]-1)-1)//stride[0]+1
        w = (in_dim[1]+2*padding[1]-dialation[1]*(kernel_size[1]-1)-1)//stride[1]+1
        return h, w

    def forward(self, x):
        x = F.relu(self.bn1(self.cnn1(x)))
        x = F.relu(self.bn2(self.cnn2(x)))
        x = x.view(x.size(0),-1)
        x = F.relu(self.fcc1(x))
        return self.fcc2(x)
