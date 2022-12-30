
import torch.nn as nn
import torch.nn.functional as F

class AtariLearnerAdv(nn.Module):
    def __init__(self, in_channels, act_dim):
        super().__init__()

        # CNN Layers
        self.cnn1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4)
        h, w = self.calc_conv2d_output_dim(in_dim = (84,84), kernel_size=(8,8), stride=(4,4))
        self.bn1 = nn.BatchNorm2d(32)
        self.cnn2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        h, w = self.calc_conv2d_output_dim(in_dim = (h,w), kernel_size=(4,4), stride=(2,2))
        self.bn2 = nn.BatchNorm2d(64)
        self.cnn3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        h, w = self.calc_conv2d_output_dim(in_dim = (h,w), kernel_size=(3,3), stride=(1,1))
        self.bn3 = nn.BatchNorm2d(64)

        # Value Estimator
        self.val1 = nn.Linear(h*w*64, 512)
        self.val2 = nn.Linear(512, 1)

        # Advantage Estimator
        self.adv1 = nn.Linear(h*w*64, 512)
        self.adv2 = nn.Linear(512, act_dim)

    def calc_conv2d_output_dim(self, in_dim, kernel_size, padding=(0,0), dialation=(1,1), stride=(1,1)):
        h = (in_dim[0]+2*padding[0]-dialation[0]*(kernel_size[0]-1)-1)//stride[0]+1
        w = (in_dim[1]+2*padding[1]-dialation[1]*(kernel_size[1]-1)-1)//stride[1]+1
        return h, w

    def forward(self, x_):
        x = x_/255.0
        x = F.relu(self.bn1(self.cnn1(x)))
        x = F.relu(self.bn2(self.cnn2(x)))
        x = F.relu(self.bn3(self.cnn3(x)))
        x = x.view(x.size(0),-1)

        v = F.relu(self.val1(x))
        v = self.val2(v)

        a = F.relu(self.adv1(x))
        a = self.adv2(a)

        q = v + (a - a.mean(dim=1, keepdim=True))
        return q
