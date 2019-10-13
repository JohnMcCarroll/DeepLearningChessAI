import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(14, 14, 3, 1, 1)
        self.conv2 = nn.Conv2d(14, 14, 3, 1, 1)
        self.conv3 = nn.Conv2d(14, 6, 3, 1, 1)
        self.out = nn.Linear(8*8*6, 1)

    def forward(self, t):
        # imput layer
        # t = t

        # conv1 layer
        t = self.conv1(t)
        t = F.leaky_relu(t, 0.05)

        # conv2 layer
        t = self.conv2(t)
        t = F.leaky_relu(t, 0.05)

        # conv3 layer
        t = self.conv3(t)
        t = F.leaky_relu(t, 0.05)

        # linear output layer
        t = t.reshape(-1, 8*8*6)
        t = self.out(t)
        t = F.leaky_relu(t, 0.05)

        return t
