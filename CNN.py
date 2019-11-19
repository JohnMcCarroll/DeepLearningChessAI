import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(14, 14, 3, 1, 1)
        self.conv2 = nn.Conv2d(14, 14, 3, 1, 1)

        self.conv3 = nn.Conv2d(14, 14, 3, 1, 1)
        self.conv4 = nn.Conv2d(14, 14, 3, 1, 1)

        self.conv5 = nn.Conv2d(28, 6, 3, 1, 1)
        self.out = nn.Linear(8*8*6, 1)

    def forward(self, t):
        # imput layer
        # t = t

        # conv1 layer
        t = self.conv1(t)
        t = F.leaky_relu(t, 0.05)

        # conv2 layer
        t = self.conv2(t)
        tSkip = F.leaky_relu(t, 0.05)


        # conv3 layer
        t = self.conv3(tSkip)
        t = F.leaky_relu(t, 0.05)

        # conv4 layer
        t = self.conv4(t)
        t = F.leaky_relu(t, 0.05)

        # Skip Connection
        t = torch.cat((t, tSkip), dim=1)

        # conv5 layer
        t = self.conv5(t)
        t = F.leaky_relu(t, 0.05)

        # linear output layer
        t = t.reshape(-1, 8*8*6)
        t = self.out(t)
        t = F.leaky_relu(t, 0.05)

        return t

# import torch
# net = CNN()
# output = net(torch.rand([1,14,8,8]))
# print(output.size())