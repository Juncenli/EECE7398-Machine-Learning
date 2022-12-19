import torch
import torch.nn as nn
import torch.nn.functional as F

class NerualNetwork(torch.nn.Module):
    def __init__(self):
        super(NerualNetwork, self).__init__()
        self.conv11 = nn.Conv2d(3, 32, 5)  # (28x28x32)
        self.conv12 = nn.Conv2d(32, 32, 1)
        self.conv13 = nn.Conv2d(32, 32, 1)
        self.mp1 = nn.MaxPool2d(3, stride=1)  # (26x26x32)
        # self.mp11 = nn.MaxPool2d(4, stride =2)
        self.bn2d1 = nn.BatchNorm2d(32)
        self.conv21 = nn.Conv2d(32, 64, 3)  # (24x24x64)
        self.conv22 = nn.Conv2d(64, 64, 1)
        self.conv23 = nn.Conv2d(64, 64, 1)
        self.mp2 = nn.MaxPool2d(4, stride=2)  # (10x10x16)
        self.bn2d2 = nn.BatchNorm2d(64)
        self.conv31 = nn.Conv2d(64, 128, 3)  # (8x8x128)
        self.conv32 = nn.Conv2d(128, 128, 1)
        self.conv33 = nn.Conv2d(128, 128, 1)
        self.bn2d3 = nn.BatchNorm2d(128)
        self.mp3 = nn.MaxPool2d(3, stride=2)  # (5x5z128)

        # self.conv31 = nn.Conv2d(16,16,3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(4 * 4 * 128, 1024)  # 5x5x16 from image mp2
        self.fc1_bn = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 10)
        self.fc2_bn = nn.BatchNorm1d(10)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.conv11(x))
        self.conv1_output = x
        x = self.bn2d1(x)
        x = self.mp1(x)
        x = F.relu(self.conv21(x))
        x = self.bn2d2(x)
        x = self.mp2(x)
        x = F.relu(self.conv31(x))
        x = self.bn2d3(x)
        x = self.mp3(x)
        # x = self.conv31(x)
        x = x.view(-1, 4 * 4 * 128)
        x = self.fc1_bn(F.relu(self.fc1(x)))
        x = F.softmax(self.fc2(x), dim=-1)
        return x
