import torch
from torch import nn

class ICE_NET(nn.Module):
    def __init__(self, num_classes = 10):
        super().__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 16, kernel_size = 3, stride = 1) # input -> 1x28x28, output -> 16x26x26
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 5, stride = 2) # input -> 16x26x26, output -> 32x11x11
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 5, stride = 1) # input -> 32x11x11, output -> 64x7x7 (3136)
        self.bn3 = nn.BatchNorm2d(64)

        self.relu = nn.ReLU(inplace = True)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Linear(3136, 512)
        self.fc2 = nn.Linear(512, self.num_classes)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))

        # x = x.view(x.size(0), -1)
        x = self.avg_pool(x)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return x