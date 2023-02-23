import torch
import torch.nn as nn
import torch.nn.functional as F
from residual_block import ResidualBlock

class ConvBlock(nn.Module):

    def __init__(self, residual_depth):
        super(ConvBlock, self).__init__()
        self.cnn1 = nn.Conv2d(
            in_channels = 1, 
            out_channels = 64, 
            kernel_size = 9, 
            stride=1, 
            padding=4, 
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.prelu1 = nn.PReLU()
        residual_blocks = [ResidualBlock(64, 64)]*residual_depth
        self.residual_layers = nn.Sequential(*residual_blocks)
        self.cnn2 = nn.Conv2d(
                in_channels = 64, 
                out_channels = 32, 
                kernel_size = 9, 
                stride=1, 
                padding=1, 
            )
        self.bn2 = nn.BatchNorm2d(32)
        self.prelu2 = nn.PReLU()
        self.cnn3 = n.Conv2d(
                in_channels = 32, 
                out_channels = 16, 
                kernel_size = 9, 
                stride=1, 
                padding=1, 
            )
        self.bn3 = nn.BatchNorm2d(16)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.prelu1(self.bn1(self.cnn1(x)))
        x = self.residual_layers(x)
        x = torch.max_pool2d(self.prelu2(self.bn2(self.cnn2(x))), 9, 5)
        x = torch.max_pool2d(self.prelu3(self.bn3(self.cnn3(x))), 9, 5)

        return self.flatten(x)