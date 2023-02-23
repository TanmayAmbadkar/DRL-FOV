import torch
import torch.nn as nn
import torch.nn.functional as F
from residual_block import ResidualBlock
from conv_block import ConvBlock

class FOVSelectionNet(nn.Module):
    
    def __init__(self, n_frames=1, residual_depth=5):
        
        super(FOVSelectionNet, self).__init__()
        
        self.n_frames = n_frames
        self.conv_blocks = [ConvBlock(residual_depth)]
        self.conv_blocks = nn.Sequential(*self.conv_blocks)
        self.fc1 = nn.Linear(1792*self.n_frames, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 9)
        
    def forward(self, X):
        
        final_outputs = []
        for i in range(len(X)):
            
            x = X[0]
            outputs = [None]*self.n_frames
            for i in range(self.n_frames):
                outputs[i] = self.conv_blocks[i](x[i:i+1])
        
            final_outputs.append(torch.stack(outputs, 1).reshape(1,-1))
        
        out = F.relu(self.fc1(torch.stack(final_outputs, 1)[0]))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        
        return out
        