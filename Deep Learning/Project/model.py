import torch
import torch.nn as nn

class RobustModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.in_dim = 28 * 28 * 3
        self.out_dim = 10
        
        self.conv = nn.Sequential( # 28*28*3
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 14*14*32

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 7*7*64
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 3*3*128
        )
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(),
            nn.Linear(in_features=3*3*128, out_features=self.out_dim)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x