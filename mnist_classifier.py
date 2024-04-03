import torch.nn as nn
import torch

class MNISTClassifier(nn.Module):
    
    def __init__(self):
        
        super().__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(4, 8, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.1)
        )
        
        self.ffn = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 10)
        )
        
    def forward(self, x):
        
        B = x.shape[0]
        
        after_cnn = self.cnn(x)
        
        output = self.ffn(after_cnn.view((B, -1)))
                
        return output
    
    def save_model(self, path):
        torch.save(self.state_dict(), path)
        print(f'Model saved to {path}')
    
    def load_model(self, path):
        self.load_state_dict(torch.load(path))
        print(f'Model loaded from {path}')