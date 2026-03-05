import torch
import torch.nn as nn

class SolarEfficiencyANN(nn.Module):
    def __init__(self, input_head=7, noise_std=0.01):
        super(SolarEfficiencyANN, self).__init__()
        self.noise_std = torch.tensor(noise_std, dtype=torch.float32)
        
        self.network = nn.Sequential(
            nn.Linear(input_head, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid() 
        )

    def forward(self, x):
        if self.training and self.noise_std > 0:
            x = x + torch.randn_like(x, dtype=torch.float32) * self.noise_std
        return self.network(x)

class SolarEfficiencyANNFWH(nn.Module):
    def __init__(self, input_head=6, noise_std=0.01):                         
        super(SolarEfficiencyANNFWH, self).__init__()                            

        self.register_buffer('noise_std', torch.tensor(noise_std, dtype=torch.float32)) 
                                                                              
        self.network = nn.Sequential(                                         
            nn.Linear(input_head, 128),                                       
            nn.ReLU(),                                                        
            nn.Linear(128, 64),                                               
            nn.ReLU(),                                                        
            nn.Dropout(0.2),                                                  
            nn.Linear(64, 32),                                                
            nn.ReLU(),                                                        
            nn.Linear(32, 1),
            # Use Sigmoid only if target is normalized efficiency (0-1)
            # If target is raw Heat Gain (Qu), use nn.Identity() or nothing here
            nn.Sigmoid()                                          
        )                                                                     
                                                                              
    def forward(self, x):                                                     
        # Only inject noise during training to help regularization
        if self.training and self.noise_std > 0:                              
            x = x + torch.randn_like(x) * self.noise_std 
        return self.network(x)
