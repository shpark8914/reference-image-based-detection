import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Con_Att(nn.Module): 
    def __init__(self, channel):
        super(Con_Att, self).__init__()
        self.spatial = nn.Conv2d(channel, channel, 1)
        #self.spatial = nn.Conv2d(channel, channel, 3, padding=1) #kernel size 3
        #self.spatial = nn.Conv2d(channel, channel, 7, padding=3) #kernsl size 7
    
    def forward(self, x1, x2):
        #d = torch.abs(x1-x2)
        d = F.relu(x1-x2)
        d = self.spatial(d)
        attention = F.sigmoid(d)
        
        return x1*attention, x2*attention
        
            
            
            
            
