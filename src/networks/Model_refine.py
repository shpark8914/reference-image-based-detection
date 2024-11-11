#######################################################################
###  Define Model 
###################################################################
import torch
from torch import nn
from torchvision import models




class ResNet_Model(torch.nn.Module):
    def __init__(self,num_layer=18):
        # 모델 부분
        super().__init__()
        if num_layer==18:
            self.features = nn.Sequential(*list(models.resnet18(weights=None).children())[:-2])
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512, 2, bias=True) 
        
        elif num_layer==34:
            self.features = nn.Sequential(*list(models.resnet34(weights=None).children())[:-2])
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512, 2, bias=True) 
        
        elif num_layer==50:
            self.features = nn.Sequential(*list(models.resnet50(weights=None).children())[:-2])
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(2048, 2, bias=True) 
        
        elif num_layer==101:
            self.features = nn.Sequential(*list(models.resnet101(weights=None).children())[:-2])
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(2048, 2, bias=True) 

        elif num_layer==152:
            self.features = nn.Sequential(*list(models.resnet152(weights=None).children())[:-2])
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(2048, 2, bias=True) 

    def forward(self, x):
        features = self.features(x)
        x = self.avgpool(features)
        x = torch.flatten(x,1)
        output = self.fc(x)
        return output, features
