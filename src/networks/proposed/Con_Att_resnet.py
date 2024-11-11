import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import init
from .Con_Att import *

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class SequentialMultiInput(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_att=True):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        
        if use_att:
            self.att = Con_Att(planes)
        else :
            self.att = None

    def forward(self, x, y):
        residual_x = x
        residual_y = y
        
        x_out = self.conv1(x)
        x_out = self.bn1(x_out)
        x_out = self.relu(x_out)
        
        y_out = self.conv1(y)
        y_out = self.bn1(y_out)
        y_out = self.relu(y_out)
        
        x_out = self.conv2(x_out)
        x_out = self.bn2(x_out)
        
        y_out = self.conv2(y_out)
        y_out = self.bn2(y_out)
        
        if self.downsample is not None:
            residual_x = self.downsample(x)
            residual_y = self.downsample(y)
        
        if self.att is not None :
            x_out, y_out = self.att(x_out, y_out)
        
        x_out += residual_x
        y_out += residual_y
        
        x_out = self.relu(x_out)
        y_out = self.relu(y_out)

        return x_out , y_out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_att = True):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        
        if use_att:
            self.att = Con_Att(planes)
        else :
            self.att = None

    def forward(self, x, y):
        residual_x = x
        residual_y = y

        x_out = self.conv1(x)
        x_out = self.bn1(x_out)
        x_out = self.relu(x_out)
        
        y_out = self.conv1(y)
        y_out = self.bn1(y_out)
        y_out = self.relu(y_out)

        x_out = self.conv2(x_out)
        x_out = self.bn2(x_out)
        x_out = self.relu(x_out)

        y_out = self.conv2(y_out)
        y_out = self.bn2(y_out)
        y_out = self.relu(y_out)

        x_out = self.conv3(x_out)
        x_out = self.bn3(x_out)
        
        y_out = self.conv3(y_out)
        y_out = self.bn3(y_out)

        if self.downsample is not None:
            residual_x = self.downsample(x)
            residual_y = self.downsample(y)
        
        if self.att is not None :
            x_out, y_out = self.att(x_out, y_out)
        
        x_out += residual_x
        y_out += residual_y
        x_out = self.relu(x_out)
        y_out = self.relu(y_out)

        return x_out, y_out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes, use_att):
        self.inplanes = 64
        
        super(ResNet, self).__init__()
        
            
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64,  layers[0], use_att = use_att)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, use_att = use_att)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, use_att = use_att) 
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, use_att = use_att) 
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        nn.init.kaiming_normal_(self.fc.weight)
        for key in self.state_dict():
            if key.split('.')[-1]=="weight":
                if "conv" in key:
                    nn.init.kaiming_normal_(self.state_dict()[key], mode='fan_out')
                if "bn" in key:
                    self.state_dict()[key][...] = 1
            elif key.split(".")[-1]=='bias':
                self.state_dict()[key][...] = 0

    def _make_layer(self, block, planes, blocks, stride=1, use_att = True):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, use_att = use_att))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, use_att = use_att))

        return SequentialMultiInput(*layers)
    
    def forward(self, current, master):
        current = self.conv1(current)
        current = self.bn1(current)
        current = self.relu(current)
        current = self.maxpool(current)
        
        master = self.conv1(master)
        master = self.bn1(master)
        master = self.relu(master)
        master = self.maxpool(master)
        
        current, master = self.layer1(current, master) #layer1    
        current, master = self.layer2(current, master) #layer2
        current, master = self.layer3(current, master) #layer3
        current, master = self.layer4(current, master) #layer4
            
        current_f_map, master_f_map = current, master
        current = self.avgpool(current)
        current = torch.flatten(current, 1)
        current = self.fc(current)

        return current, current_f_map, master_f_map

def ResidualNet(depth, num_classes, use_att):

    assert depth in [18, 34, 50, 101], 'network depth should be 18, 34, 50 or 101'
        
    if depth == 18:
        model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes, use_att)

    elif depth == 34:
        model = ResNet(BasicBlock, [3, 4, 6, 3], num_classes, use_att) 

    elif depth == 50:
        model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes, use_att) 

    elif depth == 101:
        model = ResNet(Bottleneck, [3, 4, 23, 3], num_classes, use_att) 

    return model
