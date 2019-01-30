
# coding: utf-8

# In[1]:


import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import MaxPool2d
import math
from functools import reduce


# In[2]:


class mnist_2Layer(nn.Module):
    def __init__(self):
        super(mnist_2Layer, self).__init__()
        self.conv_layer = Conv2d(in_channels=1, out_channels=20, kernel_size=3, stride=1, padding=0)
        self.fully_connected_layer = Linear(in_features=20*26*26, out_features=10, bias=True)
        
    def forward(self,x):
        #print("Input size: ", x.size())
        x1 = self.conv_layer(x)
        #print("Convolve size output: ", x1.size())
        x2 = F.relu(x1)
        #print("ReLU size output: ", x2.size())  
        # Resize output to linear tensor, to match product of dimensions of fully connected layer
        x3 = x2.view(-1, 20*26*26)
        x4 = self.fully_connected_layer(x3)
        #x5 = F.relu(x4) removing this increases accuracy a lot!!!!
        #print("Fully connected size output: ", x3.size())        
        return x4


# In[3]:


class mnist_5Layer(nn.Module):
    def __init__(self):
        super(mnist_5Layer, self).__init__()
        self.layer1 = Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.layer2 = MaxPool2d(kernel_size=2, stride=2)
        self.layer3 = Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.layer4 = Linear(in_features=64*14*14, out_features=1024, bias=True)
        self.layer5 = Linear(in_features=1024, out_features=10, bias=True)

    def forward(self, x):
        #print("Input size: ", x.size())
        
        x1 = self.layer1(x)
        #print("Layer 1: Convolve output size: ", x1.size())
        
        x1p5 = F.relu(x1)
        #print("Layer 1.5: Relu output: ", x1p5.size())
        
        x2 = self.layer2(x1p5)
        #print("Layer 2: Pooling output size: ", x2.size())
        
        x2p5 = F.relu(x2)
        #print("Layer 2.5: Relu output: ", x2p5.size())
        
        x3 = self.layer3(x2p5)
        #print("Layer 3: Convolve output size: ", x3.size())
        
        x3p5 = F.relu(x3)
        #print("Layer 3.5: Relu output: ", x3p5.size())
        
        # Do a 'view' transformation going from convolve to connected layers
        x3p5_vectorized = x3p5.view(-1, 64*14*14)
        
        x4 = self.layer4(x3p5_vectorized)
        #print("Layer 4: Connected output: ", x4.size())
        
        x5 = self.layer5(x4)
        #print("Layer 5: Connected output: ", x5.size())
        
        return x5


# In[12]:


class mnist_modified(nn.Module):
    def __init__(self):
        super(mnist_modified, self).__init__()
        self.layer1 = Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.layer2 = MaxPool2d(kernel_size=2, stride=2)
        self.layer3 = Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.layer4 = Linear(in_features=64*14*14, out_features=64*14*14, bias=True)
        self.layer5 = Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2)
        self.layer6 = MaxPool2d(kernel_size=2, stride=2)
        self.layer7 = Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=1, padding=2)
        self.layer8 = Linear(in_features=256*7*7, out_features=1024, bias=True)
        self.layer9 = Linear(in_features=1024, out_features=10, bias=True)
        
    def forward(self,x):
        x = self.layer1(x)
        x = F.relu(x)
        x = self.layer2(x)
        x = F.relu(x)
        x = self.layer3(x)
        x = F.relu(x)        
        x = x.view(-1, 64*14*14)
        x = self.layer4(x)
        x = x.view(-1, 64, 14, 14)
        x = F.relu(x)
        x = self.layer5(x)
        x = F.relu(x)
        x = self.layer6(x)
        x = F.relu(x)
        x = self.layer7(x)
        x = F.relu(x)
        x = x.view(-1, 256*7*7)
        x = self.layer8(x)
        x = self.layer9(x)        
        
        return x


# In[13]:


# Run this to convert the juypter notebook to a python file
get_ipython().system('jupyter nbconvert --to script model.ipynb')

