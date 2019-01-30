
# coding: utf-8

# In[1]:


import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset as Dataset
from torchvision import transforms
import multiprocessing
import os
import torchvision
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from torchvision.transforms import ToPILImage
import torchvision.transforms as transforms


# In[4]:


class MNISTDataSet(Dataset):
    '''
    path2file - string - path of the MNIST dataset, must be .pt file
    transform - torchvision.transforms - synonymous with handler, what does this do
    '''
    def __init__(self, path2file, transform_handler=None):
        self.data_path = path2file
        self.transform_handler = transform_handler
        self.data_model = torch.load(self.data_path)
        # Extract and save the information in the class
        
    """
    index - index of image in dataset, no bounds checking done
    Returns tuple of (image, number), where image is a 2D tensor, and number is a 0-dim tensor
    """
    def __getitem__(self, index):
        tens = self.data_model[0][index]
        tens = tens.float() # To float tensor
        tens = tens / 255.0 # Normalize
        if (self.transform_handler != None):
            """
            tens = self.data_model[0][index]
            tens = tens.float()
            print("size of tens: ", tens.size())
            tens = tens.unsqueeze(-1) # Make tensor 3D, since ToPILImage complains
            print("size unsqueezed: ", tens.size())
            
            import IPython;IPython.embed()
            
            transform = transforms.Compose([
                transforms.ToPILImage(),
                # normalize
                # transforms.Normalize(),
                #self.transform_handler,
                transforms.Resize(10),
                transforms.ToTensor()
            ])
            b = transform(tens)
            b = tens.squeeze(-1)
            return (b, self.data_model[1][index])
            """
            tens = self.transform_handler(tens)
        else:
            return (tens, self.data_model[1][index])

    def __len__(self):
        return len(self.data_model[0])


# In[6]:


'''
Displays a tensor that represents a monocrome image
'''
'''
def show_mono_img(tens):
    if (type(tens) != torch.Tensor):
        raise Exception("Input argument should be torch.Tensor type")
    img = tens.numpy()
    
    plt.figure(figsize=(1,1))
    plt.rcParams["axes.edgecolor"] = "black"
    plt.rcParams["axes.linewidth"] = 1
    plt.xticks([], [])
    plt.yticks([], [])
    
    plt.imshow(img, cmap='Greys', interpolation='none')
    
    plt.show()

# DEBUGGING, COMMENT OUT TO TEST
# Get input data set file
inp_file = os.getcwd()
inp_file = inp_file + "/data/processed/training.pt"
print(inp_file, type(inp_file))

# Instantiate dataset object

ds = MNISTDataSet(inp_file, None)  # Without transformation
#ds = MNISTDataSet(inp_file, transforms.Resize(5)) # With a transformation

print("model type: ", type(ds.data_model))
#for i in range(len(ds.data_model)):
#    print("Model tuple index ", i, " is of type: ", type(ds.data_model[i]), ", example: ", ds.data_model[i][0])
print("num pics: ", len(ds))
#print(ds.data_model)

for i in range(0, 5):
    img_tensor, class_tensor = ds[i]
    #print("type of image: ", type(ds[i]))
    #print("dim of image: ", ds[i].size())        
    show_mono_img(img_tensor)
    print(class_tensor.item())
'''


