import os,sys
import numpy as np
import torch

class load_data():
    """
    Class for loading and manipualate data to be trained by a neural network. Using Pytorch format to make it easier to create data augmentation and put data in pytorch Data loader.
    """
    def __init__(self,name):
        """
        input: name. Directory of files
        """
        self.data_dir = name
        self.list_data = os.listdir(self.data_dir) 
        self.dataset = [np.load(self.data_dir+l).astype(np.float32) for l in self.list_data]
        self.x = None
        
    def sample(self,transform,shape,N, crop_width):
        """
        Generate sample from original data. 
        inputs:
        -------
        transform: List of transforms (commonly from torchvision).
        Shape: The input shape of the data array for training.
        N: number of images for output. As we are augmenting data, We need to consider how much data is possible to generate from a finite sample.
        crop_width: Size of the augmented data.
        """
        self.N = N
        self.map_size = crop_width
        map_ids = torch.randint(low=0,high=len(self.list_data),size=(self.N,))
        self.map_ids = map_ids
        data_N = np.array(self.dataset)[self.map_ids.numpy()]
        ds = [transform(s) for s in data_N]
        ds = torch.cat(ds)
        ds = ds.reshape(shape)
        self.x = ds

    def __getitem__(self,index):
        """
        For indexing. Internal.
        """
        x_i = self.x[index]
        return x_i
    
    def __len__(self):
        """
        For reading size. Internal
        """
        return len(self.x)
        
