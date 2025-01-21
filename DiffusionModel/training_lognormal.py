import torch
from torch.utils import data
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import h5py
import numpy as np

from tqdm import tqdm
from torchvision.utils import save_image, make_grid
from torchvision import transforms
import torch.utils.data as data

from tqdm import tqdm
from torch.optim import Adam

import math,os,sys

from DeepLensingFlow.load_data import load_data
from DeepLensingFlow.LensingUtils import *

from DeepLensingFlow.DiffusionModel.Diffusion import Diffusion, SimpleUnet

cuda = True
device = torch.device("cuda:0" if cuda else "cpu")

#++++++++++++++++++++++++ Parameters +++++++++++++++++++++++++++++++++#
timestep_embedding_dim = 300
n_layers = 8
n_timesteps = 300
beta_minmax=[1e-4, 2e-2]

train_batch_size = 100
lr = 1e-5

epochs = 500


#====================== Load data ================================#
output_dir = '/home/x-jarmijotorre/torchDLF_weights/'#sys.argv[1]
dataset_name = '/anvil/scratch/x-jarmijotorre/Kappamaps/Flask/ML_datasets/logGaussianMaps_10000imgs.hdf5'#sys.argv[1]
dataset_file = h5py.File(dataset_name,'r')
dataset = dataset_file['training_set'][:]

map_size = len(dataset[0])

dataset_shape = (len(dataset),1,map_size,map_size)
dataset_tensor = torch.reshape(torch.Tensor(dataset),dataset_shape)

#training_set =  datanorm(dataset_tensor,pmin = -0.015,pmax = 0.2)

train_dataset = dataset_tensor[:int(dataset_shape[0]*(0.95))]#dataset.x[:int(N_imgs*(0.95))]
test_dataset = dataset_tensor[int(dataset_shape[0]*(0.95)):]#dataset.x[int(N_imgs*(0.05)):]

kwargs = {'num_workers': 1, 'pin_memory': True} 
train_loader = data.DataLoader(dataset=train_dataset, batch_size=train_batch_size, shuffle=True, **kwargs)
test_loader  = data.DataLoader(dataset=test_dataset,  batch_size=5, shuffle=False,  **kwargs)


#------------------------------ create functions ------------------------------------#
img_size = dataset_shape[1:]
model = SimpleUnet(ch_inp=1)

diffusion = Diffusion(model, image_resolution=img_size, n_times=n_timesteps, 
                      beta_minmax=beta_minmax, device=device).to(device)

optimizer = Adam(diffusion.parameters(), lr=lr)
denoising_loss = nn.L1Loss()

#================ training and save ================================#

print("Start training DDPMs...")
model.train()

loss_epochs = []

normal_dist = torch.distributions.normal.Normal(loc=0.0,scale=1)

for epoch in range(epochs):
    noise_prediction_loss = 0
    for batch_idx, x in tqdm(enumerate(train_loader), total=len(train_loader)):
        optimizer.zero_grad()

        x = x.to(device)
        
        noisy_input, epsilon, pred_epsilon = diffusion(x)
        
        loss = denoising_loss(pred_epsilon, epsilon)
        
        noise_prediction_loss += loss.item()
        
        loss.backward()
        optimizer.step()
        
    print("\tEpoch", epoch + 1, "complete!", "\tDenoising Loss: ", noise_prediction_loss / batch_idx)
    if (epoch % 100) == 0:
        torch.save(model.state_dict(), output_dir+'chekcpoints/DM_weights_lognormalkappa_checkpoint_%d.pt'%epoch)
    loss_epochs.append(noise_prediction_loss/ batch_idx)
    
print("Finish!!")

torch.save(model.state_dict(), output_dir+'DM_weights_lognormalkappa_maxepochs%d.pt'%epochs)

epochs_array = np.arange(epochs)
loss_epochs = np.array(loss_epochs)
S = np.array([epochs_array,loss_epochs]).T
np.savetxt(output_dir+'Loss/MAE_loss_epochs_lognormalmaps.dat',S)
