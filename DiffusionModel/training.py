import torch
from torch.utils import data
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

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
output_dir = sys.argv[1]
timestep_embedding_dim = 150
n_layers = 8
n_timesteps = 300
beta_minmax=[1e-4, 2e-2]

train_batch_size = 500
inference_batch_size = 4
lr = 1e-5

epochs = 201


#====================== Load data ================================#
dataset_name = sys.argv[2]
list_data = os.listdir(dataset_name)

dataset = [np.load(dataset_name+s) for s in list_data]
map_size = len(dataset[0])

#map_ids = torch.randint(low=0,high=Nmap_gen,size=(N_imgs,))
log_dataset,_mu,_sigma = shifted_logN_kappa_samples(dataset)
log_dataset = np.array(log_dataset)
clean_logdata = log_dataset[~np.isnan(log_dataset).any(axis=(1,2))]

N_imgs = len(clean_logdata)
dataset_shape = (N_imgs,1,map_size,map_size)
logdataset_tensor = torch.tensor(clean_logdata).reshape(dataset_shape)


training_set = logdatanorm(logdataset_tensor)

del logdataset_tensor,log_dataset
#dataset.sample(map_ids,transform,dataset_shape)

train_dataset = training_set[:int(N_imgs*(0.95))]#dataset.x[:int(N_imgs*(0.95))]
test_dataset = training_set[int(N_imgs*(0.95)):]#dataset.x[int(N_imgs*(0.05)):]

kwargs = {'num_workers': 1, 'pin_memory': True} 
train_loader = data.DataLoader(dataset=train_dataset, batch_size=train_batch_size, shuffle=True, **kwargs)
test_loader  = data.DataLoader(dataset=test_dataset,  batch_size=5, shuffle=False,  **kwargs)


#------------------------------ create functions ------------------------------------#

img_size = dataset_shape[1:]
hidden_dims = [88,88]

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
        
        #log_p1 = normal_dist.log_prob(pred_epsilon)
        #log_p2 = normal_dist.log_prob(epsilon)
        
        loss = denoising_loss(pred_epsilon, epsilon)
        #loss = denoising_loss(np.exp(log_p1), log_p2)
        
        noise_prediction_loss += loss.item()
        
        loss.backward()
        optimizer.step()
        
    print("\tEpoch", epoch + 1, "complete!", "\tDenoising Loss: ", noise_prediction_loss / batch_idx)
    if (epoch % 100) == 0:
        torch.save(model.state_dict(), output_dir+'weights/weights_checkpoint_%d.pt'%epoch)
    loss_epochs.append(noise_prediction_loss/ batch_idx)
    
print("Finish!!")

torch.save(model.state_dict(), output_dir+'weights/weights_.pt')

epochs_array = np.arange(epochs)
loss_epochs = np.array(loss_epochs)
S = np.array([epochs_array,loss_epochs]).T
np.savetxt(output_dir+'MAE_loss_epochs.dat',S)
