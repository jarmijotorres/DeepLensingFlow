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

output_dir = sys.argv[1]
n_timesteps = 300
beta_minmax=[1e-4, 2e-2]
inference_batch_size = 10

map_size = 256
N_imgs = 10000
dataset_shape = (N_imgs,1,map_size,map_size)
img_size = dataset_shape[1:]
hidden_dims = [256,256]


model = SimpleUnet(ch_inp=1)
diffusion = Diffusion(model, image_resolution=img_size, n_times=n_timesteps, 
                      beta_minmax=beta_minmax, device=device).to(device)

weight_file = sys.argv[2]
weight_dict = torch.load(weight_file)
model.load_state_dict(weight_dict)
model.eval()

inference_batch_size = 10
id_i = 0
Nc = 1
for c in range(Nc):
    with torch.no_grad():
        generated_images = diffusion.sample(N=inference_batch_size)
    for i,map_i in enumerate(generated_images):
        id_i = c*inference_batch_size + i
        np.save(output_dir+'DMgen_%d.npy'%id_i,map_i[0].cpu().detach().numpy())    
    
