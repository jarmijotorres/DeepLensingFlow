import os,sys
import math
import time
import numpy as np
from tqdm import tqdm
from glob import glob
from torchvision import transforms
from torch.utils import data

#from load_data import load_data
from DeepLensingFlow.load_data import *
#
from DeepLensingFlow.LensingUtils import *

# PyTorch Lightning
import pytorch_lightning as pl

from DeepLensingFlow.NormalizingFlow.flows import create_multiscale_flow
#
pl.seed_everything(42)
# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Fetching the device that will be used throughout this notebook
device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
print("Using device", device)
#
#
#Load data
dataset_name = sys.argv[1]
list_data = os.listdir(dataset_name)
N_maps = len(list_data)
data_sample = [np.load(dataset_name+l) for l in list_data]
#Nmap_gen = 100#int(sys.argv[1])#500
#dataset = load_data(name=dataset_name)
#dataset.gen_from_maps(Nmap_gen=Nmap_gen,is_all_maps = False)
#
#metadata params
map_size = 88#256
batch_size = 500
N_imgs = len(data_sample)#60000
max_epochs = 300#int(sys.argv[1])
NF_shape = (N_imgs,1,1,map_size,map_size)
#map_ids = torch.randint(low=0,high=Nmap_gen,size=(N_imgs,))

#transform = transforms.Compose([transforms.ToTensor(),
#                            transforms.RandomCrop(size=map_size),
#                            transforms.RandomHorizontalFlip(),
#                            transforms.RandomVerticalFlip(),
#                            datanorm,
#                            discretize])

dataset = np.array(data_sample)
#clean_logdata = log_dataset[~np.isnan(log_dataset).any(axis=(1,2))]

dataset_tensor = torch.tensor(dataset).reshape(NF_shape)

ts = datanorm(dataset_tensor,pmin = -0.01,pmax = 0.15)
training_set = discretize(ts)[0]

train_set = training_set[:int(N_imgs*0.8)]
val_set = training_set[int(N_imgs*0.8):int(N_imgs*0.95)]
test_set = training_set[int(N_imgs*0.95):]


#dataset.sample(map_ids,transform,NF_shape)
#train_set = dataset.x[:int(N_imgs*0.8)]
#val_set = dataset.x[int(N_imgs*0.8):int(N_imgs*0.95)]
#test_set = dataset.x[int(N_imgs*0.95):]



train_loader = data.DataLoader(train_set,
                               batch_size=batch_size,
                               shuffle=False,
                               num_workers=16,
				drop_last=False)
val_loader = data.DataLoader(val_set,
                             batch_size=16,
                             shuffle=False,
                             num_workers=16,
				drop_last=False)

test_loader = data.DataLoader(test_set,
                              batch_size=16,
                              shuffle=False,
                              num_workers=16,
				drop_last=False)
#
#
#
#

def print_num_params(model):
    num_params = sum([np.prod(p.shape) for p in model.parameters()])
    print("Number of parameters: {:,}".format(num_params))

def train_flow(flow, model_name="RealNVP_MS"):
    # Create a PyTorch Lightning trainer
    trainer = pl.Trainer(default_root_dir=None, 
                         accelerator="gpu" if str(device).startswith("cuda") else "cpu",
                         devices=1,
                         max_epochs=max_epochs, 
                         gradient_clip_val=1.0,
                         check_val_every_n_epoch=5)
    trainer.logger._log_graph = True
    trainer.logger._default_hp_metric = None # Optional logging argument that we don't need
    
    result = None
    print("Start training", model_name)
    trainer.fit(flow, train_loader, val_loader)
    
    # Test best model on validation and test set if no result has been found
    # Testing can be expensive due to the importance sampling.
    if result is None:
        val_result = trainer.test(flow, val_loader, verbose=False)
        start_time = time.time()
        test_result = trainer.test(flow, test_loader, verbose=False)
        duration = time.time() - start_time
        result = {"test": test_result, "val": val_result, "time": duration / len(test_loader) / flow.import_samples}        
    return flow, result
#
#Create flow and train
flow_model = create_multiscale_flow(grid_size=map_size,device=device)

flow_model,res = train_flow(flow_model)

weights_out = sys.argv[2]

torch.save(flow_model.state_dict(), weights_out+'_maxepochs_%d.pt'%max_epochs)
#loss = flow_model.
#print('current loss: %.2lf')
#import json

#with open('/home/jarmijo/Normalizing-flow-for-WL-map-statistic/weights/NF_model.json', 'w') as fp:
#    json.dump(flow_dict, fp)
#
#

