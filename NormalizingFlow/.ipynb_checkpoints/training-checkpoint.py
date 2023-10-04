import os,sys
import time
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from torch.utils import data
# PyTorch Lightning
import pytorch_lightning as pl

from DeepLensingFlow.load_data import load_data
from DeepLensingFlow.NormalizingFlow.flow import create_multiscale_flow
from DeepLensingFlow.LensingUtils import *
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
map_size = 360
batch_size = 128
N_imgs=1000
NF_shape = (N_imgs,1,1,map_size,map_size)
dataset_name = "/gpfs02/work/jarmijo/KappaMaps/norm_quad_SLICS_Cov/slics_norm/"

transform = transforms.Compose([transforms.ToTensor(),
                            transforms.RandomCrop(size=map_size),
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomVerticalFlip(),
                            datanorm,
                            discretize])

dataset = load_data(name=dataset_name)
dataset.sample(transform,NF_shape,N=N_imgs,crop_width=map_size)
train_set = dataset.x[:int(N_imgs*0.8)]
val_set = dataset.x[int(N_imgs*0.8):int(N_imgs*0.95)]
test_set = dataset.x[int(N_imgs*0.95):]

train_loader = data.DataLoader(train_set,
                               batch_size=batch_size,
                               shuffle=False,
                               num_workers=16,
				drop_last=False)
val_loader = data.DataLoader(val_set,
                             batch_size=64,
                             shuffle=False,
                             num_workers=16,
				drop_last=False)

test_loader = data.DataLoader(test_set,
                              batch_size=64,
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

def train_flow(flow, model_name="RealNVP_Lensing"):
    # Create a PyTorch Lightning trainer
    trainer = pl.Trainer(default_root_dir=None, 
                         accelerator="gpu" if str(device).startswith("cuda") else "cpu",
                         devices=1,
                         max_epochs=151, 
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
flow_model = create_multiscale_flow(grid_size=map_size)
flow_model,res = train_flow(flow_model)
#save network in pytorch pickle object
torch.save(flow_model.state_dict(), '/home/jarmijo/Normalizing-flow-for-WL-map-statistic/weights/NF_RNVP+_Nimgs30000.pt')
