import os,sys
import math
import h5py
import time
import numpy as np
from tqdm import tqdm
from glob import glob
from torchvision import transforms
from torch.utils import data
import torch
from scipy.ndimage import zoom

#sys.path.append('/home/x-jarmijotorre/')
from DeepLensingFlow.load_data import *
from DeepLensingFlow.LensingUtils import *
from DeepLensingFlow.NormalizingFlow.flows import create_multiscale_flow,create_multiscale_flow_simple

# PyTorch Lightning
import pytorch_lightning as pl
#
pl.seed_everything(42)
# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#def addnoise(mapi,scale=1):
#    std_maps = 0.00868998
#    noise = np.random.normal(loc=0.0,scale=scale*std_maps,size=(256,256))
#    return mapi + noise

# Fetching the device that will be used throughout this notebook
device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
print("Using device", device)
#

dataset_name =  h5py.File('/anvil/scratch/x-jarmijotorre/Kappamaps/ML_datasets/SLICSdataset/SLICS_5deg_256x256noisy_Ngen100.hdf5','r') 

data_sample_in = dataset_name['training_set'][:]
#data_sample_in = [addnoise(m) for m in data_sample[:,0,:,:]]
#data_sample_in = dataset_name['SLICS_64x64']
#metadata params
map_size = data_sample_in[0].shape[-1]
batch_size = 10
N_imgs = len(data_sample_in)#60000
#max_epochs = 200#int(sys.argv[1])
NF_shape = (N_imgs,1,map_size,map_size)
#map_ids = torch.randint(low=0,high=Nmap_gen,size=(N_imgs,))
np.savetxt('dataset_min_max.dat',np.array([data_sample_in.min()-0.01,data_sample_in.max()+0.01]))
#dataset = np.array(data_sample_in)
dataset_tensor = torch.tensor(data_sample_in).reshape(NF_shape)

ts = datanorm(dataset_tensor,pmin = dataset_tensor.min()-0.01,pmax = dataset_tensor.max()+0.01)
training_set = discretize(ts)[0]

train_set = training_set[:int(N_imgs*0.8)]
val_set = training_set[int(N_imgs*0.8):]
#test_set = training_set[int(N_imgs*0.95):]


train_loader = data.DataLoader(train_set,
                               batch_size=batch_size,
                               shuffle=False,
                               num_workers=1,
				drop_last=False)
val_loader = data.DataLoader(val_set,
                             batch_size=5,
                             shuffle=False,
                             num_workers=1,
				drop_last=False)


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

#flow_model,res = train_flow(flow_model)

weights_out = '/home/x-jarmijotorre/torchDLF_weights/NF_multiscale3_NormpixelNoisy_normalkappa_256x256_DataAug_noise_'#sys.argv[2]

epochs=301
losses=[]
val_losses = []
optimizer,_ = flow_model.configure_optimizers()
for epoch in range(epochs):
    loss_epoch = []
    val_loss_epoch = []
    for batch,val_batch in zip(train_loader,val_loader):
        batch = batch.to(device)
        flow_model.on_train_batch_start(batch=batch,batch_idx=0)
        loss = flow_model.training_step(batch=batch,batch_idx=0)
        flow_model.optimizer_zero_grad(epoch=epoch,batch_idx=0,optimizer=optimizer[0])
        loss.backward()
        flow_model.optimizer_step(epoch=epoch,batch_idx=0,optimizer=optimizer[0])        
        loss_epoch.append(loss.detach().to('cpu'))
        
        val_batch = val_batch.to(device)
        val_loss = flow_model.validation_step(batch=val_batch,batch_idx=0)
        val_loss_epoch.append(val_loss.detach().to('cpu'))
    if epoch % 5 == 0:
        print('Epoch: %d\t Loss: %.5lf'%(epoch,loss))
        print('Epoch: %d\t val Loss: %.5lf'%(epoch,val_loss))
    if epoch % 100 == 0:
        #print('Epoch: %d\t Loss: %.2lf'%(epoch,loss))
        torch.save(flow_model.state_dict(), weights_out+'checkpoint_%d.pt'%epoch)
    loss_mean_epoch = torch.mean(torch.Tensor(loss_epoch))
    valloss_mean_epoch = torch.mean(torch.Tensor(val_loss_epoch))
    losses.append(loss_mean_epoch)
    val_losses.append(valloss_mean_epoch)

loss_epochs = torch.Tensor(losses)
val_loss_epochs = torch.Tensor(val_losses)
t = torch.arange(loss_epochs.shape[0])

loss_array = torch.stack((t,loss_epochs,val_loss_epochs)).T

np.savetxt('/home/x-jarmijotorre/torchDLF_weights/Loss/NF_multiscale3_normpix_normalkappa_256x256_epochs_loss_dataAug.dat',loss_array)
        
torch.save(flow_model.state_dict(), weights_out+'maxepochs_%d.pt'%epochs)
