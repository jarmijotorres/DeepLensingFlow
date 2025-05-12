import os,sys
import numpy as np
import torch
import h5py
#sys.path.append('/home/jarmijo/Normalizing-flow-for-WL-map-statistic/')

from DeepLensingFlow.NormalizingFlow.flows import create_multiscale_flow,create_multiscale_flow_simple
from DeepLensingFlow.LensingUtils import *

device = torch.device("cpu") #if not torch.cuda.is_available() else torch.device("cuda:0")
print("Using device", device)

map_size = 256#int(sys.argv[1])
Ngen = 100#int(sys.argv[1])
flow_model = create_multiscale_flow(grid_size = map_size,device=device)
#model_name = '/home/jarmijo/Normalizing-flow-for-WL-map-statistic/weights/NF_training_Nimgs60000_{0}slics.pt'.format(Ngen)
model_name = '/home/x-jarmijotorre/torchDLF_weights/NF_multiscale3_NormpixelNoisy_normalkappa_256x256_DataAug_noise_checkpoint_200.pt'
flow_model.load_state_dict(torch.load(model_name))
flow_model = flow_model.to(device)

#vmin,vmax = (-0.0675,0.5568)
#vmin,vmax = (-0.0254,0.3069) #for 64x64 SLICS
#vmin,vmax =-0.0319, 0.5011 #for 128x128 SLICS
#vmin,vmax = (-0.0244,0.5389)#for 256x256
vmin,vmax = (-0.0680, 0.50)#for 256x256 SLICS#with noise

sample_data = []
for i in range(40):
    z_init = flow_model.prior.sample(sample_shape=[Ngen,32,16,16])#for size 256 double multiscale

    sample = flow_model.sample(img_shape=z_init.shape,z_init=z_init)

    norm_sample = pixel2normval(sample)
    unnorm_sample = unnormdata(norm_sample,pmin = vmin,pmax = vmax)
    sample_data.append(unnorm_sample)

sample_data = torch.Tensor(np.array(sample_data))    
    
S1 = sample_data.reshape((4000,256,256))

sample_file = h5py.File('/anvil/scratch/x-jarmijotorre/Kappamaps/NF_gen/sample_NFgen_img256x256_128x128_64x64_SLICSogSample_kappaMaps.hdf5','a')

   
sample_file.create_dataset(name='NFgen_256x256maps_300_epochs',data=S1)

sample_file.close()

#z shape is Ngen,8,16,16 for 1x64x64 maps using MS_simple (1 squeeze, 1split,1squeeze)
#z shape is Ngen,8,32,32 for 1x128x128 maps using MS_simple (1 squeeze, 1split,1squeeze)
#z shape is Ngen,8,64,64 for 1x256x256 maps using MS_simple (1 squeeze, 1split,1squeeze)

#z shape is Ngen,32,16,16 for 1x256x256 maps using MS3 ()