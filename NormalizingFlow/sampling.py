#cd
#cd WLNFCov/
import os,sys
import numpy as np
import torch
#sys.path.append('/home/jarmijo/Normalizing-flow-for-WL-map-statistic/')

from DeepLensingFlow.NormalizingFlow.flows import create_multiscale_flow
from DeepLensingFlow.LensingUtils import *
#

#Load data
device = torch.device("cpu") #if not torch.cuda.is_available() else torch.device("cuda:0")
print("Using device", device)

map_size = 256#int(sys.argv[1])
Ngen = 10#int(sys.argv[1])
flow_model = create_multiscale_flow(grid_size = map_size,device=device)
#model_name = '/home/jarmijo/Normalizing-flow-for-WL-map-statistic/weights/NF_training_Nimgs60000_{0}slics.pt'.format(Ngen)
model_name = '/home/x-jarmijotorre/torchDLF_weights/NF_multiscale_normpix_pixel_normalkappa_weights_maxepochs_200.pt'
flow_model.load_state_dict(torch.load(model_name))
flow_model = flow_model.to(device)
#


for c in range(100):
    z_init = flow_model.prior.sample(sample_shape=[Ngen,8,int(map_size/8),int(map_size/8)])

    sample = flow_model.sample(img_shape=z_init.shape,z_init=z_init)

    norm_sample = pixel2normval(sample)
    unnorm_sample = unnormdata(norm_sample,pmin = -0.05,pmax = 0.3)

    A = [np.save('/anvil/scratch/x-jarmijotorre/Kappamaps/NF_gen/gen3_G_ms/NFgen_256x256_%d.npy'%(c*100+i),s) for i,s in enumerate(sample)]



   
