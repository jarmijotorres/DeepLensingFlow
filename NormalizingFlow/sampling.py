#cd
#cd WLNFCov/
import os,sys
import numpy as np
import torch
sys.path.append('/home/jarmijo/Normalizing-flow-for-WL-map-statistic/')

from RNVP_architechture import create_multiscale_flow_samp
#

#Load data
device = torch.device("cpu") #if not torch.cuda.is_available() else torch.device("cuda:0")
print("Using device", device)

map_size = 88#int(sys.argv[1])
Ngen = 1000#int(sys.argv[1])
flow_model = create_multiscale_flow_samp(grid_size = map_size,device=device)
#model_name = '/home/jarmijo/Normalizing-flow-for-WL-map-statistic/weights/NF_training_Nimgs60000_{0}slics.pt'.format(Ngen)
model_name = '/home/jarmijo/Normalizing-flow-for-WL-map-statistic/weights/NF_training_datanormed_CosmoGrid_maxepochs_301.pt'
flow_model.load_state_dict(torch.load(model_name))
flow_model = flow_model.to(device)
#


z_init = flow_model.prior.sample(sample_shape=[Ngen,16,int(map_size/8),int(map_size/8)])

sample,_ = flow_model.sample(img_shape=z_init.shape,z_init=z_init)


A = [np.save('/gpfs02/work/jarmijo/data/imgs/NF/normCosmoGrid/NFgen_88x88_%d.npy'%i,s) for i,s in enumerate(sample)]



   
