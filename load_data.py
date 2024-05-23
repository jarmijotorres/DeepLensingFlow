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
        self.list_data = []
        self.x = None
        
    def gen_from_maps(self,Nmap_gen,is_all_maps = True):
        self.Nmaps = Nmap_gen
        og_maps_list = os.listdir(self.data_dir)
        if is_all_maps:
            self.list_data = np.array(og_maps_list)
            dlist = [np.load(self.data_dir+l).astype(np.float32) for l in self.list_data]
            self.dataset = np.array(dlist)                  
        else:
            rand_ids = np.random.randint(0,len(og_maps_list),self.Nmaps)
            self.list_data = np.array(og_maps_list)[rand_ids]
            dlist = [np.load(self.data_dir+l).astype(np.float32) for l in self.list_data]
            self.dataset = np.array(dlist)
            
    def sample(self,map_ids,transform,shape):
        """
        Generate sample from original data. 
        inputs:
        -------
        transform: List of transforms (commonly from torchvision).
        Shape: The input shape of the data array for training.
        N: number of images for output. As we are augmenting data, We need to consider how much data is possible to generate from a finite sample.
        crop_width: Size of the augmented data.
        is_subsample: Dataset is created from a subsample. Only a N number of maps will be used. Default False.
        """
        self.Nsample = len(map_ids)
        self.map_shape = shape[1:]
        T = torch.empty(shape)
        for t_i in range(self.Nsample):
            T[t_i] = pre_process(self.dataset,map_ids[t_i],transform)
        self.x = T.reshape(shape)
        
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
        
def pre_process(A,id_i,transform):
    data_i = A[id_i]
    ds = transform(data_i)
    return ds


class load_diffused_data():
    """
    Class for loading and manipualate data to be trained by a neural network. Using Pytorch format to make it easier to create data augmentation and put data in pytorch Data loader.
    """
    def __init__(self,name):
        """
        input: name. Directory of files
        """
        self.data_dir = name
        self.list_data = []
        self.x = None
        
    def gen_from_maps(self,Nmap_gen,is_all_maps = True):
        self.Nmaps = Nmap_gen
        og_maps_list = os.listdir(self.data_dir)
        if is_all_maps:
            self.list_data = np.array(og_maps_list)
            dlist = [np.load(self.data_dir+l).astype(np.float32) for l in self.list_data]
            self.dataset = np.array(dlist)                  
        else:
            rand_ids = np.random.randint(0,len(og_maps_list),self.Nmaps)
            self.list_data = np.array(og_maps_list)[rand_ids]
            dlist = [np.load(self.data_dir+l).astype(np.float32) for l in self.list_data]
            self.dataset = np.array(dlist)
            
    def sample(self,map_ids,transform,shape):
        """
        Generate sample from original data. 
        inputs:
        -------
        transform: List of transforms (commonly from torchvision).
        Shape: The input shape of the data array for training.
        N: number of images for output. As we are augmenting data, We need to consider how much data is possible to generate from a finite sample.
        crop_width: Size of the augmented data.
        is_subsample: Dataset is created from a subsample. Only a N number of maps will be used. Default False.
        """
        self.Nsample = len(map_ids)
        self.map_shape = shape[1:]
        T = torch.empty(shape)
        for t_i in range(self.Nsample):
            T[t_i] = pre_process(self.dataset,map_ids[t_i],transform)
        self.x = T.reshape(shape)

        
    def diffuse_data(self,T, noise_dist_st = 1.0):
        
        betas = np.linspace(0.01, noise_dist_st, T,endpoint=False)

        # Pre-calculate different terms for closed form
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.concatenate([np.array([1.0]),alphas_cumprod[:-1]],axis=0)
        sqrt_recip_alphas = np.sqrt(1.0 / alphas)
        sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = np.sqrt(1. - alphas_cumprod)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        u = np.random.normal(size=self.x.shape)
        u = torch.Tensor(u)
        #timestep t
        #t = tf.random.uniform(shape=(1,1,1),minval=1,maxval=T,dtype=tf.int32)
        #t = tf.cast(t,tf.float32)
        t = np.random.randint(low=0,high=T,size=(self.x.shape[0],1))
        sqrt_alphas_cumprod_t = torch.Tensor(sqrt_alphas_cumprod[t])
        sqrt_one_minus_alphas_cumprod_t = torch.Tensor(sqrt_one_minus_alphas_cumprod[t])
        # Create noisy image
        #y = sqrt_alphas_cumprod_t * self.x + sqrt_one_minus_alphas_cumprod_t * u
        y = noise_img_torch_vec(self.x,u,sqrt_alphas_cumprod_t,sqrt_one_minus_alphas_cumprod_t)
        self.y = y
        self.u = u
        self.t = torch.Tensor(t)
        self.s = sqrt_one_minus_alphas_cumprod_t

        #return {'x':x, 'y':y, 'u':u,'s':sqrt_one_minus_alphas_cumprod_t}  
        
    def __getitem__(self,index):
        """
        For indexing. Internal.
        """
        x_i = self.x[index]
        y_i = self.y[index]
        u_i = self.u[index]
        t_i = self.t[index]
        s_i = self.s[index]
        return  {'x': x_i, 'y': y_i, 'u': u_i, 't':t_i, 's':s_i}
#        return x_i
    
    def __len__(self):
        """
        For reading size. Internal
        """
        return len(self.x)
    
    
        
def pre_process(A,id_i,transform):
    data_i = A[id_i]
    ds = transform(data_i)
    return ds

def noise_img(x,u,noise1,noise2):
    return noise1 * x + noise2 * u
    
noise_img_torch_vec = torch.vmap(noise_img)


