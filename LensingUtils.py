import numpy as np
from numba import jit
import torch

#Normalization of kappa to obtain a log-normal(almost) distribution
kmin = -0.015
k0 = -kmin

@jit(nopython=True)
def log_kappa(kappa):
    log_kappa_k0 = np.log(k0+kappa)
    return log_kappa_k0

@jit(nopython=True)
def sigma_logN(log_kappa_k0):
    #log_kappa_k0_flat = log_kappa_k0.flatten()
    sigma = np.std(log_kappa_k0)
    return sigma

@jit(nopython=True)
def mu_logN(kappa,sigma):
    mu = np.log(k0+np.mean(kappa)) - sigma**2/2
    return mu

@jit(nopython=True)
def logN_kappa(log_kappa_k0,mu,sigma):
    x = (log_kappa_k0 - mu) / (2*sigma)
    return x

def shifted_logN_kappa_samples(samples):
    kappa_flat = [kappa.flatten() for kappa in samples]
    sqr_s = np.sqrt(len(kappa_flat[0])).astype(int)
    log_kappa_k0 = [log_kappa(kappa) for kappa in kappa_flat]
    sigma = [sigma_logN(log_k) for log_k in log_kappa_k0]
    mu = [mu_logN(kappa,s) for kappa,s in zip(kappa_flat,sigma)]
    x= [logN_kappa(log_k,m,s).reshape(sqr_s,sqr_s) for log_k,m,s in zip(log_kappa_k0,mu,sigma)]
    
    return x,mu,sigma

pmin = -4
pmax = 4
def datanorm(sample):
    """
    Normalize data between 0 and 1 (optional)
    """
    ids_max = sample > pmax
    sample[ids_max] = pmax
    sample_norm = (sample.to(torch.float32) - pmin) / (pmax - pmin)
    return sample_norm

def discretize(sample):
    return (sample * 255).to(torch.int32).unsqueeze(dim=0)

def pixel2normval(sample):
    return  np.squeeze(sample) / 255

def unnormdata(sample_norm):
    return (sample_norm * (pmax - pmin)) + pmin
    
@jit(nopython=True)
def desgaussianize(x,mu,sigma):
    kappa = np.exp((x*2*sigma) + mu) - k0
    return kappa