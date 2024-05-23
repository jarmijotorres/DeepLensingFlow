import numpy as np
from numba import jit
import torch

#Normalization of kappa to obtain a log-normal(almost) distribution
kmin = -0.016
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


def logdatanorm(sample,pmin = -4,pmax = 4):
    """
    Normalize data between 0 and 1 (optional)
    """
    ids_max = sample > pmax
    sample[ids_max] = pmax
    sample_norm = (sample.to(torch.float32) - pmin) / (pmax - pmin)
    return sample_norm

def datanorm(sample,pmin = -0.016,pmax = 0.55):
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

def unnormdata(sample_norm,pmin = -0.016,pmax = 0.55):
    return (sample_norm * (pmax - pmin)) + pmin

def logunnormdata(sample_norm,pmin = -4,pmax = 4):
    return (sample_norm * (pmax - pmin)) + pmin

@jit(nopython=True)
def desgaussianize(x,mu,sigma):
    kappa = np.exp((x*2*sigma) + mu) - k0
    return kappa

def radial_profile(data):
    """
    Compute the radial profile of 2d image
    :param data: 2d image
    :return: radial profile
    """
    center = data.shape[0]/2
    y, x = np.indices((data.shape))
    r = np.sqrt((x - center)**2 + (y - center)**2)
    r = r.astype('int32')

    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr
    return radialprofile

def measure_power_spectrum(map_data, pixel_size):
    """
    measures power 2d data
    :param power: map (nxn)
    :param pixel_size: pixel_size (rad/pixel)
    :return: ell
    :return: power spectrum

    """
    data_ft = np.fft.fftshift(np.fft.fft2(map_data)) / map_data.shape[0]
    nyquist = int(map_data.shape[0]/2)
    power_spectrum_1d =  radial_profile(np.real(data_ft*np.conj(data_ft)))[:nyquist] * (pixel_size)**2
    k = np.arange(power_spectrum_1d.shape[0])
    ell = 2. * np.pi * k / pixel_size / 360
    return ell, power_spectrum_1d