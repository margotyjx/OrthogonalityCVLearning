import numpy as np
import torch
import torch.nn as nn
"""
This file contains all utils for preprossessing the data
1, all LJ7 related preprocessing, including computing the mu2, mu3 and potential energy
2, adjustment on LJ7 data including translation and rotational adjustment
"""

def dist2_LJ8(x):
    n = len(x)
    if x.ndim == 2:
        x = x.reshape((n,3,8))
    Na = x.shape[2]
    r2 = np.zeros((n,Na,Na)) # matrix of distances squared

    for k in range(Na):
        r2[:,k,:] = (x[:,0,:]-x[:,0,k].reshape(-1,1))**2 + (x[:,1,:]-x[:,1,k].reshape(-1,1))**2 + (x[:,2,:]-x[:,2,k].reshape(-1,1))**2
        r2[:,k,k] = 1
    return r2

def dist2_pt_LJ8(x):
    n = len(x)
    if torch.Tensor.dim(x) == 2:
        x = x.reshape((n,3,8))
    Na = x.shape[2]
    r2 = torch.zeros((n,Na,Na)) # matrix of distances squared

    for k in range(Na):
        r2[:,k,:] = (x[:,0,:]-x[:,0,k].reshape(-1,1))**2 + (x[:,1,:]-x[:,1,k].reshape(-1,1))**2 + (x[:,2,:]-x[:,2,k].reshape(-1,1))**2
        r2[:,k,k] = 1
    return r2

def LJpot_LJ8(x): # Lennard-Jones potential, x is the position of each particles
    if torch.Tensor.dim(x) == 1:
        x = x.reshape((3,8))
    Na = x.shape[1] # x has shape [2,7] 
    r2 = torch.zeros((Na,Na)) # matrix of distances squared
    for k in range(Na):
        r2[k,:] = (x[0,:]-x[0,k])**2 + (x[1,:]-x[1,k])**2 + (x[2,:]-x[2,k])**2
        r2[k,k] = 1
    er6 = torch.div(torch.ones_like(r2),r2**3) 
    L = (er6-torch.tensor(1))*er6
    V = 2*torch.sum(L) 
    return V

def LJgrad_LJ8(x):
    if torch.Tensor.dim(x) == 1:
        x = x.reshape((3,8))
    Na = x.shape[1]
    r2 = torch.zeros((Na,Na)) # matrix of distances squared
    for k in range(Na):
        r2[k,:] = (x[0,:]-x[0,k])**2 + (x[1,:]-x[1,k])**2 + (x[2,:]-x[2,k])**2
        r2[k,k] = 1
    r6 = r2**3
    L = -6*torch.div((2*torch.div(torch.ones_like(r2),r6)-1),(r2*r6)) # use r2 as variable instead of r
    g = torch.zeros_like(x)
    for k in range(Na):
        Lk = L[:,k]
        g[0,k] = torch.sum((x[0,k] - x[0,:])*Lk)
        g[1,k] = torch.sum((x[1,k] - x[1,:])*Lk)
        g[2,k] = torch.sum((x[2,k] - x[2,:])*Lk)
    
    g = 4*g 
    return g

def LJgrad_batch_LJ8(x):
    n = len(x)
    if torch.Tensor.dim(x) == 2:
        x = x.reshape((n,3,8))
    Na = x.shape[2]
    r2 = torch.zeros((n,Na,Na)) # matrix of distances squared

    for k in range(Na):
        r2[:,k,:] = (x[:,0,:]-x[:,0,k].reshape(-1,1))**2 + (x[:,1,:]-x[:,1,k].reshape(-1,1))**2 + (x[:,2,:]-x[:,2,k].reshape(-1,1))**2
        r2[:,k,k] = 1

    r6 = r2**3
    L = -6*torch.div((2*torch.div(torch.ones_like(r2),r6)-1),(r2*r6))
    g = torch.zeros_like(x)
    for k in range(Na):
        Lk = L[:,:,k]
        g[:,0,k] = torch.sum((x[:,0,k].reshape(-1,1) - x[:,0,:])*Lk, dim = 1)
        g[:,1,k] = torch.sum((x[:,1,k].reshape(-1,1) - x[:,1,:])*Lk, dim = 1)
        g[:,2,k] = torch.sum((x[:,2,k].reshape(-1,1) - x[:,2,:])*Lk, dim = 1)
    g = 4*g 
    return g.reshape(n,3*Na)

def LJpot_np_LJ8(x): # Lennard-Jones potential, x is the position of each particles
    if x.ndim == 1:
        x = x.reshape((3,8))
    Na = np.size(x,axis = 1) # x has shape [2,7] 
    r2 = np.zeros((Na,Na)) # matrix of distances squared
    for k in range(Na):
        r2[k,:] = (x[0,:]-x[0,k])**2 + (x[1,:]-x[1,k])**2 + (x[2,:]-x[2,k])**2
        r2[k,k] = 1
        
    er6 = np.divide(np.ones_like(r2),r2**3) 
    L = (er6-1)*er6
    V = 2*np.sum(L) 
    return V

def LJpot_np_batch_LJ8(x): # Lennard-Jones potential, x is the position of each particles
    n = len(x)
    if x.ndim == 2:
        x = x.reshape((n,3,8))
    Na = x.shape[2]
    r2 = np.zeros((n,Na,Na)) # matrix of distances squared

    for k in range(Na):
        r2[:,k,:] = (x[:,0,:]-x[:,0,k].reshape(-1,1))**2 + (x[:,1,:]-x[:,1,k].reshape(-1,1))**2 + (x[:,2,:]-x[:,2,k].reshape(-1,1))**2
        r2[:,k,k] = 1
        
    er6 = np.divide(np.ones_like(r2),r2**3) 
    L = (er6-1)*er6
    V = 2*np.sum(np.sum(L, axis= 1), axis = 1)  
    return V

#dV/dx_i = 4*sum_{i\neq j}(-12r_{ij}^{-13} + 6r_{ij}^{-7})*(x_i/r_{ij})
def LJgrad_np_LJ8(x):
    n = len(x)
    if x.ndim == 2:
        x = x.reshape((n,3,8))
    Na = np.size(x,axis = 1)
    r2 = np.zeros((Na,Na)) # matrix of distances squared
    for k in range(Na):
        r2[k,:] = (x[0,:]-x[0,k])**2 + (x[1,:]-x[1,k])**2 + (x[2,:]-x[2,k])**2
        r2[k,k] = 1
    r6 = r2**3
    L = -6*np.divide((2*np.divide(np.ones_like(r2),r6)-1),(r2*r6)) # use r2 as variable instead of r
    g = np.zeros_like(x)
    for k in range(Na):
        Lk = L[:,k]
        g[0,k] = np.sum((x[0,k] - x[0,:])*Lk)
        g[1,k] = np.sum((x[1,k] - x[1,:])*Lk)
        g[2,k] = np.sum((x[2,k] - x[2,:])*Lk)
    g = 4*g 
    return g

def C_batch_LJ8(x):
    n = len(x)
    if torch.Tensor.dim(x) == 2:
        x = x.reshape((n,3,8))
    Na = x.shape[2]
    r2 = torch.zeros((n,Na,Na)) # matrix of distances squared
    C = torch.zeros((n,Na))
    
    for k in range(Na):
        r2[:,k,:] = (x[:,0,:]-x[:,0,k].reshape(-1,1))**2 + (x[:,1,:]-x[:,1,k].reshape(-1,1))**2 + (x[:,2,:]-x[:,2,k].reshape(-1,1))**2
        r2[:,k,k] = 0
    
    for i in range(Na):
        ci = torch.div(torch.ones_like(r2[:,i,:]) - (r2[:,i,:]/2.25)**4, torch.ones_like(r2[:,i,:]) - (r2[:,i,:]/2.25)**8)
        ci_sum = torch.sum(ci, dim = 1) - torch.tensor(1)
        C[:,i] = ci_sum
    return C

def C_np_batch_LJ8(x):
    n = len(x)
    if x.ndim == 2:
        x = x.reshape((n,3,8))
    Na = x.shape[2] # x has shape [2,7]
    C = np.zeros((n,Na))
    r2 = np.zeros((n, Na,Na)) # matrix of distances squared

    for k in range(Na):
        r2[:,k,:] = (x[:,0,:]-x[:,0,k].reshape(-1,1))**2 + (x[:,1,:]-x[:,1,k].reshape(-1,1))**2 + (x[:,2,:]-x[:,2,k].reshape(-1,1))**2
        r2[:,k,k] = 0

    for i in range(Na):
        ci = np.divide(np.ones_like(r2[:,i,:]) - (r2[:,i,:]/2.25)**4, np.ones_like(r2[:,i,:]) - (r2[:,i,:]/2.25)**8)
        ci_sum = np.sum(ci, 1) - 1
        C[:,i] = ci_sum
    return np.sort(C, axis=1)


def C_LJ8(x):
    if torch.Tensor.dim(x) == 1:
        x = x.reshape((3,8))
    Na = x.shape[1] # x has shape [2,7]
    C = torch.zeros(Na)
    r2 = torch.zeros((Na,Na)) # matrix of distances squared
    for k in range(Na):
        r2[k,:] = (x[0,:]-x[0,k])**2 + (x[1,:]-x[1,k])**2 + (x[2,:]-x[2,k])**2
        r2[k,k] = 0
    for i in range(Na):
        ci = torch.div(torch.ones_like(r2[i,:]) - (r2[i,:]/2.25)**4, torch.ones_like(r2[i,:]) - (r2[i,:]/2.25)**8)
        ci_sum = torch.sum(ci) - torch.tensor(1)
        C[i] = ci_sum
    return C

def C_np_LJ8(x):
    if x.ndim == 1:
        x = x.reshape((3,8))
    Na = x.shape[1] # x has shape [2,7]
    C = np.zeros(Na)
    r2 = np.zeros((Na,Na)) # matrix of distances squared
    for k in range(Na):
        r2[k,:] = (x[0,:]-x[0,k])**2 + (x[1,:]-x[1,k])**2 + (x[2,:]-x[2,k])**2
        r2[k,k] = 0
    for i in range(Na):
        ci = np.divide(np.ones_like(r2[i,:]) - (r2[i,:]/2.25)**4, np.ones_like(r2[i,:]) - (r2[i,:]/2.25)**8)
        ci_sum = np.sum(ci) - 1
        C[i] = ci_sum
    return C

def mu2n3_LJ8(x):
    C_list = C_LJ8(x)
    ave_C = torch.mean(C_list)
    mu2 = torch.mean((C_list - ave_C)**2)
    mu3 = torch.mean((C_list - ave_C)**3)
    return mu2, mu3

def mu2n3_batch_LJ8(x):
    C_list = C_batch_LJ8(x)
    ave_C = torch.mean(C_list, dim = 2)
    mu2 = torch.mean((C_list - ave_C)**2, dim = 1)
    mu3 = torch.mean((C_list - ave_C)**3, dim = 1)
    return mu2, mu3

def mu2n3_np_LJ8(x):
    C_list = C_np_LJ8(x)
    ave_C = np.mean(C_list)
    mu2 = np.mean((C_list - ave_C)**2)
    mu3 = np.mean((C_list - ave_C)**3)
    return mu2, mu3

def muZ_np_LJ8(x,moments):
    C_list = C_np_batch_LJ8(x)
    ave_C = np.mean(C_list, 1)
    Mu = np.zeros((len(C_list), len(moments)))
    cnt = 0
    for m in moments:
        Mu[:,cnt] = np.mean((C_list - ave_C[:,None])**m, 1)
        cnt += 1
    return Mu
