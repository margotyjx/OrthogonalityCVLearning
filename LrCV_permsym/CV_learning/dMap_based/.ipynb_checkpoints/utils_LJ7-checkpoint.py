import numpy as np
import torch
import torch.nn as nn
def dist2(x):
    n = len(x)
    if x.ndim == 2:
        x = x.reshape((n,2,7))
    Na = x.shape[2]
    r2 = np.zeros((n,Na,Na)) # matrix of distances squared

    for k in range(Na):
        r2[:,k,:] = (x[:,0,:]-x[:,0,k].reshape(-1,1))**2 + (x[:,1,:]-x[:,1,k].reshape(-1,1))**2
        r2[:,k,k] = 1
    return r2

def dist2_pt(x):
    n = len(x)
    if torch.Tensor.dim(x) == 2:
        x = x.reshape((n,2,7))
    Na = x.shape[2]
    r2 = torch.zeros((n,Na,Na)) # matrix of distances squared

    for k in range(Na):
        r2[:,k,:] = (x[:,0,:]-x[:,0,k].reshape(-1,1))**2 + (x[:,1,:]-x[:,1,k].reshape(-1,1))**2
        r2[:,k,k] = 1
    return r2

def LJpot(x): # Lennard-Jones potential, x is the position of each particles
    if torch.Tensor.dim(x) == 1:
        x = x.reshape((2,7))
    Na = x.shape[1] # x has shape [2,7] 
    r2 = torch.zeros((Na,Na)) # matrix of distances squared
    for k in range(Na):
        r2[k,:] = (x[0,:]-x[0,k])**2 + (x[1,:]-x[1,k])**2
        r2[k,k] = 1
    er6 = torch.div(torch.ones_like(r2),r2**3) 
    L = (er6-torch.tensor(1))*er6
    V = 2*torch.sum(L) 
    return V

def LJgrad(x):
    if torch.Tensor.dim(x) == 1:
        x = x.reshape((2,7))
    Na = x.shape[1]
    r2 = torch.zeros((Na,Na)) # matrix of distances squared
    for k in range(Na):
        r2[k,:] = (x[0,:]-x[0,k])**2 + (x[1,:]-x[1,k])**2
        r2[k,k] = 1
    r6 = r2**3
    L = -6*torch.div((2*torch.div(torch.ones_like(r2),r6)-1),(r2*r6)) # use r2 as variable instead of r
    g = torch.zeros_like(x)
    for k in range(Na):
        Lk = L[:,k]
        g[0,k] = torch.sum((x[0,k] - x[0,:])*Lk)
        g[1,k] = torch.sum((x[1,k] - x[1,:])*Lk)
    
    g = 4*g 
    return g

def LJgrad_batch(x):
    n = len(x)
    if torch.Tensor.dim(x) == 2:
        x = x.reshape((n,2,7))
    Na = x.shape[2]
    r2 = torch.zeros((n,Na,Na)) # matrix of distances squared

    for k in range(Na):
        r2[:,k,:] = (x[:,0,:]-x[:,0,k].reshape(-1,1))**2 + (x[:,1,:]-x[:,1,k].reshape(-1,1))**2
        r2[:,k,k] = 1

    r6 = r2**3
    L = -6*torch.div((2*torch.div(torch.ones_like(r2),r6)-1),(r2*r6))
    g = torch.zeros_like(x)
    for k in range(Na):
        Lk = L[:,:,k]
        g[:,0,k] = torch.sum((x[:,0,k].reshape(-1,1) - x[:,0,:])*Lk, dim = 1)
        g[:,1,k] = torch.sum((x[:,1,k].reshape(-1,1) - x[:,1,:])*Lk, dim = 1)
    g = 4*g 
    return g.reshape(n,2*Na)

def LJpot_np(x): # Lennard-Jones potential, x is the position of each particles
    if x.ndim == 1:
        x = x.reshape((2,7))
    Na = np.size(x,axis = 1) # x has shape [2,7] 
    r2 = np.zeros((Na,Na)) # matrix of distances squared
    for k in range(Na):
        r2[k,:] = (x[0,:]-x[0,k])**2 + (x[1,:]-x[1,k])**2
        r2[k,k] = 1
        
    er6 = np.divide(np.ones_like(r2),r2**3) 
    L = (er6-1)*er6
    V = 2*np.sum(L) 
    return V

def LJpot_np_batch(x): # Lennard-Jones potential, x is the position of each particles
    n = len(x)
    if x.ndim == 2:
        x = x.reshape((n,2,7))
    Na = x.shape[2]
    r2 = np.zeros((n,Na,Na)) # matrix of distances squared

    for k in range(Na):
        r2[:,k,:] = (x[:,0,:]-x[:,0,k].reshape(-1,1))**2 + (x[:,1,:]-x[:,1,k].reshape(-1,1))**2
        r2[:,k,k] = 1
        
    er6 = np.divide(np.ones_like(r2),r2**3) 
    L = (er6-1)*er6
    V = 2*np.sum(np.sum(L, axis= 1), axis = 1)  
    return V

#dV/dx_i = 4*sum_{i\neq j}(-12r_{ij}^{-13} + 6r_{ij}^{-7})*(x_i/r_{ij})
def LJgrad_np(x):
    if x.ndim == 2:
        x = x.reshape((n,2,7))
    Na = np.size(x,axis = 1)
    r2 = np.zeros((Na,Na)) # matrix of distances squared
    for k in range(Na):
        r2[k,:] = (x[0,:]-x[0,k])**2 + (x[1,:]-x[1,k])**2
        r2[k,k] = 1
    r6 = r2**3
    L = -6*np.divide((2*np.divide(np.ones_like(r2),r6)-1),(r2*r6)) # use r2 as variable instead of r
    g = np.zeros_like(x)
    for k in range(Na):
        Lk = L[:,k]
        g[0,k] = np.sum((x[0,k] - x[0,:])*Lk)
        g[1,k] = np.sum((x[1,k] - x[1,:])*Lk)
    g = 4*g 
    return g

def C_batch(x):
    n = len(x)
    if torch.Tensor.dim(x) == 2:
        x = x.reshape((n,2,7))
    Na = x.shape[2]
    r2 = torch.zeros((n,Na,Na)) # matrix of distances squared
    C = torch.zeros((n,Na))
    
    for k in range(Na):
        r2[:,k,:] = (x[:,0,:]-x[:,0,k].reshape(-1,1))**2 + (x[:,1,:]-x[:,1,k].reshape(-1,1))**2
        r2[:,k,k] = 0
    
    for i in range(Na):
        ci = torch.div(torch.ones_like(r2[:,i,:]) - (r2[:,i,:]/2.25)**4, torch.ones_like(r2[:,i,:]) - (r2[:,i,:]/2.25)**8)
        ci_sum = torch.sum(ci, dim = 1) - torch.tensor(1)
        C[:,i] = ci_sum
    return C


def C(x):
    if torch.Tensor.dim(x) == 1:
        x = x.reshape((2,7))
    Na = x.shape[1] # x has shape [2,7]
    C = torch.zeros(Na)
    r2 = torch.zeros((Na,Na)) # matrix of distances squared
    for k in range(Na):
        r2[k,:] = (x[0,:]-x[0,k])**2 + (x[1,:]-x[1,k])**2
        r2[k,k] = 0
    for i in range(Na):
        ci = torch.div(torch.ones_like(r2[i,:]) - (r2[i,:]/2.25)**4, torch.ones_like(r2[i,:]) - (r2[i,:]/2.25)**8)
        ci_sum = torch.sum(ci) - torch.tensor(1)
        C[i] = ci_sum
    return C

def C_np(x):
    if x.ndim == 1:
        x = x.reshape((2,7))
    Na = x.shape[1] # x has shape [2,7]
    C = np.zeros(Na)
    r2 = np.zeros((Na,Na)) # matrix of distances squared
    for k in range(Na):
        r2[k,:] = (x[0,:]-x[0,k])**2 + (x[1,:]-x[1,k])**2
        r2[k,k] = 0
    for i in range(Na):
        ci = np.divide(np.ones_like(r2[i,:]) - (r2[i,:]/2.25)**4, np.ones_like(r2[i,:]) - (r2[i,:]/2.25)**8)
        ci_sum = np.sum(ci) - torch.tensor(1)
        C[i] = ci_sum
    return C

def mu2n3(x):
    C_list = C(x)
    ave_C = torch.mean(C_list)
    mu2 = torch.mean((C_list - ave_C)**2)
    mu3 = torch.mean((C_list - ave_C)**3)
    return mu2, mu3

def mu2n3_batch(x):
    C_list = C_batch(x)
    ave_C = torch.mean(C_list, dim = 2)
    mu2 = torch.mean((C_list - ave_C)**2, dim = 1)
    mu3 = torch.mean((C_list - ave_C)**3, dim = 1)
    return mu2, mu3

def mu2n3_np(x):
    C_list = C_np(x)
    ave_C = np.mean(C_list)
    mu2 = np.mean((C_list - ave_C)**2)
    mu3 = np.mean((C_list - ave_C)**3)
    return mu2, mu3

def deriv_mu(mu2,mu3,x):
    derivmu2 = torch.autograd.grad(mu2,x,allow_unused=True, retain_graph=True, \
                                             grad_outputs = torch.ones_like(mu2), create_graph=True)
    derivmu3 = torch.autograd.grad(mu3,x,allow_unused=True, retain_graph=True, \
                                             grad_outputs = torch.ones_like(mu3), create_graph=True)
    return derivmu2,derivmu3



def four_wells_conf():
    rstar = np.power(2.0,1/6) # the optimal LJ distance
    aux = 0.5*np.sqrt(3)
    a = np.arange(0,2*np.pi,np.pi/3.0)
    # trapezoid
    x_trap = rstar*np.array([[-1.5,-0.5,0.5,1.5,-1.0,0.0,1.0],[0.0,0.0,0.0,0.0,aux,aux,aux]]) # trapezoid
    # hexagon
    x_hex = rstar*np.array([np.cos(a),np.sin(a)])
    x_hex = np.concatenate((np.zeros((2,1)),x_hex),axis=1)
    # capped parallelogram 1
    x_cp1 = rstar*np.array([[-0.5,0.5,1.5,-1.0,0.0,1.0,-0.5],[0.0,0.0,0.0,aux,aux,aux,2.0*aux]])
    # capped parallelogram 2
    x_cp2 = rstar*np.array([[-0.5,0.5,1.5,-1.0,0.0,1.0,0.5],[0.0,0.0,0.0,aux,aux,aux,2.0*aux]])
    x_trap_flat = x_trap.reshape(14)
    x_hex_flat = x_hex.reshape(14)
    x_cp1_flat = x_cp1.reshape(14)
    x_cp2_flat = x_cp2.reshape(14)

    X_combined = np.vstack((x_hex_flat, x_cp1_flat, x_cp2_flat, x_trap_flat))

   
    return X_combined



def sort_r2(data):
    r2 = dist2(data)
    r,c = np.triu_indices_from(r2[0], k=1)
    rd_vec = r2[:,r,c]
    rd_vec.sort(axis = 1)
    return rd_vec

def sort_r2_pt(data):
    r2 = dist2_pt(data)
    r,c = torch.torch.triu_indices(r2[0].shape[0], r2[0].shape[1], 1)
    rd_vec = r2[:,r,c]
    sort_vec, indices = torch.sort(rd_vec, 1)
    return sort_vec

def C_np_batch_LJ7(x):
    n = len(x)
    if x.ndim == 2:
        x = x.reshape((n,2,7))

    print("x shape: ", x.shape)
    Na = x.shape[2] # x has shape [2,7]
    C = np.zeros((n,Na))
    r2 = np.zeros((n, Na,Na)) # matrix of distances squared

    for k in range(Na):
        r2[:,k,:] = (x[:,0,:]-x[:,0,k].reshape(-1,1))**2 + (x[:,1,:]-x[:,1,k].reshape(-1,1))**2
        r2[:,k,k] = 0

    for i in range(Na):
        ci = np.divide(np.ones_like(r2[:,i,:]) - (r2[:,i,:]/2.25)**4, np.ones_like(r2[:,i,:]) - (r2[:,i,:]/2.25)**8)
        ci_sum = np.sum(ci, 1) - 1
        C[:,i] = ci_sum
    return np.sort(C, axis=1)
