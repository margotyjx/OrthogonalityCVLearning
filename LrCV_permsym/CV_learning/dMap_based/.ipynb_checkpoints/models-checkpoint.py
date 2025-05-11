import torch
import torch.utils.data
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.nn import functional as F
import copy

import numpy as np
import matplotlib.pyplot as plt
import math

np.random.seed(42)
torch.manual_seed(42)


# For defining operations on cuda if computer has it available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

"""
Models and loss functions
"""

# Define the Neural Network Class for learning diffusion net
class DDnet(nn.Module):
    def __init__(self, input_shape, hidden_dims, latent_dim, 
                 activation = nn.Tanh()):
        super(DDnet, self).__init__()
        
        self.input_shape = input_shape
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        
        modules = []
        in_channels = np.prod(input_shape)
        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(in_channels, h_dim),
                    activation)
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.latent_layer = nn.Linear(hidden_dims[-1], latent_dim)
        
        # self.weight_init()
        
    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)
        
    def forward(self, features):
        h = self.encoder(features)
        h = self.latent_layer(h)
        return h
    
    
class DDnet_inv(nn.Module):
    def __init__(self, output_shape, hidden_dims, latent_dim, 
                 activation = nn.Tanh()):
        super(DDnet_inv, self).__init__()
        
        self.out_shape = output_shape
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        
        modules = []
        in_channels = latent_dim
        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(in_channels, h_dim),
                    activation)
            )
            in_channels = h_dim

        self.decoder = nn.Sequential(*modules)
        self.output_layer = nn.Linear(hidden_dims[-1], np.prod(output_shape))
        
        # self.weight_init()
        
    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)
        
    def forward(self, features):
        h = self.decoder(features)
        return self.output_layer(h)

    
class Manifold_learner(nn.Module):
    def __init__(self, input_shape, hidden_dims, output_dim, 
                 activation = nn.Tanh()):
        # activation = nn.GELU()
        super(Manifold_learner, self).__init__()
        
        self.input_shape = input_shape
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.final_act = nn.Sigmoid()
        
        modules = []
        in_channels = np.prod(input_shape)
        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(in_channels, h_dim),
                    activation)
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)
        
        # self.weight_init()
        
    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)
        
    def forward(self, features):
        h = self.encoder(features)
        out = self.output_layer(h)
        return self.final_act(out)
    
class CV_learner(nn.Module):
    def __init__(self, input_shape, hidden_dims, latent_dim, 
                 activation = nn.ELU()):
        # nn.ELU(alpha = 1.0)
        
        super(CV_learner, self).__init__()
        
        self.input_shape = input_shape
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        
        modules = []
        in_channels = np.prod(input_shape)
        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(in_channels, h_dim),
                    activation)
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.latent_layer = nn.Linear(hidden_dims[-1], latent_dim)
        
        # self.weight_init()
        
        dmodules = []
        hidden_dims.reverse()
        in_dim = latent_dim
        for i in range(len(hidden_dims)):
            dmodules.append(
                nn.Sequential(
                    nn.Linear(in_dim,
                            hidden_dims[i]),
                    activation)
            )
            in_dim = hidden_dims[i]


        self.decoder = nn.Sequential(*dmodules)
        
        self.final_layer = nn.Linear(hidden_dims[-1],3)
        
    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)
        
    def forward(self, features):
        h = self.encoder(features)
        latent = self.latent_layer(h)
        # deno = torch.norm(latent, dim = 1)
        # latent = latent/deno.reshape(-1,1)
        
        d = self.decoder(latent)
        reconstructed = self.final_layer(d)
        return latent, reconstructed 
    
def CV_loss(CV,x,psi, xhat, theta1, theta2):
    n = len(x)
    if torch.Tensor.dim(CV) == 1:
        latent_dim = 1
    else:
        latent_dim = CV.shape[1]
        
    input_dim = x.shape[1]
    
    x_grad = torch.zeros((n,latent_dim,input_dim))
    
    for i in range(latent_dim):
        gi = torch.autograd.grad(CV[:,i],x,allow_unused=True, retain_graph=True, 
                                              grad_outputs = torch.ones_like(CV[:,i]), create_graph=True)[0]
       
        x_grad[:,i,:] = gi
        
    grad_V1 = torch.autograd.grad(psi,x,allow_unused=True, retain_graph=True, 
                                              grad_outputs = torch.ones_like(psi), create_graph=True)[0]
    
    # print(x_grad)
    # print(grad_V1)
    product = torch.einsum('ijk,ik->ij',x_grad,grad_V1)
    deno = torch.linalg.matrix_norm(x_grad)*torch.norm(grad_V1, dim = 1)
    loss_ol = 0.5*theta1*torch.sum(torch.norm(product, dim = 1)/deno)
    # loss_ol = 0.5*theta1*torch.sum(torch.norm(product, p = 'fro', dim = 1))
    
    
    if theta2 > 0.:
        Jaco_prod = torch.einsum('bij,bjk->bik', x_grad,torch.transpose(x_grad, 1, 2))
        loss_reg = theta2 * torch.sum(torch.linalg.matrix_norm(Jaco_prod - torch.eye(latent_dim)))
        
    else:
        loss_reg = torch.tensor([0.])
        
    recon_loss = torch.mean(torch.norm(xhat - psi, dim = 1))
        
    return loss_ol+loss_reg + recon_loss, loss_ol, recon_loss#loss_reg