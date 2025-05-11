import os
import sys
import time
# import matplotlib.pyplot as plt

import torch
import numpy as np
import torch.nn as nn
import torch.nn.init as init
# from torchvision import datasets
# from torchvision import transforms as T
import torch.distributed as dist
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.nn import functional as F
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

import utils, utils_LJ7, models
"""
This file is the py file to run the code of learning CV
"""

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

        
def CV_loss(CV,x,psi, xhat, theta1, theta2, device):
    n = len(x)
    if torch.Tensor.dim(CV) == 1:
        latent_dim = 1
    else:
        latent_dim = CV.shape[1]
        
    input_dim = x.shape[1]
    
    x_grad = torch.zeros((n,latent_dim,input_dim)).to(device)
    
    for i in range(latent_dim):
        gi = torch.autograd.grad(CV[:,i],x,allow_unused=True, retain_graph=True, 
                                              grad_outputs = torch.ones_like(CV[:,i]), create_graph=True)[0]
       
        x_grad[:,i,:] = gi
        
    grad_V1 = torch.autograd.grad(psi,x,allow_unused=True, retain_graph=True, 
                                              grad_outputs = torch.ones_like(psi), create_graph=True)[0]
    
    product = torch.einsum('ijk,ik->ij',x_grad,grad_V1)
    deno = torch.linalg.matrix_norm(x_grad)*torch.norm(grad_V1, dim = 1)
    # loss_ol = 0.5*theta1*torch.sum(torch.norm(product, dim = 1)/deno)
    loss_ol = 0.5*theta1*torch.sum(torch.norm(product, p = 'fro', dim = 1)/deno)
    
    
    if theta2 > 0.:
        Jaco_prod = torch.einsum('bij,bjk->bik', x_grad,torch.transpose(x_grad, 1, 2))
        loss_reg = theta2 * torch.sum(torch.linalg.matrix_norm(Jaco_prod - torch.eye(latent_dim).to(Jaco_prod.device)))
        
    else:
        loss_reg = torch.tensor([0.]).to(device)
        
    recon_loss = torch.mean(torch.norm(xhat - psi, dim = 1))
        
    return loss_ol+loss_reg + recon_loss, loss_ol, recon_loss#loss_reg


def main():
    
    # parallel computing stuff
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ## load data here
    
    example = "LJ7"
    knn = 
    epsilon = 1.0
    use_C = True
    MEP_loss = False
    
    output_folder = f'../{example}/model_output/'
    data_folder = f'../{example}/data/'
    DDnet = torch.load(output_folder + f"{example}_DDnet_knn{knn}_eps{epsilon}_BinsConf_CrdNum.pt").to(device)
    manifold = torch.load(output_folder + f"{example}_manifold_learner_knn{knn}_eps{epsilon}_CrdNum.pt").to(device)

    import re
    sigma = 0.02
    fname = data_folder + 'LJ7bins_confs.txt'

    LJtraj = [] 
    with open(fname, "r") as f:
        for line in f:
            # cleaning the bad chars in line
            line = line.strip()
            line = line.strip(" \\n")
            line = re.sub(r"(-[0-9]+\.)", r" \1", line)
            values = [float(value) for value in line.split()]
            LJtraj14D.append(values)
    data = np.array(LJtraj).astype(np.float32)
    print('adjusted data shape: ',np.shape(data))

    coord_data = torch.from_numpy(data).to(device)
    
    if MEP_loss:
        MEP = np.loadtxt(data_folder + 'MEP.txt', delimiter=',').astype(np.float32)
        Arc_len = torch.from_numpy(np.loadtxt(data_folder + 'arc_len_MEP.txt', delimiter=',').astype(np.float32).reshape(-1,1)).to(device)
    
    if example == "LJ7":
        # different transformation of the input
        if use_C:
            C_vec = torch.from_numpy(utils_LJ7.C_np_batch_LJ7(data).astype(np.float32)).to(device)
            if MEP_loss:
                C_mep = torch.from_numpy(utils_LJ7.C_np_batch_LJ7(MEP).astype(np.float32)).to(device)
                C_mep.requires_grad_(True)
        else:
            C_vec = torch.from_numpy(utils_LJ7.sort_r2(data).astype(np.float32)).to(device)
            if MEP_loss:
                C_mep = torch.from_numpy(utils_LJ7.sort_r2(MEP).astype(np.float32)).to(device)
                C_mep.requires_grad_(True)
    elif example == "LJ8":
        C_vec = torch.from_numpy(utils_LJ8.C_np_batch_LJ8(data).astype(np.float32)).to(device)
        if MEP_loss:
            C_mep = torch.from_numpy(utils_LJ8.C_np_batch_LJ8(MEP).astype(np.float32)).to(device)
            C_mep.requires_grad_(True)
    
    C_vec.requires_grad_(True)
    train_loader = DataLoader(dataset=C_vec,
                          batch_size=1024,shuffle=True) 
    
    theta1 = torch.tensor(1.).to(device)
    theta2 = torch.tensor(0.2).to(device)
    
    
    z_dim = 2
    hidden = [45,45]
    model = models.CV_learner(C_vec.shape[1], hidden, z_dim)
    model = model.to(device)
    

    learning_rate = 0.01*float(128)/256.
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    
    for epoch in range(1500):
        model.train()
        i = 0
        for x in train_loader:
            i += 1
            batch_time = AverageMeter()
            reg_losses = AverageMeter()
            ol_losses = AverageMeter()
            losses = AverageMeter()
            end = time.time()
            
            CV,xhat = model(x)
            z = DDnet(x)
            psi = manifold(z)
            
            loss, loss_ol, loss_reg = CV_loss(CV, x, psi, xhat, theta1, theta2, device)
            
            if MEP_loss:
                cv_mep, mep_hat = model(C_mep)

                mep_loss = torch.norm(cv_mep[:,0].reshape(-1) - Arc_len.reshape(-1))

                loss = loss + 10.*mep_loss

            # compute gradient and update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        if epoch%50 == 0:

            reduced_loss = loss.data
            reduced_reg = loss_reg.data
            reduced_ol = loss_ol.data
            losses.update(float(reduced_loss), x.size(0))
            reg_losses.update(float(reduced_reg), x.size(0))
            ol_losses.update(float(reduced_ol), x.size(0))
            # torch.cuda.synchronize()
            batch_time.update((time.time() - end)/15)
            end = time.time()

            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.5f} ({loss.avg:.4f})\t'
                  'reconstruction loss {reg_losses.avg:.4f}\t'
                  'loss for orthogonality {ol_losses.avg:.4f}\t'.format(
                        epoch, i, len(train_loader),
                        batch_time=batch_time,
                        loss=losses,
                        reg_losses = reg_losses,
                  ol_losses = ol_losses))
            
            torch.save(model, output_folder + f'{example}_CVlearner_2D_crdnum_{use_C}_mep_{MEP_loss}.pt')
                
    torch.save(model, output_folder + f'{example}_CVlearner_2D_crdnum_{use_C}_mep_{MEP_loss}.pt')
        
if __name__ == '__main__':
    
    main()        