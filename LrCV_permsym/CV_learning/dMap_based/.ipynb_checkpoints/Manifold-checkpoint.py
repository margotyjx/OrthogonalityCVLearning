import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
# from torchvision import datasets
# from torchvision import transforms as T
import torch.distributed as dist
import torch.optim as optim
# from torch.nn.parallel import DistributedDataParallel as DDP
# from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.nn import functional as F
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

import utils, utils_LJ7, models

"""
This file contains the code to learn the manifold
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


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    example = "LJ7"
    generate = True
    knn = 
    epsilon = 1.0
    
    output_folder = f"../{example}/model_output/"
    data_folder = f"../{example}/data/"
    

    import re
    sigma = 0.02
    fname = f"../{example}/data/{example}_eigens_knn{knn}_eps{epsilon}_CrdNum.csv"
    
    z_data = torch.from_numpy(np.loadtxt(fname,delimiter=",")).to(torch.float32)

    beta = 10.
    
    
    
    if generate:
        ndim = 3
        rad = 0.05
        rtol = 0.005
        target_num = 5000
        random_pts = utils.gen_points(ndim, rad, z_data, rtol, target_num)
        np.savetxt(data_folder + f"ptsCloud_knn{knn}_eps{epsilon}_CrdNum.csv",random_pts,delimiter=",")
    else:
        random_pts = np.loadtxt(data_folder + f"ptsCloud_knn{knn}_eps{epsilon}_CrdNum.csv",delimiter=",")
        pass
        

    Cloud = torch.from_numpy(random_pts.astype(np.float32)).to(device)
    batch_size = int(len(z_data)/125)

    z_data = z_data.to(device)
    Cloud = Cloud.to(device)
    train_loader_on = DataLoader(dataset=z_data,
                          batch_size=batch_size,shuffle=True) 
    train_loader_off = DataLoader(dataset=Cloud,
                          batch_size=batch_size,shuffle=True)
    
    
    theta1 = 1.
    theta2 = 0.002
    
    manifold = models.Manifold_learner(z_data.shape[1], [45,30], 1)
    manifold = manifold.to(device)

    optimizer = optim.Adam(manifold.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[500,1000,1250,2000,3000], gamma=0.5)
    
    for epoch in range(1500):
        manifold.train()
        i = 0
        batch_time = AverageMeter()
        losses = AverageMeter()
        onLosses = AverageMeter()
        offLosses = AverageMeter()
        for x_on in train_loader_on:
            i += 1
            optimizer.zero_grad()
            x_on = x_on.to(device)
            Manifold_on = manifold(x_on)
            onLoss = theta1 * torch.mean(Manifold_on**2)
            
            loss = onLoss

            # compute gradient and update
            loss.backward(retain_graph=True)
            optimizer.step()

            
        for x_off in train_loader_off:
            i += 1
            optimizer.zero_grad()
            end = time.time()

            x_off = x_off.to(device)
            
            Manifold_off = manifold(x_off)
            offLoss = theta2 * torch.mean(1/Manifold_off**2)
            
            loss = offLoss

            # compute gradient and update
            loss.backward(retain_graph=True)
            optimizer.step()
            # scheduler.step()

        if epoch%50 == 0:

            reduced_loss = loss.data
            reduced_on = onLoss.data
            reduced_off = offLoss.data
            losses.update(float(reduced_loss), x_on.size(0))
            onLosses.update(float(reduced_on), x_on.size(0))
            offLosses.update(float(reduced_off), x_on.size(0))

            # torch.cuda.synchronize()
            batch_time.update((time.time() - end)/15)
            end = time.time()

            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.6f} ({loss.avg:.6f})\t'
                  'on loss {onLosses.avg:.6f}\t'
                  'off loss {offLosses.avg:.6f}\t'.format(
                        epoch, i, len(train_loader_on),
                        batch_time=batch_time,
                        loss=losses,onLosses = onLosses,
                        offLosses = offLosses))
            torch.save(manifold, output_folder + f"{example}_manifold_learner_knn{knn}_eps{epsilon}_CrdNum.pt")
    
    torch.save(manifold, output_folder + f"{example}_manifold_learner_knn{knn}_eps{epsilon}_CrdNum.pt")

    
if __name__ == '__main__':
    
    main() 
    