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
import matplotlib.pyplot as plt

import utils_LJ7, utils, models, utils_LJ8

"""
This file contains the code to train diffusion net
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


def main_DDnet():
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    example = "LJ7"
    generate = True
    knn = 
    epsilon = 1.0
    
    output_folder = f"../{example}/model_output/"
    data_folder = f"../{example}/data/"
    
    import re
    sigma = 0.02
    fname = f'../{example}/data/ConfBins_data_CrdNum.csv'
    

    data = np.loadtxt(fname,delimiter=",")

    # ======== transformed data ======== 
    train_data = torch.from_numpy(data).to(torch.float32)

    target = np.loadtxt(f"../{example}/data/{example}_eigens_knn{knn}_eps{epsilon}_CrdNum.csv", delimiter=',').astype(np.float32)

    target_data = torch.from_numpy(target)    
    
    train_ds = TensorDataset(train_data,target_data)
    batch_size = int(len(train_data)/125)
    train_loader = DataLoader(dataset=train_ds,
                          batch_size=128,shuffle=True) 
    
    z_dim = target_data.shape[1]
    print('target dimension: ', z_dim)
    # hidden = [45,45]
    hidden = [45,30,25]
    model = models.DDnet(train_data.shape[1],hidden,z_dim,activation = nn.ELU())
    model = model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=5e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.9)
    loss_fn = nn.L1Loss()
    # loss_fn = nn.MSELoss()
    
    for epoch in range(4000):
        model.train()
        i = 0
        for x,psi in train_loader:
            i += 1
            batch_time = AverageMeter()
            losses = AverageMeter()
            end = time.time()
            x = x.to(device)
            psi = psi.to(device)

            eigen = model(x)
            loss = loss_fn(eigen,psi)

            # compute gradient and update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        if epoch%50 == 0:

            reduced_loss = loss.data
            losses.update(float(reduced_loss), x.size(0))
            # torch.cuda.synchronize()
            batch_time.update((time.time() - end)/15)
            end = time.time()

            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.6f} ({loss.avg:.6f})\t'.format(
                        epoch, i, len(train_loader),
                        batch_time=batch_time,
                        loss=losses))
            torch.save(model, output_folder + f"{example}_DDnet_knn{knn}_eps{epsilon}_BinsConf_CrdNum.pt")

    torch.save(model, output_folder + f"{example}_DDnet_knn{knn}_eps{epsilon}_BinsConf_CrdNum.pt")
    
    
        
if __name__ == '__main__':
    
    main_DDnet()        