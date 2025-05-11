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


def main(example):

    if example == 'LJ7':
        output_folder = '/Users/jiaxinyuan/CV_learning/LJ7/model_output/'
        data_folder = "/Users/jiaxinyuan/CV_learning/LJ7/data/"
        
        DDnet = torch.load(output_folder + 'LJ7_DDnet_r2.pt')
        
        import re
        sigma = 0.02
        fname = '/Users/jiaxinyuan/CV_learning/LJ7/data/LJ7delta_net.txt'

        LJtraj14D = [] 
        with open(fname, "r") as f:
            for line in f:
                # cleaning the bad chars in line
                line = line.strip()
                line = line.strip(" \\n")
                line = re.sub(r"(-[0-9]+\.)", r" \1", line)
                values = [float(value) for value in line.split()]
                LJtraj14D.append(values)
        data = np.array(LJtraj14D).astype(np.float32)
        print('adjusted data shape: ',np.shape(data))
        
        # data = np.loadtxt(data_folder +'LJ7_Original_Data_1.csv', delimiter=',').astype(np.float32)
        
        coord_data = utils.sort_r2_pt(torch.from_numpy(data))
        # coord_data = torch.from_numpy(data)
        z_data = DDnet(coord_data)
        np.savetxt(data_folder + "LJ7_DDnet_z.csv", z_data.detach(), delimiter=",")
        beta = 10
        
    elif example == 'butane':
        output_folder = '/Users/jiaxinyuan/CV_learning/butane/model_output/'
        beta = 2.49
    
    generate = False
    
    if generate:
        ndim = 3
        rad = 0.06
        rtol = 0.005
        target_num = 5000
        random_pts = utils.gen_points(ndim, rad, z_data, rtol, target_num)
        np.savetxt(data_folder + "ptsCloud_r2.csv",random_pts,delimiter=",")
    else:
        random_pts = np.loadtxt(data_folder + "ptsCloud_r2.csv",delimiter=",")

    Cloud = torch.from_numpy(random_pts.astype(np.float32))
    batch_size = int(len(z_data)/125)
    # train_loader_on = DataLoader(dataset=z_data,
    #                       batch_size=128,shuffle=True) 
    # train_loader_off = DataLoader(dataset=Cloud,
    #                       batch_size=128,shuffle=True)
    """
    labels for the points NOT on the manifold
    """
    labels_on = torch.zeros(len(z_data),dtype=torch.long)
    labels_off = torch.ones(len(Cloud),dtype=torch.long)
    
    input_data = torch.cat((z_data, Cloud), dim = 0)
    labels = torch.cat((labels_on, labels_off), dim = 0)
    
    train_ds = TensorDataset(input_data, labels)
    train_loader = DataLoader(dataset=train_ds,
                          batch_size=128,shuffle=True) 
    lossfn = nn.CrossEntropyLoss()
    
    theta1 = 1
    theta2 = 0.002
    
    manifold = models.Manifold_learner(z_data.shape[1], [45,30], 2)
    optimizer = optim.Adam(manifold.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1000,2000,3000], gamma=0.5)
    # optimizer = optim.Adam(manifold.parameters(), lr=1e-4, weight_decay = 1e-5)
    
    for epoch in range(3000):
        # ddp_model.train()
        manifold.train()
        i = 0
        losses = AverageMeter()
        for x, label in train_loader:
            i += 1
            optimizer.zero_grad()
            Manifold = manifold(x)
            # manifold_label = torch.cat((Manifold,1.- Manifold), dim = 1)
            
            # print(manifold_label.dtype)
            # print(label.dtype)
            
            loss = lossfn(Manifold,label)

            # compute gradient and do SGD step
            loss.backward(retain_graph=True)
            optimizer.step()


        if epoch%50 == 0:

            reduced_loss = loss.data
            losses.update(float(reduced_loss), x.size(0))

            # torch.cuda.synchronize()

            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.6f} ({loss.avg:.6f})\t'.format(
                        epoch, i, len(train_loader),
                        loss=losses))
            torch.save(manifold, output_folder + 'LJ7_manifold_learner_r2_cls_epoch_'+str(epoch)+'.pt')
    
    if example == 'LJ7':
        torch.save(manifold, output_folder + 'LJ7_manifold_learner_r2_cls.pt')
    else:
        torch.save(manifold, output_folder + 'butane_manifold_learner.pt')

if __name__ == '__main__':
    
    main('LJ7') 

