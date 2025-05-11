import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

import utils_LJ7, utils, models, utils_LJ8


class AverageMeter(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = self.avg = self.sum = self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    example = args.example
    read_knn = args.knn
    epsilon = args.epsilon
    hidden = [int(h) for h in args.hidden.split(',')]
    epochs = args.epochs

    output_folder = f"../{example}/model_output/"
    data_folder = f"../{example}/data/"
    os.makedirs(output_folder, exist_ok=True)

    data = np.loadtxt(f"{data_folder}/ConfBins_data_CrdNum.csv", delimiter=",")
    N = data.shape[0]
    train_data = torch.from_numpy(data).to(torch.float32)

    knn = int(read_knn * N) if read_knn < 1. else int(read_knn)
    target = np.loadtxt(f"{data_folder}/{example}_eigens_knn{knn}_eps{epsilon}_CrdNum.csv", delimiter=",").astype(np.float32)
    target_data = torch.from_numpy(target)

    train_ds = TensorDataset(train_data, target_data)
    train_loader = DataLoader(dataset=train_ds, batch_size=128, shuffle=True)

    z_dim = target_data.shape[1]
    print('Target dimension:', z_dim)

    model = models.DDnet(train_data.shape[1], hidden, z_dim, activation=nn.ELU()).to(device)

    optimizer = optim.Adam(model.parameters(), lr=5e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.9)
    loss_fn = nn.L1Loss()

    for epoch in range(epochs):
        model.train()
        i = 0
        for x, psi in train_loader:
            i += 1
            batch_time = AverageMeter()
            losses = AverageMeter()
            end = time.time()

            x = x.to(device)
            psi = psi.to(device)

            eigen = model(x)
            loss = loss_fn(eigen, psi)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        if epoch % 50 == 0:
            reduced_loss = loss.data
            losses.update(float(reduced_loss), x.size(0))
            batch_time.update((time.time() - end) / 15)
            print(f"Epoch: [{epoch}][{i}/{len(train_loader)}]\t"
                  f"Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                  f"Loss {losses.val:.6f} ({losses.avg:.6f})")

            torch.save(model, output_folder + f"{example}_DDnet_knn{knn}_eps{epsilon}_BinsConf_CrdNum.pt")

    torch.save(model, output_folder + f"{example}_DDnet_knn{knn}_eps{epsilon}_BinsConf_CrdNum.pt")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Diffusion Net (DDNet) for LJ system")
    parser.add_argument("--example", type=str, choices=["LJ7", "LJ8"], default="LJ7", help="System type")
    parser.add_argument("--knn", type=float, default=0.25, 
                        help="KNN fraction or absolute value. If less than 1, interpreted as ratio to the total number.")
    parser.add_argument("--epsilon", type=float, default=1.0, help="Epsilon for diffusion map")
    parser.add_argument("--hidden", type=str, default="45,30,25", help="Comma-separated hidden layer sizes")
    parser.add_argument("--epochs", type=int, default=4000, help="Number of training epochs")
    args = parser.parse_args()
    main(args)
