import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import utils, utils_LJ7, models


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
    knn = args.read_knn
    epsilon = args.epsilon
    ndim = args.ndim
    rad = args.rad
    rtol = args.rtol
    target_num = args.target_num
    theta1 = args.theta1
    theta2 = args.theta2
    epochs = args.epochs
    generate = args.generate

    output_folder = f"../{example}/model_output/"
    data_folder = f"../{example}/data/"
    os.makedirs(output_folder, exist_ok=True)

    # Load eigenvectors
    fname = f"{data_folder}/{example}_eigens_knn0_eps{epsilon}_CrdNum.csv"
    z_data = torch.from_numpy(np.loadtxt(fname, delimiter=",")).to(torch.float32)

    N = z_data.shape[0]
    knn = int(read_knn * N) if read_knn < 1. else int(read_knn)

    # Sample or load off-manifold points
    pts_file = data_folder + f"ptsCloud_knn{knn}_eps{epsilon}_CrdNum.csv"
    if generate:
        random_pts = utils.gen_points(ndim, rad, z_data, rtol, target_num)
        np.savetxt(pts_file, random_pts, delimiter=",")
    else:
        random_pts = np.loadtxt(pts_file, delimiter=",")

    Cloud = torch.from_numpy(random_pts.astype(np.float32)).to(device)
    z_data = z_data.to(device)

    batch_size = max(1, int(len(z_data) / 125))
    train_loader_on = DataLoader(dataset=z_data, batch_size=batch_size, shuffle=True)
    train_loader_off = DataLoader(dataset=Cloud, batch_size=batch_size, shuffle=True)

    manifold = models.Manifold_learner(z_data.shape[1], [45, 30], 1).to(device)
    optimizer = optim.Adam(manifold.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[500, 1000, 1250, 2000, 3000], gamma=0.5)

    for epoch in range(epochs):
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
            loss.backward(retain_graph=True)
            optimizer.step()

        for x_off in train_loader_off:
            i += 1
            optimizer.zero_grad()
            end = time.time()
            x_off = x_off.to(device)
            Manifold_off = manifold(x_off)
            offLoss = theta2 * torch.mean(1 / (Manifold_off**2 + 1e-8))  # Add epsilon for stability
            loss = offLoss
            loss.backward(retain_graph=True)
            optimizer.step()

        if epoch % 50 == 0:
            losses.update(loss.item(), x_on.size(0))
            onLosses.update(onLoss.item(), x_on.size(0))
            offLosses.update(offLoss.item(), x_on.size(0))
            batch_time.update((time.time() - end) / 15)

            print(f"Epoch: [{epoch}][{i}/{len(train_loader_on)}]\t"
                  f"Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                  f"Loss {losses.val:.6f} ({losses.avg:.6f})\t"
                  f"on loss {onLosses.avg:.6f}\t"
                  f"off loss {offLosses.avg:.6f}")

            torch.save(manifold, output_folder + f"{example}_manifold_learner_knn{knn}_eps{epsilon}_CrdNum.pt")

    torch.save(manifold, output_folder + f"{example}_manifold_learner_knn{knn}_eps{epsilon}_CrdNum.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train manifold learner")
    parser.add_argument("--example", type=str, choices=["LJ7", "LJ8"], default="LJ7", help="System example")
    parser.add_argument("--knn", type=float, default=0.25, help="KNN value or fraction")
    parser.add_argument("--epsilon", type=float, default=1.0, help="Epsilon value for diffusion map")
    parser.add_argument("--ndim", type=int, default=3, help="Dimensionality of latent space")
    parser.add_argument("--rad", type=float, default=0.05, help="Radius to sample off-manifold points")
    parser.add_argument("--rtol", type=float, default=0.005, help="Tolerance in radius")
    parser.add_argument("--target_num", type=int, default=5000, help="Number of points to sample")
    parser.add_argument("--theta1", type=float, default=1.0, help="Weight for on-manifold loss")
    parser.add_argument("--theta2", type=float, default=0.002, help="Weight for off-manifold loss")
    parser.add_argument("--epochs", type=int, default=1500, help="Number of training epochs")
    parser.add_argument("--generate", action="store_true", help="Whether to generate new off-manifold points")
    args = parser.parse_args()
    main(args)

    