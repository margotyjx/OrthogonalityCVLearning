import os
import time
import re
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import utils, utils_LJ7, utils_LJ8, models


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


def CV_loss(CV, x, psi, xhat, theta1, theta2, device):
    n = len(x)
    latent_dim = 1 if CV.dim() == 1 else CV.shape[1]
    input_dim = x.shape[1]
    x_grad = torch.zeros((n, latent_dim, input_dim)).to(device)

    for i in range(latent_dim):
        gi = torch.autograd.grad(CV[:, i], x, allow_unused=True, retain_graph=True,
                                 grad_outputs=torch.ones_like(CV[:, i]), create_graph=True)[0]
        x_grad[:, i, :] = gi

    grad_V1 = torch.autograd.grad(psi, x, allow_unused=True, retain_graph=True,
                                  grad_outputs=torch.ones_like(psi), create_graph=True)[0]

    product = torch.einsum('ijk,ik->ij', x_grad, grad_V1)
    deno = torch.linalg.matrix_norm(x_grad) * torch.norm(grad_V1, dim=1)
    loss_ol = 0.5 * theta1 * torch.sum(torch.norm(product, p='fro', dim=1) / deno)

    if theta2 > 0.:
        Jaco_prod = torch.einsum('bij,bjk->bik', x_grad, torch.transpose(x_grad, 1, 2))
        loss_reg = theta2 * torch.sum(torch.linalg.matrix_norm(Jaco_prod - torch.eye(latent_dim).to(Jaco_prod.device)))
    else:
        loss_reg = torch.tensor([0.]).to(device)

    recon_loss = torch.mean(torch.norm(xhat - psi, dim=1))
    return loss_ol + loss_reg + recon_loss, loss_ol, recon_loss


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    example = args.example
    knn_read = args.knn
    epsilon = args.epsilon
    use_C = args.use_C
    MEP_loss = args.mep_loss
    hidden = [int(h) for h in args.hidden.split(",")]
    theta1 = torch.tensor(args.theta1).to(device)
    theta2 = torch.tensor(args.theta2).to(device)
    output_folder = f'../{example}/model_output/'
    data_folder = f'../{example}/data/'

    fname = data_folder + f'{example}bins_confs.txt'
    LJtraj = []
    with open(fname, "r") as f:
        for line in f:
            line = line.strip().strip(" \\n")
            line = re.sub(r"(-[0-9]+\.)", r" \1", line)
            values = [float(value) for value in line.split()]
            LJtraj.append(values)
    data = np.array(LJtraj).astype(np.float32)
    print('adjusted data shape:', data.shape)
    
    N = data.shape[0]
    
    if read_knn < 1.:
        knn = int(read_knn * N) 
    else:
        knn = int(read_knn)
    
    DDnet = torch.load(output_folder + f"{example}_DDnet_knn{knn}_eps{epsilon}_BinsConf_CrdNum.pt").to(device)
    manifold = torch.load(output_folder + f"{example}_manifold_learner_knn{knn}_eps{epsilon}_CrdNum.pt").to(device)
    

    coord_data = torch.from_numpy(data).to(device)

    if MEP_loss:
        MEP = np.loadtxt(data_folder + 'MEP.txt', delimiter=',').astype(np.float32)
        Arc_len = torch.from_numpy(
            np.loadtxt(data_folder + 'arc_len_MEP.txt', delimiter=',').astype(np.float32).reshape(-1, 1)
        ).to(device)

    if example == "LJ7":
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
    train_loader = DataLoader(dataset=C_vec, batch_size=1024, shuffle=True)

    z_dim = 2 # want to learn 2D CV
    model = models.CV_learner(C_vec.shape[1], hidden, z_dim).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    for epoch in range(args.epochs):
        model.train()
        i = 0
        for x in train_loader:
            i += 1
            batch_time = AverageMeter()
            reg_losses = AverageMeter()
            ol_losses = AverageMeter()
            losses = AverageMeter()
            end = time.time()

            CV, xhat = model(x)
            z = DDnet(x)
            psi = manifold(z)

            loss, loss_ol, loss_reg = CV_loss(CV, x, psi, xhat, theta1, theta2, device)

            if MEP_loss:
                cv_mep, mep_hat = model(C_mep)
                mep_loss = torch.norm(cv_mep[:, 0].reshape(-1) - Arc_len.reshape(-1))
                loss += 10. * mep_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        if epoch % 50 == 0:
            losses.update(loss.item(), x.size(0))
            reg_losses.update(loss_reg.item(), x.size(0))
            ol_losses.update(loss_ol.item(), x.size(0))
            batch_time.update((time.time() - end) / 15)
            end = time.time()

            print(f'Epoch: [{epoch}][{i}/{len(train_loader)}]\t'
                  f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  f'Loss {losses.val:.5f} ({losses.avg:.4f})\t'
                  f'Reconstruction loss {reg_losses.avg:.4f}\t'
                  f'Orthogonality loss {ol_losses.avg:.4f}')

            torch.save(model, output_folder + f'{example}_CVlearner_2D_crdnum_{use_C}_mep_{MEP_loss}.pt')

    torch.save(model, output_folder + f'{example}_CVlearner_2D_crdnum_{use_C}_mep_{MEP_loss}.pt')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CV Learner model")
    parser.add_argument("--example", type=str, choices=["LJ7", "LJ8"], default="LJ7", help="System to use")
    parser.add_argument("--knn", type=float, default = 0.25,
                        help="KNN used for loading model. If less than 1, interpreted as ratio to the total number.")
    parser.add_argument("--epsilon", type=float, default=1.0, help="Epsilon for diffusion map")
    parser.add_argument("--use_C", action="store_true", help="Use coordination number input")
    parser.add_argument("--mep_loss", action="store_true", help="Use MEP loss")
    parser.add_argument("--hidden", type=str, default="45,45", help="Hidden layer sizes, comma separated")
    parser.add_argument("--theta1", type=float, default=1.0, help="Weight for orthogonality loss")
    parser.add_argument("--theta2", type=float, default=0.2, help="Weight for Jacobian regularization")
    parser.add_argument("--epochs", type=int, default=1500, help="Weight for Jacobian regularization")
    args = parser.parse_args()
    main(args)
