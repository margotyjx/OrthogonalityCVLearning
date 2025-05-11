import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset


class LJ8_3(nn.Module):
    def __init__(self, in_size, hidden_sizes, out_size):
        super().__init__()
        layers = [nn.Linear(in_size, hidden_sizes[0]), nn.ReLU()]
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_sizes[-1], out_size))
        layers.append(nn.Sigmoid())
        self.model = nn.Sequential(*layers)

    def forward(self, xb):
        return self.model(xb)


def main(args):
    beta = args.beta
    method = args.method
    load = args.load

    method_to_dir = {
        "LDA12": "CV_LDA12_SortCNum",
        "LDA23": "CV_LDA23_SortCNum",
        "MLCV": "CV_OrderedCoordNum",
        "mu2mu3": "mu2mu3",
        "CV_OrderedCoordNum2": "CV_OrderedCoordNum2"
    }
    directory = method_to_dir.get(method, None)
    if directory is None:
        raise ValueError(f"Unknown method: {method}")

    fname = os.path.join(directory, f"Data/Committor_{method}_BETA{beta}.npz")
    save_folder = os.path.join(directory, f"Data/Committor_{method}_BETA{beta}")
    os.makedirs(save_folder, exist_ok=True)

    inData = np.load(fname)
    pts = inData["points"]
    Q = inData["committor"]
    tri = inData['tri']
    CVlist = inData['CVlist']

    train_data = torch.tensor(pts, dtype=torch.float32)
    Q = torch.tensor(Q, dtype=torch.float32)

    dirname = f"FEMdataBETA{beta}/"

    if method == 'LDA12':
        ptsA = torch.tensor(np.loadtxt(dirname + "/ptsA.csv", delimiter=','), dtype=torch.float32)
        ptsB = [torch.tensor(np.loadtxt(dirname + f"/ptsB{i}.csv", delimiter=','), dtype=torch.float32) for i in range(1, 5)]
        train_data = torch.cat([train_data, ptsA] + ptsB, 0)
        Q = torch.cat([Q, torch.zeros(len(ptsA))] + [torch.ones(len(pb)) for pb in ptsB], 0)
    else:
        ptsA = torch.tensor(np.loadtxt(dirname + "ptsA.csv", delimiter=','), dtype=torch.float32)
        ptsB = torch.tensor(np.loadtxt(dirname + "ptsB.csv", delimiter=','), dtype=torch.float32)
        train_data = torch.cat((train_data, ptsA, ptsB), 0)
        Q = torch.cat((Q, torch.zeros(len(ptsA)), torch.ones(len(ptsB))), 0)

    train_data.requires_grad_(True)

    model = LJ8_3(2, [40, 40], 1)
    train_ds = TensorDataset(train_data, Q)
    train_dl = DataLoader(train_ds, batch_size=256, shuffle=True)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.7)

    if load:
        for name, param in model.named_parameters():
            shape_str = ",".join(map(str, param.shape))
            filename = os.path.join(save_folder, f"{name.replace('.', '_')}_[{shape_str}].csv")
            if os.path.exists(filename):
                param_data = np.loadtxt(filename, delimiter=",")
                param_data = torch.tensor(param_data, dtype=param.dtype)
                if param.shape != param_data.shape:
                    if len(param.shape) == 2 and len(param_data.shape) == 1:
                        param_data = param_data.view(param.shape)
                    elif len(param.shape) == 1 and param_data.numel() == 1:
                        param_data = param_data.view(1)
                    else:
                        continue
                if param.shape == param_data.shape:
                    param.data.copy_(param_data)
                    print(f"Loaded: {filename}")
        model.eval()
    else:
        for epoch in range(1000):
            for X, y in train_dl:
                optimizer.zero_grad()
                Q_pred = model(X)
                loss = loss_fn(Q_pred.squeeze(), y)
                loss.backward()
                optimizer.step()
            scheduler.step()
            if epoch % 25 == 0:
                print(f"Epoch: {epoch}, Loss: {loss.item():.4f}")
        for name, param in model.named_parameters():
            shape_str = ",".join(map(str, param.shape))
            filename = os.path.join(save_folder, f"{name.replace('.', '_')}_[{shape_str}].csv")
            np.savetxt(filename, param.detach().cpu().numpy(), delimiter=',')
            print(f"Saved: {filename}")

    titles = {
        "CV_OrderedCoordNum2": (r'CV sort[c], NN', r"CV1, sort[c]", r"CV2, sort[c]"),
        "LDA12": (r'LDA1-2, NN', r"LDA1", r"LDA2"),
        "LDA23": (r'LDA2-3, NN', r"LDA2", r"LDA3"),
        "mu2mu3": (r'$(\mu_2, \mu_3)$, NN', r"$\mu_2$", r"$\mu_3$")
    }
    title, xlabel, ylabel = titles.get(method, ("Committor", "CV1", "CV2"))
    title = r'$\beta = $' + str(beta) + ' ' + title

    q = model(torch.tensor(pts, dtype=torch.float32)).detach()
    plt.figure(figsize=(8, 7))
    plt.rcParams.update({'font.size': 20})
    plt.tricontourf(pts[:, 0], pts[:, 1], tri, q[:, 0], np.linspace(0, 1, 11))
    plt.colorbar(label="Committor", orientation="vertical")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    os.makedirs(os.path.join(directory, "Figures"), exist_ok=True)
    fig_path = os.path.join(directory, f"Figures/NN_LJ8_committor_{method}_BETA{beta}.pdf")
    plt.savefig(fig_path)
    print(f"Saved plot to {fig_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or load LJ8 committor model")
    parser.add_argument("--beta", type=int, default=9, help="Beta value")
    parser.add_argument("--method", type=str, default="CV_OrderedCoordNum2",
                        choices=["CV_OrderedCoordNum2", "LDA12", "LDA23", "mu2mu3"], help="Method type")
    parser.add_argument("--load", action="store_true", help="Load saved model weights")
    args = parser.parse_args()
    main(args)