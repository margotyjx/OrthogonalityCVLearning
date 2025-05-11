import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import argparse


class LJ7_2(nn.Module):
    def __init__(self, in_size, hidden_size, hidden_size2, out_size):
        super().__init__()
        self.linear1 = nn.Linear(in_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size2)
        self.linear3 = nn.Linear(hidden_size2, hidden_size2)
        self.linear4 = nn.Linear(hidden_size2, out_size)

    def forward(self, xb):
        relu = nn.ReLU()
        out = relu(self.linear1(xb))
        out = relu(self.linear2(out))
        out = relu(self.linear3(out))
        out = torch.sigmoid(self.linear4(out))
        return out


def main(args):
    beta = args.beta
    method = args.method
    load = args.load

    if method == "SortedDistSquared":
        directory = "CV_SortedDistSquared"
    elif method == "SortCNum":
        directory = "CV_OrderedCoordNum"
    elif method == "mu2mu3":
        directory = "mu2mu3"
    else:
        raise ValueError(f"Unknown method: {method}")

    fname = os.path.join(directory, f"Data/Committor_LJ7_{method}_BETA{beta}.npz")
    save_folder = os.path.join(directory, f"Data/Committor_LJ7_{method}_BETA{beta}")
    os.makedirs(save_folder, exist_ok=True)

    inData = np.load(fname)
    pts = inData["points"]
    Q = inData["committor"]
    tri = inData["tri"]
    CVlist = inData["CVlist"]

    train_data = torch.tensor(pts, dtype=torch.float32)
    Q = torch.tensor(Q, dtype=torch.float32)

    dirname = f"FEMdataBETA{beta}/"
    ptsA_file = "LJ7_ptsA.csv" if method == "mu2mu3" else "ptsA.csv"
    ptsB_file = "LJ7_ptsB.csv" if method == "mu2mu3" else "ptsB.csv"

    ptsA = torch.tensor(np.loadtxt(os.path.join(dirname, ptsA_file), delimiter=','), dtype=torch.float32)
    ptsB = torch.tensor(np.loadtxt(os.path.join(dirname, ptsB_file), delimiter=','), dtype=torch.float32)

    train_data = torch.cat((train_data, ptsA, ptsB), dim=0)
    Q = torch.cat((Q, torch.zeros(len(ptsA)), torch.ones(len(ptsB))), dim=0)
    train_data.requires_grad_(True)

    input_size = 2
    output_size = 1
    model = LJ7_2(input_size, 40, 40, output_size)

    train_ds = TensorDataset(train_data, Q)
    train_dl = DataLoader(train_ds, batch_size=256, shuffle=True)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.7)

    if load:
        for name, param in model.named_parameters():
            shape_str = ",".join(map(str, param.shape))
            param_path = os.path.join(save_folder, f"{name.replace('.', '_')}_[{shape_str}].csv")

            if os.path.exists(param_path):
                param_data = np.loadtxt(param_path, delimiter=",")
                param_tensor = torch.tensor(param_data, dtype=param.dtype)

                if param.shape != param_tensor.shape:
                    if len(param.shape) == 2 and len(param_tensor.shape) == 1:
                        param_tensor = param_tensor.view(param.shape)
                    elif len(param.shape) == 1 and param_tensor.numel() == 1:
                        param_tensor = param_tensor.view(1)
                    else:
                        print(f"Skipping {name} due to incompatible shape")
                        continue

                if param.shape == param_tensor.shape:
                    param.data.copy_(param_tensor)
                    print(f"Loaded: {param_path}")
            else:
                print(f"File not found: {param_path}")

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
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

        for name, param in model.named_parameters():
            shape_str = ",".join(map(str, param.shape))
            param_path = os.path.join(save_folder, f"{name.replace('.', '_')}_[{shape_str}].csv")
            np.savetxt(param_path, param.detach().cpu().numpy(), delimiter=",")
            print(f"Saved: {param_path}")

    # Plotting
    q = model(torch.tensor(pts, dtype=torch.float32)).detach().numpy()

    if method == "SortedDistSquared":
        title = r'$\beta = $' + str(beta) + r'sort[$d^2$]' + ', NN'
        xlabel = r"CV1, sort[$d^2$]"
        ylabel = r"CV2, sort[$d^2$]"
    elif method == "SortCNum":
        title = r'$\beta = $' + str(beta) + ' sort[c], NN'
        xlabel = r"CV1, sort[c]"
        ylabel = r"CV2, sort[c]"
    elif method == "mu2mu3":
        title = r'$\beta = $' + str(beta) + r' $(\mu_2, \mu_3)$, NN'
        xlabel = r"$\mu_2$"
        ylabel = r"$\mu_3$"

    plt.figure(figsize=(8, 7))
    plt.rcParams.update({'font.size': 20})
    plt.tricontourf(pts[:, 0], pts[:, 1], tri, q[:, 0], np.linspace(0, 1, 11))
    plt.colorbar(label="Committor", orientation="vertical")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    fig_path = os.path.join(directory, f"Figures/NN_LJ72D_committor_{method}_BETA{beta}.pdf")
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    plt.savefig(fig_path)
    print(f"Saved plot: {fig_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or load neural network to predict committor function.")
    parser.add_argument("--beta", type=int, default=9, help="Beta value for the system")
    parser.add_argument("--method", type=str, choices=["SortedDistSquared", "SortCNum", "mu2mu3"],
                        default="SortedDistSquared", help="Method for coordinate transformation")
    parser.add_argument("--load", action="store_true", help="Whether to load model from saved weights")

    args = parser.parse_args()
    main(args)
