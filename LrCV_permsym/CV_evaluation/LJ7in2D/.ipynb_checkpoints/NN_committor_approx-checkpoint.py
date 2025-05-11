import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
from pathlib import Path
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.optim as optim
import os

class LJ7_2(nn.Module):
    """Feedfoward neural network with 2 hidden layer"""
    def __init__(self, in_size, hidden_size,hidden_size2, out_size):
        super().__init__()
        # 1st hidden layer
        self.linear1 = nn.Linear(in_size, hidden_size)
        # 2nd hidden layer
        self.linear2 = nn.Linear(hidden_size,hidden_size2)
        self.linear3 = nn.Linear(hidden_size2,hidden_size2)
        # output layer
        self.linear4 = nn.Linear(hidden_size2, out_size)
        
    def forward(self, xb):
        # Get information from the data
        # Get intermediate outputs using hidden layer
        out = self.linear1(xb)
        # Apply activation function
        relu = nn.ReLU()
        out = relu(out)
        # Get predictions using output layer
        out = self.linear2(out)
        # apply activation function again
        out = relu(out)
        # last hidden layer 
        out = self.linear3(out)
        
        out = relu(out)
        # last hidden layer 
        out = self.linear4(out)        
        
        #sigmoid function
        out = torch.sigmoid(out)
        return out
    

def main():
    beta = 9
    load = True
    method = "SortedDistSquared" # or SortCNum or mu2mu3
    if method == "SortedDistSquared":
        directory = "CV_SortedDistSquared"
    elif method == "SortCNum":
        directory = "CV_OrderedCoordNum"
    elif method == "mu2mu3":
        directory = "mu2mu3"
    fname = os.path.join(directory, f"Data/Committor_LJ7_{method}_BETA{beta}.npz")
    save_folder = os.path.join(directory, f"Data/Committor_LJ7_{method}_BETA{beta}")
    
    inData = np.load(fname)
    pts = inData["points"]
    Q = inData["committor"]
    tri = inData['tri']
    CVlist = inData['CVlist']
    print(f"Shape of trajectory data:{pts.shape}")
    train_data = torch.tensor(pts,dtype=torch.float32)
    Q = torch.tensor(Q,dtype=torch.float32)

    dirname = "FEMdataBETA"+str(beta)+"/"
    if method == "mu2mu3":
        ptsA = torch.tensor(np.loadtxt(dirname + "LJ7_ptsA.csv", delimiter=',', dtype=float), dtype=torch.float32)
        ptsB = torch.tensor(np.loadtxt(dirname + "LJ7_ptsB.csv", delimiter=',', dtype=float), dtype=torch.float32)
    else:
        ptsA = torch.tensor(np.loadtxt(dirname + "ptsA.csv", delimiter=',', dtype=float), dtype=torch.float32)
        ptsB = torch.tensor(np.loadtxt(dirname + "ptsB.csv", delimiter=',', dtype=float), dtype=torch.float32)

    train_data = torch.cat((train_data, ptsA, ptsB), 0)
    Q = torch.cat((Q, torch.zeros(len(ptsA)), torch.ones(len(ptsB))), 0)
    train_data.requires_grad_(True)

    print(train_data.shape, Q.shape)

    # initialization
    input_size = 2
    output_size = 1
    model = LJ7_2(input_size,40,40, output_size)

    train_ds = TensorDataset(train_data,Q)
    batch_size = 256
    print(f"batch size:{batch_size}")
    train_dl = DataLoader(train_ds, batch_size, shuffle=True)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50,100,150,200,300,400], gamma=0.5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.7)
    
    
    if load:
        for name, param in model.named_parameters():
            shape_str = ",".join(map(str, param.shape))
            filename = os.path.join(save_folder, f"{name.replace('.', '_')}_[{shape_str}].csv")

            if os.path.exists(filename):
                param_data = np.loadtxt(filename, delimiter=",")  # Load CSV data
                param_data = torch.tensor(param_data, dtype=param.dtype)  # Convert to tensor

                # **Fix shape mismatches**
                if param.shape != param_data.shape:
                    if len(param.shape) == 2 and len(param_data.shape) == 1:
                        param_data = param_data.view(param.shape)  # Reshape 1D -> 2D
                    elif len(param.shape) == 1 and param_data.numel() == 1:
                        param_data = param_data.view(1)  # Reshape scalar -> (1,)
                    else:
                        print(f"Skipping {name} due to incompatible shape {param_data.shape}")

                # Load into model if shape matches
                if param.shape == param_data.shape:
                    param.data.copy_(param_data)
                    print(f"Loaded: {filename}")
                else:
                    print(f"Shape mismatch for {name}: expected {param.shape}, got {param_data.shape}")
            else:
                print(f"File not found: {filename}")

        # Set model to evaluation mode
        model.eval()
    else:
        for epoch in range(1000):
            for X,y in train_dl:
                optimizer.zero_grad()

                Q_pred = model(X)

                loss = loss_fn(Q_pred.squeeze(), y)
                loss.backward()
                optimizer.step()
            scheduler.step()

            if epoch%25 == 0:
                print('Epoch: {}, Loss : {:.4f}'.format(epoch, loss))
        
        os.makedirs(save_folder, exist_ok=True)  # Create folder if not exists

        # Loop through each named parameter in the model
        for name, param in model.named_parameters():
            shape_str = ",".join(map(str, param.shape))  # Convert shape tuple to string (e.g., "10x5")
            filename = os.path.join(save_folder, f"{name.replace('.', '_')}_[{shape_str}].csv")  # Include shape in filename
            np.savetxt(filename, param.detach().cpu().numpy(), delimiter=',')  # Save as .npy
            print(f"Saved: {filename}")
    
    
    if method == "SortedDistSquared":
        title = r'$\beta = $' + str(beta) + r'sort[$d^2$]' +', NN'
        xlabel = r"CV1, sort[$d^2$]"
        ylabel = r"CV2, sort[$d^2$]"
    elif method == "SortCNum":
        title = r'$\beta = $' + str(beta) + 'sort[c]' +', NN'
        xlabel = r"CV1, sort[c]"
        ylabel = r"CV2, sort[c]"
    elif method == "mu2mu3":
        title = r'$\beta = $' + str(beta) + r'$(\mu_2, \mu_3)$' +', NN'
        xlabel = r"$\mu_2$"
        ylabel = r"$\mu_3$"
        
    q = model(torch.tensor(pts, dtype = torch.float32))
    plt.figure(figsize=(8,7)) 
    plt.rcParams.update({'font.size': 20})
    plt.tricontourf(pts[:,0], pts[:,1],tri,q[:,0].detach(),np.array([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]))

    plt.colorbar(label="Committor", orientation="vertical")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    figure_name = os.path.join(directory, f"Figures/NN_LJ72D_committor_{method}_BETA{beta}.pdf")
    plt.savefig(fname)
    