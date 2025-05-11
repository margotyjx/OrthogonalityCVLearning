import torch
import torch.utils.data
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.nn import functional as F
import copy
import sys

import numpy as np
import matplotlib.pyplot as plt
#import src.diffusion_map as dmap
#import src.helpers as helpers
plt.rcParams['figure.dpi'] = 100 # default is 75
plt.style.use('default')

import math

sys.path.insert(0,"/export/jyuan98/CV_learning/")
sys.path.insert(0,"/export/jyuan98/CV_learning/dMap_based/")


# import L0L1.utils_LJ7, L0L1.l0l1_models
import dMap_based.models, dMap_based.utils
import utils_LJ8, utils


model_output = '/export/jyuan98/CV_learning/LJ8/model_output/'
data_folder = '/export/jyuan98/CV_learning/LJ8/data/'
fig_folder = '/export/jyuan98/CV_learning/LJ8/figures/'
# adj_file = '/export/jyuan98/CV_learning/LJ8/data/BinConfs.csv'
# adj_file = "/export/jyuan98/CV_learning/LJ8/data/LJ8bins_confs_list.csv"
# adj_file = '/export/jyuan98/CV_learning/LJ8/data/LJ8combined_all.csv'
# adj_file = '/export/jyuan98/CV_learning/LJ8/data/LJ8combined_half.csv'
adj_file = '/export/jyuan98/CV_learning/LJ8/data/WTMetad_data_20000.csv'

graph = True

Adj_data = np.loadtxt(adj_file,delimiter=",")
    
    
if graph:
    N = len(Adj_data)
    Mu2n3 = np.zeros((N, 2))
    for i in range(N):
        mu2, mu3 = utils_LJ8.mu2n3_np_LJ8(Adj_data[i,:])
        Mu2n3[i,0] = mu2
        Mu2n3[i,1] = mu3
    
    
    plt.scatter(Mu2n3[:,0],Mu2n3[:,1],s = 1)
    plt.colorbar()

data_file = 'WT'
# data_file = 'binconf'
# data_file = 'combine'
data_file = 'wt_CrdNum'

if data_file == 'WT':
    AE_model = torch.load(model_output + 'LJ8_CVlearner_2D_wt20000_noMEP.pt',map_location=torch.device('cpu'))
    DDnet = torch.load(model_output + 'LJ8_DDnet_knn1_eps4_WT20000.pt',map_location=torch.device('cpu'))
    manifold = torch.load(model_output + 'LJ8_manifold_learner_WT_20000.pt',map_location=torch.device('cpu'))
    ptsClound = np.loadtxt(data_folder + 'ptsCloud_WT_20000.csv',delimiter=",").astype(np.float32)
    target = np.loadtxt(data_folder + 'LJ8_eigens_knn1_eps4_wt20000.csv', delimiter=',').astype(np.float32)

elif data_file == 'binconf':
    AE_model = torch.load(model_output + 'LJ8_CVlearner_2D_largeData.pt',map_location=torch.device('cpu'))
    # AE_model = torch.load(model_output + 'LJ8_CVlearner_2D.pt',map_location=torch.device('cpu'))
    DDnet = torch.load(model_output + 'LJ8_DDnet.pt', map_location=torch.device('cpu'))
    manifold = torch.load(model_output + 'LJ8_manifold_learner.pt',map_location=torch.device('cpu'))
    ptsClound = np.loadtxt(data_folder + 'ptsCloud.csv',delimiter=",").astype(np.float32)
    target= np.loadtxt(data_folder + 'LJ8_eigens_knn1_eps3.csv', delimiter=',').astype(np.float32)

elif data_file == 'binconflist':
    AE_model = torch.load(model_output + 'LJ8_CVlearner_2D_largeData.pt',map_location=torch.device('cpu'))
    DDnet = torch.load(model_output + 'LJ8_DDnet_binconflist.pt',map_location=torch.device('cpu'))
    manifold = torch.load(model_output + 'LJ8_manifold_learner_binconflist.pt',map_location=torch.device('cpu'))
    ptsClound = np.loadtxt(data_folder + 'ptsCloud_binconflist.csv',delimiter=",").astype(np.float32)
    target= np.loadtxt(data_folder + 'LJ8_eigens_knn1_eps4_binconflist.csv', delimiter=',').astype(np.float32)

elif data_file == 'combine':
    # AE_model = torch.load(model_output + 'LJ8_CVlearner_2D_combine_test.pt',map_location=torch.device('cpu'))
    AE_model = torch.load(model_output + 'LJ8_CVlearner_2D_combine_21_1.pt',map_location=torch.device('cpu'))
    DDnet = torch.load(model_output + 'LJ8_DDnet_combined_1.pt',map_location=torch.device('cpu'))
    manifold = torch.load(model_output + 'LJ8_manifold_learner_combine.pt',map_location=torch.device('cpu'))
    ptsClound = np.loadtxt(data_folder + 'ptsCloud_combine.csv',delimiter=",").astype(np.float32)
    target= np.loadtxt(data_folder + 'LJ8_eigens_knn1_eps4_combined.csv', delimiter=',').astype(np.float32)
    
elif data_file == 'wt_CrdNum':
    # AE_model = torch.load(model_output + 'LJ8_CVlearner_2D_knn5_eps4_wt20000_CrdNum.pt',map_location=torch.device('cpu'))
    # AE_model = torch.load(model_output + 'LJ8_CV_knn5_eps05_wt20000_CrdNum_normalized.pt',map_location=torch.device('cpu'))
    AE_model = torch.load(model_output + 'LJ8_CV_knn5_eps05_wt20000_CrdNum.pt',map_location=torch.device('cpu'))
    DDnet = torch.load(model_output + 'LJ8_CV_knn5_eps05_wt20000_CrdNum_normalized.pt',map_location=torch.device('cpu'))
    manifold = torch.load(model_output + 'LJ8_manifold_learner_knn5_eps1_wt20000_CrdNum.pt',map_location=torch.device('cpu'))
    # ptsClound = np.loadtxt(data_folder + 'ptsCloud_WT20000_CrdNum.csv',delimiter=",").astype(np.float32)
    ptsClound = np.loadtxt(data_folder + 'ptsCloud_WT20000_knn5_eps1_CrdNum.csv',delimiter=",").astype(np.float32)
    target= np.loadtxt(data_folder + 'LJ8_eigens_knn5_eps1_wt20000_CrdNum.csv', delimiter=',').astype(np.float32)


pts_off = torch.from_numpy(ptsClound)



store_data = '/export/jyuan98/CV_learning/LJ8/data/' + data_file + '/'

# ========== CV ============

# CV space
# r2_data = (torch.from_numpy(utils.sort_r2_LJ8(Adj_data))).to(torch.float32)
r2_data = torch.from_numpy(utils_LJ8.C_np_batch_LJ8(Adj_data)).to(torch.float32)
# CV_dmap,recon = AE_model(r2_data)
CV_dmap = AE_model(r2_data)

evecs = DDnet(r2_data)

MEP= np.loadtxt(data_folder + 'StringLJ8_Min2-Min1.csv', delimiter=',').astype(np.float32)

MEP = MEP.reshape(-1, 8, 3).transpose(0, 2, 1).reshape(-1, 24)

n = len(MEP)
Mu2n3_mep = np.zeros((n, 2))
for i in range(n):
    mu2, mu3 = utils_LJ8.mu2n3_np_LJ8(MEP[i,:])
    Mu2n3_mep[i,0] = mu2
    Mu2n3_mep[i,1] = mu3

# r2_mep = (torch.from_numpy(utils.sort_r2_LJ8(MEP))).to(torch.float32)
r2_mep = (torch.from_numpy(utils_LJ8.C_np_batch_LJ8(MEP))).to(torch.float32)
# CV_mep,recon = AE_model(r2_mep)
CV_mep = AE_model(r2_mep)

localmin = (np.loadtxt(data_folder + 'LJ8min_xyz.txt').astype(np.float32)).reshape(-1,24)

# r2_min = (torch.from_numpy(utils.sort_r2_LJ8(localmin))).to(torch.float32)
r2_min = (torch.from_numpy(utils_LJ8.C_np_batch_LJ8(localmin))).to(torch.float32)
# cv_min, recon_min = AE_model(r2_min)
cv_min = AE_model(r2_min)

print('CV dmap shape: ', CV_dmap.shape, evecs.shape)

np.savetxt(store_data + 'Evec_3D.csv', evecs.cpu().detach().numpy(), delimiter=',')
np.savetxt(store_data + 'CV_in_2D.csv', CV_dmap.cpu().detach().numpy(), delimiter=',')


plt.figure()
plt.scatter(CV_dmap[:,0].detach(),CV_dmap[:,1].detach(), s = 1, color = 'skyblue')
plt.scatter(CV_mep[:,0].detach(),CV_mep[:,1].detach(), s = 5, color = 'olivedrab')
plt.scatter(cv_min[:,0].detach(),cv_min[:,1].detach(), s = 10, color = 'red')

for i in range(cv_min.shape[0]):
    plt.text(cv_min[i, 0].detach().item(), 
             cv_min[i, 1].detach().item(), 
             f"min {i + 1}", 
             fontsize=12, 
             color="black", 
             ha='right')  # Adjust 'ha' (horizontal alignment) as needed
plt.savefig(fig_folder + "CV2D.jpeg")
plt.close()

plt.figure()
plt.scatter(Mu2n3[:,0],Mu2n3[:,1],s = 1,c = CV_dmap[:,0].detach())
plt.scatter(Mu2n3_mep[:,0],Mu2n3_mep[:,1], s = 1, color = 'red')
plt.colorbar()
plt.savefig(fig_folder + "CVinMu23_0.jpeg")

plt.figure()
plt.scatter(Mu2n3[:,0],Mu2n3[:,1],s = 1,c = CV_dmap[:,1].detach())
plt.colorbar()
plt.savefig(fig_folder + "CVinMu23_1.jpeg")

# # ========== DDnet ============

# evecs_mep = DDnet(r2_mep)
# evecs_min = DDnet(r2_min)

# np.savetxt(store_data + 'mep_in_3D.csv', evecs_mep.cpu().detach().numpy(), delimiter=',')
# np.savetxt(store_data + 'localmin_in_3D.csv', evecs_min.cpu().detach().numpy(), delimiter=',')

# fig = plt.figure(figsize=(12, 10), dpi=80)
# ax = fig.add_subplot(projection='3d')
# ax.scatter(evecs[:,0].detach(), evecs[:,1].detach(), evecs[:,2].detach(), s = 1,c = CV_dmap[:,0].detach(), alpha = 0.5)
# ax.scatter(evecs_min[:,0].detach(), evecs_min[:,1].detach(), evecs_min[:,2].detach(), s = 100,c = 'red', alpha = 1.)
# # ax.scatter(evecs_mep[:,0].detach(), evecs_mep[:,1].detach(), evecs_mep[:,2].detach(), s = 100,c = 'red')
# for i in range(evecs_min.shape[0]):
#     ax.text(evecs_min[i, 0].detach().item(), 
#              evecs_min[i, 1].detach().item(), 
#              evecs_min[i, 2].detach().item(), 
#              f"{i + 1}", 
#              fontsize=25, 
#              color="orange", 
#              ha='right')
# plt.title('values of 1D cv on eigenfunction space')
# plt.savefig(fig_folder + "CV1inEigenfunction.jpeg")

# fig = plt.figure(figsize=(12, 10), dpi=80)
# ax = fig.add_subplot(projection='3d')
# ax.scatter(evecs[:,0].detach(), evecs[:,1].detach(), evecs[:,2].detach(), s = 0.5,c = CV_dmap[:,1].detach(), alpha = 0.1)
# # ax.scatter(evecs_min[:,0].detach(), evecs_min[:,1].detach(), evecs_min[:,2].detach(), s = 100,c = 'red', alpha = 1.)
# ax.scatter(evecs_min[0,0].detach(), evecs_min[0,1].detach(), evecs_min[0,2].detach(), s = 100,c = 'red', alpha = 1.)
# # ax.scatter(evecs_mep[:,0].detach(), evecs_mep[:,1].detach(), evecs_mep[:,2].detach(), s = 100,c = 'red')
# for i in range(evecs_min.shape[0]):
#     ax.text(evecs_min[i, 0].detach().item(), 
#              evecs_min[i, 1].detach().item(), 
#              evecs_min[i, 2].detach().item(), 
#              f"{i + 1}", 
#              fontsize=25, 
#              color="orange", 
#              ha='right')
# plt.title('values of 1D cv on eigenfunction space')
# plt.savefig(fig_folder + "CV2inEigenfunction.jpeg")



# fig = plt.figure(figsize=(10, 8), dpi=80)
# ax = fig.add_subplot(projection='3d')
# ax.scatter(target[:,0], target[:,1], target[:,2],s = 1)
# ax.set_xlabel('$\Psi_1$', fontsize = 15)
# ax.set_ylabel('$\Psi_2$',fontsize = 15)
# ax.set_zlabel('$\Psi_3$',fontsize = 15)
# ax.tick_params(axis='both', which='major', labelsize=8)
# ax.tick_params(axis='both', which='minor', labelsize=25)
# plt.savefig(fig_folder + 'LJ8_dmap_test.jpeg')
# plt.show()

# # Diffusion net result on testing data

# evecs = DDnet(r2_data)

# fig = plt.figure(figsize=(10, 8), dpi=80)
# ax = fig.add_subplot(projection='3d')
# ax.scatter(evecs[:,0].detach(), evecs[:,1].detach(), evecs[:,2].detach(),s = 1)
# ax.set_xlabel('$\Psi_1$', fontsize = 15)
# ax.set_ylabel('$\Psi_2$',fontsize = 15)
# ax.set_zlabel('$\Psi_3$',fontsize = 15)
# ax.tick_params(axis='both', which='major', labelsize=8)
# ax.tick_params(axis='both', which='minor', labelsize=25)
# plt.savefig(fig_folder + 'LJ8_dnet.jpeg')
# plt.show()

# Manifold_on = manifold(torch.from_numpy(target))
# Manifold_off = manifold(pts_off)

# print(Manifold_on.mean())
# print(Manifold_off.mean())

# All_pts = np.append(target, ptsClound, axis = 0)
# manifold_value = np.append(Manifold_on.detach().numpy(),Manifold_off.detach().numpy(), axis = 0)

# fig = plt.figure(figsize=(12, 10), dpi=80)
# ax = fig.add_subplot(projection='3d')
# # p1 = ax.scatter(target[:,0], target[:,1], target[:,2], s = 1, c = Manifold_on.detach())
# # p2 = ax.scatter(ptsClound[:,0], ptsClound[:,1], ptsClound[:,2], s = 1, c = Manifold_off.detach())
# p =  ax.scatter(All_pts[:,0], All_pts[:,1], All_pts[:,2], s = 1, c = manifold_value)
# fig.colorbar(p)
# plt.savefig(fig_folder + 'Manifold.jpeg')
# plt.show()


save = True

class LJ8_2(nn.Module):
    """Feedfoward neural network with 2 hidden layer"""
    def __init__(self, in_size, hidden_size,hidden_size2, out_size):
        super().__init__()
        # 1st hidden layer
        self.linear1 = nn.Linear(in_size, hidden_size)
        # 2nd hidden layer
        self.linear2 = nn.Linear(hidden_size,hidden_size2)
        # output layer
        self.linear3 = nn.Linear(hidden_size2, out_size)
        
        
    def forward(self, xb):
        # Get information from the data
#         xb = torch.cat((torch.sin(xb),torch.cos(xb)),dim = 1)
        # Get intermediate outputs using hidden layer
        out = self.linear1(xb)
        # Apply activation function
        # tanhf = nn.Tanh()
        tanhf = nn.ReLU()
        out = tanhf(out)
        # Get predictions using output layer
        out = self.linear2(out)
        # apply activation function again
        out = tanhf(out)
        # last hidden layer 
        out = self.linear3(out)
        #sigmoid function
        out = torch.sigmoid(out)
        return out
    
committor_file = '/export/jyuan98/CV_learning/LJ8/model_output/LJ8_2hidden_committor.pt'
committor_model = torch.load(committor_file)

if save:
    # store_data_folder = "/export/jyuan98/CV_learning/LJ8/data/CV_parameters_" + data_file + '/'
    store_data_folder = "/export/jyuan98/CV_learning/LJ8/data/CV_parameters_eigens_normalized/"

    # import os
    # os.makedirs(store_data_folder, exist_ok=True)
    # for name, param in AE_model.named_parameters():
    #     np.savetxt(store_data_folder + "parameter_{}_shape_{}.csv".format(name,param.shape), param.data, delimiter=",")
    
    # print(CV_dmap.shape)
    # store_cv_file = "/export/jyuan98/CV_learning/LJ8/data/CV_values.csv"
    # np.savetxt(store_cv_file,CV_dmap.detach(), delimiter=",")

    store_committor_folder = "/export/jyuan98/CV_learning/LJ8/data/committor_unnormalized_LJ8/"

    import os
    os.makedirs(store_committor_folder, exist_ok=True)
    for name, param in committor_model.named_parameters():
        np.savetxt(store_committor_folder + "parameter_{}_shape_{}.csv".format(name,param.shape), param.data.cpu(), delimiter=",")

