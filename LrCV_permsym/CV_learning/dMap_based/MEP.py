import numpy as np
import torch
import utils

output_folder = '../LJ8/model_output/'
data_folder = '../LJ8/data/'


MEP= np.loadtxt(data_folder + 'StringLJ8_Min2-Min1.csv', delimiter=',').astype(np.float32)

MEP = MEP.reshape(-1, 8, 3).transpose(0, 2, 1).reshape(-1, 24)

# r2_mep = utils.sort_r2_LJ8(MEP)

# arc length of MEP

dMEP = MEP - np.roll(MEP, 1, axis = 0)
dMEP[0,:] = np.zeros_like(dMEP[0,:])

dS = np.sqrt(np.sum(dMEP**2, axis = 1))
S = np.cumsum(dS)

np.savetxt(data_folder+'arc_len_MEP_min2-min1.txt', S, delimiter=',')