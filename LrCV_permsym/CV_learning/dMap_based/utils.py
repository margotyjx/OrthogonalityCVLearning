from itertools import combinations, permutations
import numpy as np
import torch

"""
This file contains all utils for learning CV for all examples
"""


# ======================= Target Measure Diffusion Map ===================

"""
Check this band derivative scheme in TMDM paper
"""
## choose the optimal epsilon
def band_derivative(K):
    K_sum_1 = np.sum(K*np.log(K))
    K_sum_2 = np.sum(K)
    return -K_sum_1/K_sum_2

def band_choice(lower, upper, X, step_size):
    num_length = int((upper - lower) / step_size + 1)
    epsilon_test = np.zeros(num_length)
    derivatives_array = np.zeros(num_length)
    for v in range(num_length):
        epsilon_test[v] = lower + v * step_size
    for k in range(len(epsilon_test)):
        dist_mat_test = kernal_mat(X, epsilon_test[k])
        derivatives_array[k] = band_derivative(dist_mat_test)
    l = np.argmax(derivatives_array)
    return epsilon_test[l], derivatives_array



def kernal_mat(X, epsilon_n, Sparsification = False):
    threshold = 3*np.sqrt(1/2*epsilon_n)
    dist_mat = np.zeros((len(X), len(X)))
    for i in range(len(X)):
        for j in range(i, len(X)):
            # Gaussian kernel
            cur_dist = np.exp(-(np.linalg.norm(X[i, :]-X[j, :])**2)/epsilon_n)
            # Sparsification Step
            if Sparsification:
                if np.linalg.norm(X[i, :]-X[j, :]) < threshold:
                    dist_mat[i, j] = cur_dist
                    dist_mat[j, i] = cur_dist
            else:
                dist_mat[i, j] = cur_dist
                dist_mat[j, i] = cur_dist
    return dist_mat



def normalize_K(x, K, beta, potential):
    q_e = np.sum(K,axis = 1)
    # target_dist = np.exp(-beta*potential(x))
    V = potential(x)
    target_dist = np.exp(-beta*(V - np.min(V)))*np.exp(-beta*np.min(V))
    target_dist = target_dist/np.max(target_dist)
    D = np.sqrt(target_dist)/q_e
    D_diag = np.diag(D)
    K_normed = K@D_diag
    
    return K_normed

def L_mat(X,K,beta, epsilon, potential):
    K_tilde = normalize_K(X,K,beta, potential)
    D_tilde_inv = np.diag(1/np.sum(K_tilde, axis = 1))
    L = (D_tilde_inv@K_tilde - np.eye(len(K_tilde)))/epsilon
    
    return L

def tm_dmap(X,beta, potential):
    # lower, upper,step
    # epsilon, derivatives_array = band_choice(lower, upper, X, step)
    epsilon = 4.0
    print('epsilon: ', epsilon)
    K_mat = kernal_mat(X, epsilon)
    print('computed K_mat')
    L = L_mat(X, K_mat, beta, epsilon, potential)
    
    return L

def eigens(X, L, ndim = 3):
    # column eigenvectors[:,i] is the eigenvector corresponding to the eigenvalue eigenvalues[i].
    eigenvals, eigenvecs = np.linalg.eig(L)
    idx = eigenvals.argsort()[::-1]   
    eigenvals_s = eigenvals[idx]
    eigenvecs_s = eigenvecs[:, idx]
    tmdm = eigenvecs[:,1:ndim+1]

    return tmdm
    

# ======================= Generate Point Clouds Around the Manifold ===================

# Function to generate points within a box of certain side length away (rad) from the manifold
def randompoint(ndim, rad, A):
    vec = np.random.uniform(low = -rad, high = rad, size = ndim) + A
    return vec

def keep(vec, z_data, rtol):
    include = True
    for i in range(len(z_data)):
        # If a point is too close to a point on the manifold, we discard it
        if np.linalg.norm(vec - z_data[i, :]) < rtol:
            include = False
            break
    return include

# Function to generate random points for the point cloud that are not too close to the manifold until we obtain target_num number of points
def gen_points(ndim, rad, z_data, rtol, target_num):
    points = []
    count = 0
    z = z_data.detach().numpy()
    A = np.mean(z, axis = 0)
    while count < target_num:
        vec = randompoint(ndim, rad, A)
        if keep(vec, z, rtol):
            points.append(vec)
            count += 1
    return np.array(points)

def sym_poly(data):
    total_dim = data.shape[1]
    new_data = np.zeros_like(data)
    dim = int(total_dim/2)
    atom = np.arange(dim)

    for i in range(dim):
        comb = list(combinations(atom, i+1))
        for ind in range(len(comb)):
            comb_set = np.asarray(comb[ind])
            tmp1 = np.ones(data.shape[0])
            tmp2 = np.ones(data.shape[0])
            for item in comb_set:
                tmp1 *= data[:,item]
                tmp2 *= data[:,item + dim]
            new_data[:,i] += tmp1
            new_data[:,i+dim] += tmp2
            
    return new_data

def sym_poly_pt(data):
    total_dim = data.shape[1]
    new_data = torch.zeros_like(data)
    dim = int(total_dim/2)
    atom = torch.arange(dim)

    for i in range(dim):
        comb = list(combinations(atom, i+1))
        for ind in range(len(comb)):
            comb_set = torch.tensor(comb[ind])
            tmp1 = torch.ones(data.shape[0])
            tmp2 = torch.ones(data.shape[0])
            for item in comb_set:
                tmp1 *= data[:,item]
                tmp2 *= data[:,item + dim]
            new_data[:,i] += tmp1
            new_data[:,i+dim] += tmp2
            
    return new_data





