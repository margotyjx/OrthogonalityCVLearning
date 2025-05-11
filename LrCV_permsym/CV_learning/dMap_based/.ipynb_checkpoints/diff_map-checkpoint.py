import numpy as np
import math
import csv
import matplotlib.pyplot as plt
import torch
import random
import sys
import os
import pathlib
import utils_LJ7, utils

from utils_LJ7 import LJ7_2,chiAB,q_theta
import utils_LJ8

import src.diffusion_map as diffusion_map

# transform to coordination number or pairwise distance
use_C = True # when this is False, use pairwise distance
example = "LJ7"

directory_path =  os.path.abspath(os.path.join("__file__" ,"../.."))

file_name = '../LJ7/data/LJ7bins_confs.txt'

import re
sigma = 0.02
fname = adj_file

LJtraj = [] 
with open(fname, "r") as f:
    for line in f:
        # cleaning the bad chars in line
        line = line.strip()
        line = line.strip(" \\n")
        line = re.sub(r"(-[0-9]+\.)", r" \1", line)
        values = [float(value) for value in line.split()]
        LJtraj.append(values)
LJtraj = np.array(LJtraj).astype(np.float32)
print('adjusted data shape: ',np.shape(LJtraj))


if example == "LJ7":
    V = utils_LJ7.LJpot_np_batch(LJtraj14D)
elif example == "LJ8":
    V = utils_LJ8.LJpot_np_batch_LJ8(Adj_data)
    
beta = 10.
tm = np.exp(-beta*(V - np.min(V)))*np.exp(-beta*np.min(V))
tm = tm/np.max(tm)

knn = int(0.25*N)
print("using value {} as knn approximation".format(knn))
epsilon = 1.0

tmdmap = diffusion_map.TargetMeasureDiffusionMap(epsilon=epsilon, n_neigh=knn, \
                                                          target_measure=np.ones_like(tm))

os.makedirs(f'../{example}/data', exist_ok=True)

if example == "LJ7":
    minima_list = utils_LJ7.four_wells_conf()
    LJtraj = np.concatenate((LJtraj, minima_list), axis=0)
    
    if use_C:
        C = utils_LJ7.C_np_batch_LJ7(LJtraj)
        np.savetxt(f'../{example}/data/ConfBins_data_CrdNum.csv', C, delimiter=',')
        tmdmap.construct_generator(C.T)
    else:
        r2 = utils_LJ7.sort_r2(LJtraj)
        np.savetxt(f'../{example}/data/ConfBins_data_dist.csv', r2, delimiter=',')
        tmdmap.construct_generator(r2.T)
elif example == "LJ8":
    C = utils_LJ8.C_np_batch_LJ8(LJtraj)
    np.savetxt(f'../{example}/data/ConfBins_data_CrdNum.csv', C, delimiter=',')
    tmdmap.construct_generator(C.T)

    
L = tmdmap.get_generator()

dmap, evecs, evals = tmdmap._construct_diffusion_coords(L)

dim_list = [1,2,3]

evecs = evecs[dim_list]

# =========== visualization =============

import plotly.graph_objects as go

fig = go.Figure(data=[
    go.Scatter3d(
        x=evecs[:, 1], 
        y=evecs[:, 2], 
        z=evecs[:, 3], 
        mode='markers',
        marker=dict(
            size=1.5,
            # color=,  # Set color for points
            # colorscale='Jet',  # Choose a colorscale (e.g., Viridis, Plasma, Jet, etc.)
            # colorbar=dict(title='CV1 value'),  # Add a colorbar
            # opacity=0.8
        )
    )
])

if example == "LJ7":
    fig.add_trace(go.Scatter3d(
        x=evecs[-4:, 1], 
        y=evecs[-4:, 2], 
        z=evecs[-4:, 3], 
        mode='markers',
        marker=dict(
            size=10,
            color=np.linspace(0, 1, 4),  # Varying color values
            colorscale='Reds',  # Shades of red
            colorbar=dict(title='Intensity'),  # Optional colorbar
        )
    ))

# fig.update_layout(title="Interactive Rotational 3D Plot")
camera = dict(
    eye=dict(x=2.5, y=0.2, z=0.5)  # Adjust x, y, z for the desired angle
)

# Update Layout
fig.update_layout(
    title=dict(
        text="CV1 value in manifold via diffusion net",  # Title text
        font=dict(size=20),  # Title font size
        x=0.5,  # Center align the title
        y=0.9
    ),
    scene_camera=camera,
    scene=dict(
        xaxis_title=r"x",  # Custom X-axis label
        yaxis_title=r"y",  # Custom Y-axis label
        zaxis_title=r"z",  # Custom Z-axis label
        xaxis=dict(backgroundcolor='lightgrey', gridcolor='white'),  # Custom X-axis background
        yaxis=dict(backgroundcolor='lightgrey', gridcolor='white'),  # Custom Y-axis background
        zaxis=dict(backgroundcolor='lightgrey', gridcolor='white')   # Custom Z-axis background
    ),
    width=700,  # Set figure width
    height=700,  # Set figure height
    margin=dict(l=10, r=10, b=10, t=10),  # Set margins
    paper_bgcolor='white',  # Background color outside the plot
    plot_bgcolor='lightgrey'  # Background color inside the plot
)

fig.write_image(f"../{example}/figures/manifold.pdf")
# fig.show()

np.savetxt(f"../{example}/data/{example}_eigens_knn{knn}_eps{epsilon}_CrdNum.csv", evecs, delimiter=',') 