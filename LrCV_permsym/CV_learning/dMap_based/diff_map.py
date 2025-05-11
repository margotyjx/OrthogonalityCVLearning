import argparse
import numpy as np
import math
import csv
import matplotlib.pyplot as plt
import torch
import random
import sys
import os
import pathlib
import re
import plotly.graph_objects as go

import utils_LJ7, utils, utils_LJ8
import src.diffusion_map as diffusion_map
from utils_LJ7 import LJ7_2, chiAB, q_theta

def main(args):
    use_C = args.use_C
    example = args.example
    fname = args.fname
    sigma = args.sigma
    beta = args.beta
    epsilon = args.epsilon
    read_knn = args.knn

    LJtraj = [] 
    with open(fname, "r") as f:
        for line in f:
            line = line.strip().strip(" \\n")
            line = re.sub(r"(-[0-9]+\.)", r" \1", line)
            values = [float(value) for value in line.split()]
            LJtraj.append(values)
    LJtraj = np.array(LJtraj).astype(np.float32)
    print('adjusted data shape:', np.shape(LJtraj))

    if example == "LJ7":
        V = utils_LJ7.LJpot_np_batch(LJtraj)
    elif example == "LJ8":
        V = utils_LJ8.LJpot_np_batch_LJ8(LJtraj)
    
    tm = np.exp(-beta * (V - np.min(V))) * np.exp(-beta * np.min(V))
    tm = tm / np.max(tm)

    N = LJtraj.shape[0]
    if read_knn < 1.:
        knn = int(read_knn * N) 
    else:
        knn = int(read_knn)
    print(f"Using value {knn} as knn approximation")

    tmdmap = diffusion_map.TargetMeasureDiffusionMap(
        epsilon=epsilon,
        n_neigh=knn,
        target_measure=np.ones_like(tm)
    )

    os.makedirs(f'../{example}/data', exist_ok=True)
    os.makedirs(f'../{example}/figures', exist_ok=True)

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

    dim_list = [1, 2, 3] # use [0,1,2] for sortDistSquared
    evecs = evecs[dim_list]

    fig = go.Figure(data=[
        go.Scatter3d(
            x=evecs[:, 1], 
            y=evecs[:, 2], 
            z=evecs[:, 3], 
            mode='markers',
            marker=dict(size=1.5)
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
                color=np.linspace(0, 1, 4),
                colorscale='Reds',
                colorbar=dict(title='Intensity')
            )
        ))

    fig.update_layout(
        title=dict(
            text="CV1 value in manifold via diffusion net",
            font=dict(size=20),
            x=0.5, y=0.9
        ),
        scene_camera=dict(eye=dict(x=2.5, y=0.2, z=0.5)),
        scene=dict(
            xaxis_title="x",
            yaxis_title="y",
            zaxis_title="z",
            xaxis=dict(backgroundcolor='lightgrey', gridcolor='white'),
            yaxis=dict(backgroundcolor='lightgrey', gridcolor='white'),
            zaxis=dict(backgroundcolor='lightgrey', gridcolor='white')
        ),
        width=700,
        height=700,
        margin=dict(l=10, r=10, b=10, t=10),
        paper_bgcolor='white',
        plot_bgcolor='lightgrey'
    )

    fig.write_image(f"../{example}/figures/manifold.pdf")
    np.savetxt(f"../{example}/data/{example}_eigens_knn{knn}_eps{epsilon}_CrdNum.csv", evecs, delimiter=',')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate manifold with diffusion map for LJ system")
    parser.add_argument("--example", type=str, choices=["LJ7", "LJ8"], default="LJ7", help="Which LJ system to use")
    parser.add_argument("--fname", type=str, required=True, help="Path to adjusted configuration file")
    parser.add_argument("--use_C", action="store_true", help="Use coordination number (C) instead of pairwise distances")
    parser.add_argument("--beta", type=float, default=10.0, help="Inverse temperature")
    parser.add_argument("--epsilon", type=float, default=1.0, help="Epsilon value for diffusion map")
    parser.add_argument("--knn", type=float, default=0.25, 
                        help="knn value for diffusion map. If less than 1, interpreted as ratio to the total number")
    args = parser.parse_args()
    main(args)
