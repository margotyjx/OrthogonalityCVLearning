# Learning collective variables for particle systems with permutational symmetry

This repository contains the implementation of paper Learning collective variables for particle systems with permutational symmetry.

## Requirements:
- python 3.7

## Usage

### Learning collective variables: CV_learning
- **Step 1**. Compute manifold with diffusion maps
  
  ```python diff_map.py --example LJ7 --fname ../LJ7/data/LJ7bins_confs.txt --use_C```
- **Step 2**. Approximate the manifold with neural network

  ```python main_DDnet.py --example LJ7 --read_knn 0.25 --epsilon 1.0 --hidden 45,30,25 --epochs 4000```

  and learn the zero-level set of the manifold

  ```python Manifold.py --example LJ7 --read_knn 0.25 --epsilon 1.0 --generate```
- **Step 3**. Learn collective variables using pre-computed components.

  ```python main_CV.py --example LJ7 --knn 125 --use_C --theta1 1.0 --theta2 0.2 --mep_loss```

### Computation of free energy and diffusion tensor and committor function via FEM

See instructions in xxx

### Visualization and approximation of committor function for forward flux sampling: CV_evaluation

Save the FEM committor information to ```Data/*.npz``` file and train neural network to approximate FEM committor function for forward flux sampling e.g.,

```
# Train from scratch
cd LJ7in2D
python NN_committor_approx.py --beta 9 --method SortCNum

# Load from saved weights
cd LJ7in2D
python NN_committor_approx.py --beta 9 --method SortCNum --load
```

### Forward flux sampling

See instructions in xxx

To visualize the free energy, diffusion tensor, committor function and probabiliy density after forward flux sampling, run the jupyter nobebook in each subfolder, i.e., 
```CV_evaluation/LJ7in2D/CV_OrderedCoordNum/LJ7_CV_SortCNum.ipynb, CV_evaluation/LJ7in2D/CV_SortedDistSquared/MLCV_SortDistSquared.ipynb``` etc.

### Rate estimation

```compute_rates.ipynb``` provides an example code to approximate escape rates and transition rates, together with the error estimation. 

#### Folder structure

```
LrCV_permsym
├── CV_evaluation
│   ├──LJ7in2D
│   │   ├── CV_OrderedCoordNum
│   │   │  ├── Data
│   │   │  ├── FEMdataBETA5
│   │   │  ├── FEMdataBETA7
│   │   │  ├── FEMdataBETA9
│   │   │  ├── Figures
│   │   │  ├── LJ7_CV_data
│   │   │  ├── FEM_TPT.py
│   │   │  └── LJ7_CV_SortCNum.ipynb
│   │   ├── CV_SortedDistSquared
│   │   │  ├── Data
│   │   │  ├── FEMdataBETA5
│   │   │  ├── FEMdataBETA7
│   │   │  ├── FEMdataBETA9
│   │   │  ├── Figures
│   │   │  ├── LJ7in2Dmin
│   │   │  ├── MLCV_SortD2_data
│   │   │  ├── FEM_TPT.py
│   │   │  └── MLCV_SortDistSquared.ipynb
│   │   ├── mu2mu3
│   │   │  ├── Data
│   │   │  ├── FEMdataBETA5
│   │   │  ├── FEMdataBETA7
│   │   │  ├── FEMdataBETA9
│   │   │  ├── Figures
│   │   │  ├── LJ7in2Dmin
│   │   │  ├── FEM_TPT.py
│   │   │  └── LJ7mu2mu3.ipynb
│   │   └──  NN_committor_approx.py
│   ├── LJ8in3D
│   │   ├── CV_LDA12_SortCNum
│   │   ├── CV_LDA23_SortCNum
│   │   ├── CV_OrderedCoordNum
│   │   ├── mu2mu3
│   │   └──  NN_committor_approx.py
│   └── compute_rates.ipynb
├── CV_learning
│   ├── dMap_based
│   │   ├── src
│   │   ├── diff_map.py
│   │   ├── main_CV.py
│   │   ├── main_DDnet.py
│   │   ├── Manifold.py
│   │   ├── MEP.py
│   │   ├── models.py
│   │   ├── utils_LJ7.py
│   │   ├── utils_LJ8.py
│   │   └── utils.py
│   ├── LJ7
│   └── LJ8
```



