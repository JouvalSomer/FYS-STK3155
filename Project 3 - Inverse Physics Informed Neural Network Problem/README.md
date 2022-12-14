# Project 3 - FYS-STK3155

## Estimating the diffusion coefficient of an MRI tracer using Physics Informed Neural Networks (PINNs)
---
### Problem statment:
> In hopes of getting a better understanding of the glymphatic system and to demonstrate the
potential of PINNs we want in this study to apply a physics-informed neural network model to
investigate the diffusive properties of a tracer molecule observed via MRI imaging over a period of
48 hours. Since it has been shown that PINNs struggle with noisy data (Zapf et al. 2022), we will
for simplicity use synthetically made simulated MRI-like images alongside the diffusion equation
to try to estimate the diffusion coefficient of the tracer.
---

### In this GitHub you'll find the source files, the data as well as the results for this project.

## **Content of folders:**

* ## src
  * PINN_main.py
  
    This is the main script file that includes all the frontend. This inclues the the main hyperparameters.
  * data.py
  
    This file contains all the handling of the data. This includes import and scaling of the data, creation of the different grids and coordinats and making the input/output pairs for the PINN.
  * network.py
  
    This file houses the neural network class.
  * optim.py
  
    In this file you'll find the losses and the optimization loop.
  * plot.py
    
    And here is were all the plotting is done. This includes: 
    - A plot of the diffusion coeffition against epochs.
    - A plot of the total loss (NN loss + PDE loss) against epochs.
    - And plots of the "true" MRI consentration images and the NNs predixtion of them.

* ## results
    This folder contains all the plotts described above.

* ## data
    And this folder contains the two data-sets and their coresponding masks for region of interest selection (used for train/test splt).
---
## Requirements
This project uses Pytorch. For documentation and installation guide [click here](https://pytorch.org/get-started/locally/ "Pytorch documentation"). Additionally the package tqdm is used to get a progress-bar for the PINN traning. For documentation on tqem [click here](https://tqdm.github.io/ "tqdm documentation"). Furthermore the packages numpy and matplotlib are used.
