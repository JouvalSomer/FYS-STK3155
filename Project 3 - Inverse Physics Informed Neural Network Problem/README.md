# Project 3 - FYS-STK3155

## Estimating the diffusion coefficient of an MRI tracer using Physics Informed Neural Networks (PINNs)
---
### Problem statment:
> In hopes of getting a better understanding of the glymphatic system and to demonstrate the
potential of PINNs we want to, in this study, apply a physics-informed neural network to
investigate the diffusive properties of a tracer molecule observed via MRI imaging over a period of
48 hours. Since it has been shown that PINNs struggle with noisy data (Zapf et al. 2022), we will
for simplicity reasons use synthetically made simulated MRI-like images. Alongside the images we will apply the diffusion equation as our physical insight to solve the inverse problem of discovering the diffusion coefficient for this process.
---

### In this GitHub repo you'll find two folders. One folder, src, containing the source files and the data that was used, and one, results, containing the results.

## **Content of folders:**

* ## src
  * **PINN_main.py**
  
    This is the main script file that includes all the frontend. This inclues the the main hyperparameters.
  * **data.py**
  
    This file contains all the handling of the data. This includes import and scaling of the data, creation of the different grids and coordinats and making the input/output pairs for the PINN.
  * **network.py**
  
    This file houses the neural network class.
  * **optim.py**
  
    In this file you'll find the losses and the optimization loop.
  * **plot.py**
    
    And here is were all the plotting is done. This includes: 
    - A plot of the diffusion coeffition against epochs.
    - A plot of the total loss (NN loss + PDE loss) against epochs.
    - And plots of the "true" MRI consentration images and the NNs predixtion of them.
    * **gravity_demo.py**
    
    This is a test demo that verifies our PINN on known data so we can ensure that it works. Here we find the gravitational acceleration g from noisy mesurements of a throw without air resistance.
    
  * ## data
       This folder contains the two data-sets used and masks for region of interest selection (used for train/test splt).

* ## results
    This folder contains all the plotts described above.

---
## Requirements
This project uses Pytorch. For documentation and installation guide [click here](https://pytorch.org/get-started/locally/ "Pytorch documentation"). Additionally, the package tqdm is used to get a progress-bar for the PINN traning. For documentation on tqem [click here](https://tqdm.github.io/ "tqdm documentation"). Furthermore, the packages numpy and matplotlib are used.
