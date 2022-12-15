import torch
import numpy as np
import time
import datetime
from sklearn.model_selection import train_test_split

from data import *
from optim import *
from plot import *
from network import Net

""" Check if there is a GPU available """
if torch.cuda.is_available():
    device = torch.device("cuda")
    using_gpu = True
else:
    device = torch.device("cpu")
    using_gpu = False


"""     HYPERPARAMETERS     """
max_epochs = 10 # Has to be greater than or equal to 10
n_pde = int(1e5) # Number of residual points
D_init = 1.0 # Intial guess for the diff. coeff. D
pde_w = 5 # PDE-weights

sched_const = -0.5 # Schedular step-size factor. init_LR * 10^(sched_const * current_epoch/max_epochs)
sched = False # Whether to use a schedular of not

learning_rate_NN = 1e-2 # Initial learningrate for the NN (the part that finds the consentration)
learning_rate_D = 1e-2  # Initial learningrate for the PDE (the part that finds the diff. coeff.)

optim = 'L-BFGS' # Optimzer. Choose between: 'ADAM' or 'L-BFGS'

reg = None # Regularization. Choose between: None, 'L1', 'L2'
lmb = 0 # Regularization parameter. Scales the regularization term

print_out = True # Prints out the iteration, total loss and current diff. coeff. every 10% of max_epochs

# Defining the NN:
NN = Net(num_hidden_units=32, num_hidden_layers=5, inputs=3, inputnormalization=True).to(device)

loss_function=torch.nn.MSELoss(reduction="mean") # Loss function. Choose between: MSELoss and L1Loss

# Making diff. coeff. D a learnable parameter
D_param = torch.tensor(D_init, device=device, dtype=torch.float)
D_param = torch.nn.Parameter(D_param)
D_param = D_param.to(device)

params = list(NN.parameters())

# params = list(NN.parameters()) + [D_param] ?

""" Set optimizer """
# ADAM
if optim == 'ADAM':
    optimizer = torch.optim.Adam([{'params': params, "lr" : learning_rate_NN},
                                {'params': D_param, 'lr': learning_rate_D}])

    lr_lambda = lambda current_epoch: 10 ** (sched_const * current_epoch / max_epochs) # LR scheduler function
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    print('Optimization loop has started with max epochs = ', max_epochs)
    start = time.time()
    D_during_train, total_losses, dloss, pdeloss = optimization_loop(max_epochs, NN, loss_function, D_param, 
                        pde_w, optimizer, reg, lmb, scheduler, sched=sched, print_out=print_out, n_pde=n_pde)
    end = time.time()
    tot_time = end - start
    print('\nOptimization loop has ended. Total time used was:', str(datetime.timedelta(seconds=tot_time)), '\n')

# L-BFGS
if optim =='L-BFGS':
    def closure():
        pde_points = init_collocation_points(input_list[0], tmax, tmin, num_points=n_pde)
        optimizer.zero_grad()

        # Forward pass for the data:
        data_loss_value = data_loss(NN, input_list, data_list, loss_function)
        
        # Forward pass for the PDE 
        pde_loss_value = pde_loss_residual(pde_points, NN, D_param, loss_function)
        
        total_loss = data_loss_value + pde_w * pde_loss_value

        if reg == 'L1': # Adding L1 regularization
            l1_penalty = torch.nn.L1Loss(reduction='sum') # size_average=False
            l1_reg_loss = 0
            for param in NN.parameters():
                l1_reg_loss += l1_penalty(param, torch.zeros_like(param))
            total_loss += lmb * l1_reg_loss
        
        elif reg == 'L2':
            l2_reg_term = 0
            for param in NN.parameters():
                l2_reg_term += torch.sum(param ** 2)
            total_loss += lmb * l2_reg_term

        if total_loss.requires_grad:
            total_loss.backward()
            
        # Log the losses to make figures
        losses.append(total_loss.item())
        dloss.append(data_loss_value.item())
        pdeloss.append(pde_loss_value.item())

        if sched:
            scheduler.step()

        return total_loss

    optimizer = torch.optim.LBFGS(params,
                                    max_iter=max_epochs,
                                    tolerance_grad=1e-7,
                                    tolerance_change=1e-10,
                                    line_search_fn="strong_wolfe")

    lr_lambda = lambda current_epoch: 10 ** (sched_const * current_epoch / max_epochs) # LR scheduler function
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    print('Optimization loop has started with max epochs = ', max_epochs)
    start = time.time()
    total_loss = optimizer.step(closure)
    end = time.time()
    tot_time = end - start
    print('\nOptimization loop has ended. Total time used was:', str(datetime.timedelta(seconds=tot_time)), '\n')



L_squared = 5648.0133
T = 172800.0
scaling_factor = L_squared / T

D_during_train = np.array(D_during_train)*scaling_factor
D_mean = sum(D_during_train[-100:])/100

print('Started plotting and saving the figs. This may take a minute.')
plot_total_losses(total_losses, dloss, pdeloss)
D_plot(D_during_train)
plot_MRI_images(NN, train=False)
print('Finished plotting and saving the figs.\n')

print(f'D_init = {D_init:.4f}') # Initial guess for diff. coeff. (dimensionless)
print(f'D end = {D_param.item():.4f}') # Final diff. coeff. (dimensionless)
print(f'D_mean end = {D_mean* 10**4:.4f} x 10^(-4)') # Mean of the last 100 diff. coeff. (still dimensionless)

# Final and mean of the last 100 diff. coeff. (now in SI units):
print(f'D end in SI = {D_mean*scaling_factor * 10**4:.4f} x 10^(-4) [mm^2 s^(-1)]')
print(f'D_param end in SI = {D_param.item()*scaling_factor * 10**4:.4f} x 10^(-4) [mm^2 s^(-1)]')
