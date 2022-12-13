import torch
import numpy as np
import time
import datetime
from sklearn.model_selection import train_test_split

from func import *
from plot import *
from network import Net

""" Check if there is a GPU available """
if torch.cuda.is_available():
    device = torch.device("cuda")
    using_gpu = True
else:
    device = torch.device("cpu")
    using_gpu = False



"    HYPERPARAMETERS    "
max_epochs = 20
n_pde = int(1e5) #Number of residual points
D_init = 1 # *10**(-4) #Intial guess for D (Valnes et al. 2020)
pde_w = 5 #PDE-weights

print_out = True

sched_const = -0.5 # Step-size factor
sched = True

learning_rate_NN = 1e-2
learning_rate_D = 1e-2
reg = None # None, 'L1', 'L2'
lmb = 0

u_nn = Net(num_hidden_units=32, num_hidden_layers=5, inputs=3, inputnormalization=True).to(device)
loss_function=torch.nn.MSELoss(reduction="mean") # L1Loss

# making D a learnable parameter
D_param = torch.tensor(D_init, device=device, dtype=torch.float)
D_param = torch.nn.Parameter(D_param)
D_param = D_param.to(device)

params = list(u_nn.parameters())
if reg == 'L2':
    optimizer = torch.optim.Adam([{'params': params, "lr" : learning_rate_NN},
                                {'params': D_param, 'lr': learning_rate_D}], weight_decay=lmb)
else:
    optimizer = torch.optim.Adam([{'params': params, "lr" : learning_rate_NN},
                                {'params': D_param, 'lr': learning_rate_D}])

lr_lambda = lambda current_epoch: 10 ** (sched_const * current_epoch / max_epochs) # Learning rate scheduler 
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)



print('Optimization loop has started with max epochs = ', max_epochs)
start = time.time()
D_during_train, total_losses, dloss, pdeloss = optimization_loop(max_epochs, u_nn, loss_function, D_param, pde_w, optimizer, reg, lmb, scheduler, sched=sched, print_out=print_out, n_pde=n_pde)
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
plot_MRI_images(u_nn)
print('Finished plotting and saving the figs.\n')


print(f'D_init = {D_init:.4f}')
print(f'D end = {D_param.item():.4f}')

print(f'D_mean end = {D_mean* 10**4:.4f} x 10^(-4)')

print(f'D end in SI = {D_mean*scaling_factor * 10**4:.4f} x 10^(-4) [mm^2 s^(-1)]')
print(f'D_param end in SI = {D_param.item()*scaling_factor * 10**4:.4f} x 10^(-4) [mm^2 s^(-1)]')
