import torch
import numpy as np
import time
import datetime


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
max_epochs = 2000 # Has to be greater than or equal to 10
n_pde = int(1e3) # Number of residual points
D_init = 1.0 # Intial guess for the diff. coeff. D
pde_w = 1.0 # PDE-weights

print(f'Current pde_w = {pde_w}')

sched_const = -1.0 # Schedular step-size factor. init_LR * 10^(sched_const * current_epoch/max_epochs)
sched = True # Whether to use a schedular of not

learning_rate_NN = 1e-3 # Initial learningrate for the NN (the part that finds the consentration)
learning_rate_D = 1e-3  # Initial learningrate for the PDE (the part that finds the diff. coeff.)

optim = 'L-BFGS' # Optimzer. Choose between: 'ADAM' or 'L-BFGS'
print(f'Using {optim}\n')

ANN = False # ANN without any physics 

print_out = True # Prints out the iteration, total loss and current diff. coeff. every 10% of max_epochs

# Defining the NN:
NN_PINN = Net(num_hidden_units=32, num_hidden_layers=5, inputs=3, inputnormalization=True).to(device)


loss_function_NN =torch.nn.MSELoss(reduction="mean") # Loss function. Choose between: MSELoss and L1Loss
loss_function_PDE =torch.nn.L1Loss(reduction="mean") # Loss function. Choose between: MSELoss and L1Loss


# Making diff. coeff. D a learnable parameter
D_param = torch.tensor(D_init, device=device, dtype=torch.float)
D_param = torch.nn.Parameter(D_param, requires_grad=True)
D_param = D_param.to(device)

make_results_folder(optim)

""" Set optimizer """
# ADAM
if optim == 'ADAM':
    params_PINN = list(NN_PINN.parameters())
    optimizerPINN = torch.optim.Adam([{'params': params_PINN, "lr" : learning_rate_NN},
                                {'params': D_param, 'lr': learning_rate_D}])


    lr_lambda = lambda current_epoch: 10 ** (sched_const * current_epoch / max_epochs) # LR scheduler function
    schedulerPINN = torch.optim.lr_scheduler.LambdaLR(optimizerPINN, lr_lambda=lr_lambda)


    print('Optimization loop for the PINN has started with max epochs = ', max_epochs)
    start = time.time()
    D_train_during_training, losses_train_PINN, dloss_train, pdeloss_train, losses_test_PINN = optimization_loop(max_epochs, \
                                                                                NN_PINN, loss_function_NN, loss_function_PDE, \
                                                                                D_param, pde_w, optimizerPINN, \
                                                                                schedulerPINN, sched=sched, print_out=print_out, \
                                                                                n_pde=n_pde)
    end = time.time()
    tot_time = end - start
    print('\nOptimization loop for the PINN has ended. Total time used was:', str(datetime.timedelta(seconds=tot_time)), '\n')
    
    if ANN: # Optional for running an ANN without any physics
        NN_ANN = Net(num_hidden_units=32, num_hidden_layers=5, inputs=3, inputnormalization=True).to(device)
        params_ANN = list(NN_ANN.parameters())
        optimizerANN = torch.optim.Adam([{'params': params_ANN, "lr" : learning_rate_NN},
                                    {'params': D_param, 'lr': learning_rate_D}])
        lr_lambda = lambda current_epoch: 10 ** (sched_const * current_epoch / max_epochs) # LR scheduler function
        schedulerANN = torch.optim.lr_scheduler.LambdaLR(optimizerANN, lr_lambda=lr_lambda)
        print('Optimization loop for the ANN has started with max epochs = ', max_epochs)
        start = time.time()
        losses_train_ANN, losses_test_ANN = optimization_loop_NN_reg(max_epochs, NN_ANN, loss_function_NN, optimizerANN, schedulerANN, sched=False)
        end = time.time()
        tot_time = end - start
        print('\nOptimization loop for the ANN has ended. Total time used was:', str(datetime.timedelta(seconds=tot_time)), '\n')
        bvto(losses_train_ANN, losses_test_ANN, losses_train_PINN, losses_test_PINN, pde_w)


# L-BFGS
if optim =='L-BFGS':
    params = list(NN_PINN.parameters()) + [D_param]


    D_train_during_training =[]
    dloss_train = []
    pdeloss_train = []
    losses_train_PINN = []

    losses_test_PINN = []

    test_data_list, test_input_list = get_test_data()
    train_data_list, train_input_list = get_train_data()

    train_pde_points = init_collocation_points(train_input_list[0], tmax, tmin, num_points=n_pde)

    def closure():
        # Free all intermediate values:
        optimizer.zero_grad() # Resetting the gradients to zeros
        
        """ Train """
        # Forward pass for the data:
        train_data_loss_value = data_loss(NN_PINN, train_input_list, train_data_list, loss_function_NN)
        # Forward pass for the PDE 
        train_pde_loss_value = pde_loss_residual(train_pde_points, NN_PINN, D_param, loss_function_PDE)
        train_total_loss = train_data_loss_value  + pde_w * train_pde_loss_value

        """ Test """
        with torch.no_grad():
            # Forward pass for the data:
            test_total_loss = data_loss(NN_PINN, test_input_list, test_data_list, loss_function_NN)


        # Backward pass, compute gradient w.r.t. weights and biases
        train_total_loss.backward()
        

        """ Train Log """
        # Log the train diffusion coeff. to make a figure
        D_train_during_training.append(D_param.item())
        # Log the train losses to make figures
        losses_train_PINN.append(train_total_loss.item())
        dloss_train.append(train_data_loss_value.item())
        if pde_w > 0.0:
            pdeloss_train.append(train_pde_loss_value.item()) 
        
        """ Test Log """
        # Log the test losses to make figures
        losses_test_PINN.append(test_total_loss.item())

        if sched:
            scheduler.step()

        return train_total_loss


    optimizer = torch.optim.LBFGS(params,
                                    max_iter=max_epochs,
                                    tolerance_grad=1e-8,
                                    tolerance_change=1e-12,
                                    line_search_fn="strong_wolfe")

    lr_lambda = lambda current_epoch: 10 ** (sched_const * current_epoch / max_epochs) # LR scheduler function
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    print('Optimization loop has started with max epochs = ', max_epochs)
    start = time.time()
    optimizer.step(closure)
    end = time.time()
    tot_time = end - start
    print('\nOptimization loop has ended. Total time used was:', str(datetime.timedelta(seconds=tot_time)), '\n')


L_squared =  5216.875
T = 172800.0
scaling_factor = L_squared / T # [mm^2 * s^(-1)]

D_train_mean = sum(D_train_during_training[-50:])/50
D_train_during_training = np.array(D_train_during_training)

print(f'\nMean of last 50 D_train= {D_train_mean:.8f}') # * scaling_factor * 10**4:.4f x 10^(-4) [mm^2 s^(-1)]
print(f'The last D_train= {D_train_during_training[-1]:.8f}') # * scaling_factor * 10**4:.4f} x 10^(-4) [mm^2 s^(-1)]')

print('\nStarted plotting and saving the figs. This may take a minute.')

train_plot_total_losses(losses_train_PINN, dloss_train, pdeloss_train, pde_w, optim)
test_plot_total_losses(losses_test_PINN, pde_w, optim)
train_test_total_losses(losses_train_PINN, losses_test_PINN, pde_w, optim)

D_plot(D_train_during_training, pde_w, optim)

train_images()

test_data_NN_prediction(NN_PINN, pde_w, optim)

print('Finished plotting and saving the figs.\n')