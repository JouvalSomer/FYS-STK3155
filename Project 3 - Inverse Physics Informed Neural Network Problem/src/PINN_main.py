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
pde_w = 1.5 # PDE-weights

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
D_param = torch.nn.Parameter(D_param, requires_grad=True)
D_param = D_param.to(device)



""" Set optimizer """
# ADAM
if optim == 'ADAM':
    params = list(NN.parameters())
    optimizer = torch.optim.Adam([{'params': params, "lr" : learning_rate_NN},
                                {'params': D_param, 'lr': learning_rate_D}])

    lr_lambda = lambda current_epoch: 10 ** (sched_const * current_epoch / max_epochs) # LR scheduler function
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    print('Optimization loop has started with max epochs = ', max_epochs)
    start = time.time()
    D_train_during_training, losses_train, dloss_train, \
        pdeloss_train, D_test_during_training, losses_test, \
        dloss_test, pdeloss_test = optimization_loop(max_epochs, \
        NN, loss_function, D_param, pde_w, optimizer, reg, lmb, \
        scheduler, sched=sched, print_out=print_out, n_pde=n_pde)

    end = time.time()
    tot_time = end - start
    print('\nOptimization loop has ended. Total time used was:', str(datetime.timedelta(seconds=tot_time)), '\n')

# L-BFGS
if optim =='L-BFGS':
    params = list(NN.parameters()) + [D_param]

    D_during_train =[]
    dloss = []
    pdeloss = []
    total_losses = []

    test_data_list, test_input_list = get_test_data()
    train_data_list, train_input_list = get_train_data()

    train_pde_points = init_collocation_points(train_input_list[0], tmax, tmin, num_points=n_pde)

    def closure():

        # Free all intermediate values:
        optimizer.zero_grad() # Resetting the gradients to zeros
        
        """ Train """
        # Forward pass for the data:
        train_data_loss_value = data_loss(NN, train_input_list, train_data_list, loss_function)
        # Forward pass for the PDE 
        train_pde_loss_value = pde_loss_residual(train_pde_points, NN, D_param, loss_function)
        train_total_loss = train_data_loss_value  + pde_w * train_pde_loss_value

        """ Test """
        with torch.no_grad():
            # Forward pass for the data:
            test_total_loss = data_loss(NN, test_input_list, test_data_list, loss_function)


        if reg == 'L1': # Adding L1 regularization
            l1_penalty = torch.nn.L1Loss(reduction='sum') # size_average=False
            l1_reg_loss = 0
            for param in NN.parameters():
                l1_reg_loss += l1_penalty(param, torch.zeros_like(param))
            train_total_loss += lmb * l1_reg_loss
        
        elif reg == 'L2': # Adding L2 regularization
            l2_reg_term = 0
            for param in NN.parameters():
                l2_reg_term += torch.sum(param ** 2)
            train_total_loss += lmb * l2_reg_term


        # Backward pass, compute gradient w.r.t. weights and biases
        train_total_loss.backward()
        

        """ Train Log """
        # Log the train diffusion coeff. to make a figure
        D_train_during_training.append(D_param.item())
        # Log the train losses to make figures
        losses_train.append(train_total_loss.item())
        dloss_train.append(train_data_loss_value.item())
        pdeloss_train.append(train_pde_loss_value.item())
        
        """ Test Log """
        # Log the test losses to make figures
        losses_test.append(test_total_loss.item())

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




L_squared = 5648.0133
T = 172800.0
scaling_factor = L_squared / T

D_train_during_training = np.array(D_train_during_training)*scaling_factor
D_train_mean = sum(D_train_during_training[-50:])/50
print(f'Mean of last 50 D_train in SI = {D_train_mean*scaling_factor * 10**4:.4f} x 10^(-4) [mm^2 s^(-1)]')
print(f'The last D_train in SI = {D_train_during_training[-1]*scaling_factor * 10**4:.4f} x 10^(-4) [mm^2 s^(-1)]')


print('Started plotting and saving the figs. This may take a minute.')
train_plot_total_losses(losses_train, dloss_train, pdeloss_train)
test_plot_total_losses(losses_test, dloss_test, pdeloss_test)
train_test_total_losses(losses_train, losses_test)
D_plot(D_train_during_training)
# plot_MRI_images(NN, train=False)
train_images()
test_data_NN_prediction(NN)
print('Finished plotting and saving the figs.\n')

# print(f'D_init = {D_init:.4f}') # Initial guess for diff. coeff. (dimensionless)
# print(f'D end = {D_param.item():.4f}') # Final diff. coeff. (dimensionless)
# print(f'D_mean end = {D_mean* 10**4:.4f} x 10^(-4)') # Mean of the last 100 diff. coeff. (still dimensionless)

# # Final and mean of the last 100 diff. coeff. (now in SI units):
# print(f'D end in SI = {D_mean*scaling_factor * 10**4:.4f} x 10^(-4) [mm^2 s^(-1)]')
# print(f'D_param end in SI = {D_param.item()*scaling_factor * 10**4:.4f} x 10^(-4) [mm^2 s^(-1)]')
