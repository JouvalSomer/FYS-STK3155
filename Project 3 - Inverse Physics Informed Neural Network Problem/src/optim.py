
import torch
import copy
from data import *



tmin, tmax = get_min_max_time()

L_squared = 5648.0133
T = 172800.0
scaling_factor = L_squared / T

def data_loss(nn, input_list, data_list, loss_function_NN):
    loss = 0.
    count = 0
    for input_, data in zip(input_list, data_list):
        count += 1
        predictions = torch.squeeze(nn(input_)) # Squeeze shape from (N,1) to (N)
        loss = loss + loss_function_NN(predictions, data)
    return loss/count

def pde_loss_residual(coords, nn, D, loss_function_PDE):

        assert isinstance(D, torch.nn.Parameter), "D should be a parameter of the network."
        assert coords.shape[-1] == 3, "Array should have size N x 3."

        coords.requires_grad = True
        output = nn(coords).squeeze() # Forward pass of coords through the network

        ones = torch.ones_like(output)

        # Take the gradient of the output
        output_grad, = torch.autograd.grad(outputs=output,
                                        inputs=coords,
                                        grad_outputs=ones,
                                        create_graph=True)

        doutput_dx = output_grad[..., 0]
        doutput_dy = output_grad[..., 1]
        doutput_dt = output_grad[..., -1]

        ddoutput_dxx, = torch.autograd.grad(outputs=doutput_dx,
                                            inputs=coords,
                                            grad_outputs=ones,
                                            create_graph=True)

        ddoutput_dyy, = torch.autograd.grad(outputs=doutput_dy,
                                            inputs=coords,
                                            grad_outputs=ones,
                                            create_graph=True)

        ddoutput_dxx = ddoutput_dxx[..., 0]
        ddoutput_dyy = ddoutput_dyy[..., 1]

        laplacian = (ddoutput_dxx + ddoutput_dyy)

        residual = doutput_dt - D * laplacian

        assert output.shape == residual.shape

        return loss_function_PDE(residual, torch.zeros_like(residual))

def optimization_loop(max_epochs, NN, loss_function_NN, loss_function_PDE, D_param, pde_w, optimizer, scheduler, sched=False, print_out=False, n_pde=int(1e5)):
    D_train_during_training_PINN =[]
    dloss_train_PINN = []
    pdeloss_train_PINN = []

    losses_train_PINN = []
    losses_test_PINN = []

    test_data_list, test_input_list = get_test_data()
    train_data_list, train_input_list = get_train_data()

    train_pde_points = init_collocation_points(train_input_list[0], tmax, tmin, num_points=n_pde)

    from tqdm import tqdm
    for i in tqdm(range(max_epochs)):

        # Free all intermediate values:
        optimizer.zero_grad() # Resetting the gradients to zeros
        
        """ Train """
        # Forward pass for the data:
        train_data_loss_value = data_loss(NN, train_input_list, train_data_list, loss_function_NN)
        # Forward pass for the PDE 
        train_pde_loss_value = pde_loss_residual(train_pde_points, NN, D_param, loss_function_PDE)
        train_total_loss = train_data_loss_value  + pde_w * train_pde_loss_value

        """ Test """
        with torch.no_grad():
            # Forward pass for the data:
            test_total_loss = data_loss(NN, test_input_list, test_data_list, loss_function_NN)


        # Backward pass, compute gradient w.r.t. weights and biases
        train_total_loss.backward()
        

        """ Train Log """
        # Log the train diffusion coeff. to make a figure
        D_train_during_training_PINN.append(D_param.item())
        # Log the train losses to make figures
        losses_train_PINN.append(train_total_loss.item())
        dloss_train_PINN.append(train_data_loss_value.item())
        if pde_w > 0.0:
            pdeloss_train_PINN.append(train_pde_loss_value.item())
        
        """ Test Log """
        # Log the test losses to make figures
        losses_test_PINN.append(test_total_loss.item())


        # Update the weights and biases 
        optimizer.step()
        if sched:
            scheduler.step()
        
        if print_out:
            if i % int(max_epochs/10) == 0:
                print('\nIteration = ',i)
                print(f'Total traning loss = {train_total_loss.item():.4f}')
                print(f"Diff. coeff. = {D_param.item():.8f}") # * scaling_factor * 10**4:.4f x 10^(-4) [mm^2 s^(-1)]"

    return D_train_during_training_PINN, losses_train_PINN, dloss_train_PINN, pdeloss_train_PINN, losses_test_PINN

            
def optimization_loop_NN_reg(max_epochs, NN, loss_function_NN, optimizer, scheduler, sched=False):
    losses_test_ANN = []
    losses_train_ANN = []
    
    test_data_list, test_input_list = get_test_data()
    train_data_list, train_input_list = get_train_data()

    from tqdm import tqdm
    for i in tqdm(range(max_epochs)):

        # Free all intermediate values:
        optimizer.zero_grad() # Resetting the gradients to zeros
        
        """ Train """
        # Forward pass for the data:
        train_total_loss = data_loss(NN, train_input_list, train_data_list, loss_function_NN)

        """ Test """
        with torch.no_grad():
            # Forward pass for the data:
            test_total_loss = data_loss(NN, test_input_list, test_data_list, loss_function_NN)


        # Backward pass, compute gradient w.r.t. weights and biases
        train_total_loss.backward()

        """ Train Log """
        # Log the train losses to make figures
        losses_train_ANN.append(train_total_loss.item())
        
        """ Test Log """
        # Log the test losses to make figures
        losses_test_ANN.append(test_total_loss.item())

        # Update the weights and biases 
        optimizer.step()
        if sched:
            scheduler.step()


    return losses_train_ANN, losses_test_ANN
