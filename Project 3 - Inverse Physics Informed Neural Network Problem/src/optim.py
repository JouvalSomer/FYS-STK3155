
import torch
import copy
from data import *

D_train_during_training =[]
dloss_train = []
pdeloss_train = []
losses_train = []

D_test_during_training =[]
dloss_test = []
pdeloss_test = []
losses_test = []

tmin, tmax = get_min_max_time()

def data_loss(nn, input_list, data_list, loss_function):
    loss = 0.
    count = 0
    for input_, data in zip(input_list, data_list):
        count += 1
        predictions = torch.squeeze(nn(input_)) # Squeeze shape from (N,1) to (N)
        loss = loss + loss_function(predictions, data)
    return loss/count

def pde_loss_residual(coords, nn, D, loss_function):
        # assert isinstance(D, torch.nn.Parameter)
        assert coords.shape[-1] == 3, "array should have size N x 3"
        
        coords.requires_grad = True
        output = nn(coords).squeeze()

        ones = torch.ones_like(output)

        output_grad, = torch.autograd.grad(outputs=output,
                                        inputs=coords,
                                        grad_outputs=ones,
                                        create_graph=True)

        doutput_dt = output_grad[..., -1]
        doutput_dx = output_grad[..., 0]
        doutput_dy = output_grad[..., 1]
        
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

        return loss_function(residual, torch.zeros_like(residual))

def optimization_loop(max_epochs, NN, loss_function, D_param, pde_w, optimizer, reg, lmb, scheduler, sched=False, print_out=False, n_pde=int(1e5)):
    
    test_data_list, test_input_list = get_test_data()
    train_data_list, train_input_list = get_train_data()

    train_pde_points = init_collocation_points(train_input_list[0], tmax, tmin, num_points=n_pde)

    from tqdm import tqdm
    for i in tqdm(range(max_epochs)):

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


        # Update the weights and biases 
        optimizer.step()
        if sched:
            scheduler.step()
        
        if print_out:
            if i % int(max_epochs/10) == 0:
                print('\nIteration = ',i)
                print(f'Total traning loss = {train_total_loss.item():.4f}')
                print(f"Diff. coeff. = {D_param.item()}")

    return D_train_during_training, losses_train, dloss_train, pdeloss_train, D_test_during_training, losses_test, dloss_test, pdeloss_test

            
