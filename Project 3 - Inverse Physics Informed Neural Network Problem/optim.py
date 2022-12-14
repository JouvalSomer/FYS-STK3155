
import torch

from data import *

D_during_train =[]
dloss = []
pdeloss = []
losses = []

tmax = float(max(datadict.keys()))
tmin = float(min(datadict.keys()))

def data_loss(nn, input_list, data_list, loss_function):
    loss = 0.
    for input_, data in zip(input_list, data_list):
        predictions = torch.squeeze(nn(input_)) # Squeeze shape from (N,1) to (N)
        loss = loss + loss_function(predictions, data)
    return loss

def pde_loss_residual(coords, nn, D, loss_function):
        
        assert isinstance(D, torch.nn.Parameter)
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

def optimization_loop(max_iters, u_nn, loss_function, D_param, pde_w, optimizer, reg, lmb, scheduler, sched=False, print_out=False, n_pde=int(1e5)):
    pde_points = init_collocation_points(input_list[0], tmax, tmin, num_points=n_pde)
    from tqdm import tqdm
    for i in tqdm(range(max_iters)):

        # Free all intermediate values:
        optimizer.zero_grad() # Resetting the gradients to zeros
        
        # Forward pass for the data:
        data_loss_value = data_loss(u_nn, input_list, data_list, loss_function)
        
        # Forward pass for the PDE 
        pde_loss_value = pde_loss_residual(pde_points, u_nn, D_param, loss_function)
        
        total_loss = data_loss_value + pde_w * pde_loss_value

        if reg == 'L1': # Adding L1 regularization
            l1_penalty = torch.nn.L1Loss(reduction='sum') # size_average=False
            reg_loss = 0
            for param in u_nn.parameters():
                reg_loss += l1_penalty(param, torch.zeros_like(param))
            total_loss += lmb * reg_loss
        
        # Backward pass, compute gradient w.r.t. weights and biases
        total_loss.backward()
        
        # Log the diffusion coeff. to make a figure
        D_during_train.append(D_param.item())

        # Log the losses to make figures
        losses.append(total_loss.item())
        dloss.append(data_loss_value.item())
        pdeloss.append(pde_loss_value.item())
        
        # Update the weights and biases
        optimizer.step()
        if sched:
            scheduler.step()
        
        if print_out:
            if i % int(max_iters/10) == 0:
                print('iteration = ',i)
                print('Loss = ',total_loss.item())
                print(f"D = {D_param.item()}")

    return D_during_train, losses, dloss, pdeloss