
import os, glob
import numpy as np
import torch
torch.manual_seed(155) #Seed for rand. functions


""" Check if there is a GPU available """
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU ", torch.cuda.get_device_name())
    using_gpu = True
else:
    device = torch.device("cpu")
    print("Using CPU")
    using_gpu = False

D_during_train =[]
dloss = []
pdeloss = []
data_list = []
input_list = []
losses = []
counter = 0
spatial_dim = 2

''' Import data'''
def import_data(dataset):
    path_to_data = os.getcwd() + "/data/"
    brainmask = np.load(path_to_data + dataset +  "/masks/mask.npy")
    box = np.load(path_to_data + dataset +  "/masks/box.npy")
    roi = brainmask * box # Apply the mask
    return path_to_data, roi

def load_images(path_to_data, dataset):
    path_to_concentrations = path_to_data + dataset +  "/concentrations/"
    images = {}
    for cfile in os.listdir(path_to_concentrations):
        c = np.load(path_to_concentrations + cfile)
        images[cfile[:-4]] = c
    return images


''' Define grid '''
def make_coordinate_grid(images):
    """ Create a (n x n x 2) array where arr[i,j, :] = (x_i, y_i) is the position of voxel (i,j)"""
    n = 256
    # We want to assign coordinates to every voxel, so the shape of the meshgrid has to be the same as the image
    assert n == images[next(iter(images.keys()))].shape[0]
    assert n == images[next(iter(images.keys()))].shape[1]

    coordinate_axis = np.linspace(0, 1, n)
    XX, YY = np.meshgrid(coordinate_axis, coordinate_axis, indexing='ij')
    arr = np.array([XX, YY])
    coordinate_grid = np.swapaxes(arr, 0, 1)
    coordinate_grid = np.swapaxes(coordinate_grid, 1, 2)
    return coordinate_grid

# def make_coordinate_grid(images):
#     shape = 256
#     """ Create a (n x n x 2) array where arr[i,j, :] = (x_i, y_i) is the position of voxel (i,j)"""
#     lx = 1
#     coordinate_axis =np.linspace(0,lx, shape)
#     XX,YY = np.meshgrid(coordinate_axis, coordinate_axis,indexing='ij')
#     arr = np.array([XX,YY])
#     coordinate_grid = np.swapaxes(arr, 0, 1)
#     coordinate_grid = np.swapaxes(coordinate_grid, 1, 2)
#     dimensions = [133.05135033600635,169.79950394304987]
#     coordinate_grid[:, :,0] *= dimensions[0]
#     coordinate_grid[:, :,1] *= dimensions[1]
#     return coordinate_grid

def get_input_output_pairs(coordinate_grid, mask, images):
    input_output_pairs = {}

    xy = coordinate_grid[mask]
    global input_max
    global input_min 
    input_max = np.max(xy)
    input_min = np.min(xy)
    max_min = input_max - input_min
    xy_scaled = (xy - np.min(xy))/max_min

    # global true_max
    # global true_min
    true_min = np.min([np.min(images[key][mask]) for key in images.keys()])
    true_max = np.max([np.max(images[key][mask]) for key in images.keys()])
    max_min_images = true_max - true_min

    for timekey, image in images.items():
        xyt_true = image[mask]
        # print(np.min(xyt_true), np.max(xyt_true))
        # input()
        # xyt_true_scaled = 2 * (xyt_true - true_min)/max_min_images - 1

        timekey = float(timekey)
        scaled_timekey = round((timekey)/(45.60), 4)
        xyt = np.zeros((xy_scaled.shape[0], 3))
        xyt[..., :2] = xy_scaled
        xyt[..., -1] = scaled_timekey
        input_output_pairs[scaled_timekey] = (xyt, xyt_true)
    return input_output_pairs

'''Get timedata'''
def get_timedata(path_to_data, dataset):
    l = glob.glob(path_to_data +  dataset +  "/concentrations/*")
    ts = []
    for f in l:
        t = f.split(".npy")[0]
        t = t.split('\\')[-1]
        t = round((float(t))/(45.60), 4)
        ts.append(t)
    return ts

''' Create space-time tensor '''
def create_space_time_tensor(ts, datadict, data_list, input_list, using_gpu, spatial_dim):
    for current_time in ts:
        xyt = torch.tensor(datadict[current_time][0]).float()
        if using_gpu == True:
            xyt = xyt.cuda()
        assert spatial_dim == 2

        u_true = torch.tensor(datadict[current_time][1]).float()
        
        if using_gpu == True:
            u_true = u_true.cuda()
        
        data_list.append(u_true)
        input_list.append(xyt)
    return data_list, input_list
    
dataset = "brain2dsmooth10"
path_to_data, roi = import_data(dataset)
images = load_images(path_to_data, dataset)
# def scale_image(images, mask):
#     for timekey, image in images.items():
#         xyt_true = image[mask]

# print(list(images.keys()))
# input()
# coordinate_grid = make_coordinate_grid(shape=len(images['00.00']))
coordinate_grid = make_coordinate_grid(images)

datadict = get_input_output_pairs(coordinate_grid, mask=roi, images=images)

ts = get_timedata(path_to_data, dataset)

data_list, input_list = create_space_time_tensor(ts, datadict, data_list, input_list, using_gpu, spatial_dim)

tmax = float(max(datadict.keys()))
tmin = float(min(datadict.keys()))
# b = list(zip(*list(datadict.values())))[0]
# print(np.min(np.array(b)))
# input()


# xyt = torch.tensor(datadict['45.60'][0]).float()

# maximum = torch.max(xyt, axis=0)[0]
# minimum = torch.min(xyt, axis=0)[0]
# maximum[..., -1] = tmax
# minimum[..., -1] = tmin

# def InputNormalization(input):
#     """ Centers the data between [-1, 1] """
#     # re = 2 * (input - minimum) / (maximum - minimum) - 1
#     re = (input - minimum) / (maximum - minimum)

#     # print(torch.min(re), torch.max(re))
#     # input()
#     return re
    
# """ The FFNN """
# class Net(torch.nn.Module):
#     def __init__(self, num_hidden_units, num_hidden_layers, inputs, outputs=1, inputnormalization=None):
        
#         super(Net, self).__init__()        
#         self.num_hidden_units = num_hidden_units
#         self.num_hidden_layers = num_hidden_layers
#         self.inputnormalization = inputnormalization

#         # Dimensions of input/output
#         self.inputs =  inputs
#         self.outputs = outputs
        
#         # Create the inputlayer 
#         self.input_layer = torch.nn.Linear(self.inputs, self.num_hidden_units)
#         torch.nn.init.xavier_uniform_(self.input_layer.weight)


#         # Create the hidden layers
#         self.linear_layers = torch.nn.ModuleList([torch.nn.Linear(
#             self.num_hidden_units, self.num_hidden_units)
#             for i in range(self.num_hidden_layers - 1)])

#         # Create the output layer
#         self.output_layer = torch.nn.Linear(self.num_hidden_units, self.outputs)
        
#         # Use hyperbolic tangent as activation:
#         """ Want to try: Tanh, ReLU, LeakyReLU and Sigmoid """
#         self.activation = torch.nn.Tanh()
        
#         self.Initialize_weights()

#     def forward(self, x):
#         """[Compute NN output]

#         Args:
#             x ([torch.Tensor]): input tensor
#         Returns:
#             [torch.Tensor]: [The NN output]
#         """
#         # # Transform the shape of the Tensor to match what is expected by torch.nn.Linear
#         # if self.inputnormalization is not None:
#         #     x = InputNormalization(x)
        
#         """ (n,) -> (n,1)  """
#         x = torch.unsqueeze(x, 1) 

#         # x[..., -1] = x[..., -1] / tmax
    
#         out = self.input_layer(x)
        
#         # The first hidden layer:
#         out = self.activation(out)

#         # The other hidden layers:
#         for i, linearLayer in enumerate(self.linear_layers):
#             out = linearLayer(out)
#             out = self.activation(out)

#         # No activation in the output layer:
#         out = self.output_layer(out)

#         """ (n,1) -> (n,)  """
#         out = torch.squeeze(out, 1)

#         return out
    
#     def Initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, torch.nn.Linear):
#                 torch.nn.init.xavier_uniform_(m.weight)
#                 if m.bias is not None:
#                     torch.nn.init.constant_(m.bias, 0.01)

def optimization_loop(max_iters, u_nn, loss_function, D_param, pde_w, optimizer, reg, lmb, scheduler, sched=False, print_out=False, n_pde=int(1e5)):
    pde_points = init_collocation_points(input_list[0], tmax, tmin, num_points=n_pde)
    from tqdm import tqdm
    for i in tqdm(range(max_iters)):

        # Free all intermediate values:
        optimizer.zero_grad() # Resetting the gradients to zeros
        
        # Forward pass for the data:
        data_loss_value = data_loss(u_nn, input_list, data_list, loss_function)
        
        # Forward pass for the PDE 
        pde_loss_value = pderesidual(pde_points, u_nn, D_param, loss_function)
        

        total_loss = data_loss_value + pde_w * pde_loss_value

        if reg == 'L1':
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



''' Residual points '''
def init_collocation_points(coords, t_max, t_min, num_points=int(1e5)):
    with torch.no_grad():

        assert len(coords.shape) == 2, "Assert mask has been applied"

        random_ints = torch.randint(high=coords.size(0), size=(num_points,), device=coords.device)    
        coords = coords[random_ints, :]
    
        a = (np.random.rand(coords.shape[0]))
        
        random_times = torch.from_numpy(a).to(coords.device)
        t = (random_times * (t_max - t_min) + t_min)

        coords[..., -1] = t

        # print("Initialized collocation points with mean t = ",
        #     format(torch.mean(t).item(), ".2f"),
        #     ", min t = ", format(torch.min(t).item(), ".2f"),
        #     ", max t = ", format(torch.max(t).item(), ".2f"))

    return coords



''' Losses '''
def data_loss(nn, input_list, data_list, loss_function):
    loss = 0.
    # Evaluate the NN at the boundary:
    for input_, data in zip(input_list, data_list):
        # print(input_.shape,'\n')
        # print(torch.min(input_[:,:2]), torch.max(input_[:,:2]))
        predictions = torch.squeeze(nn(input_))
        loss = loss + loss_function(predictions, data)
    
    # print('ferdig')
    return loss

def pderesidual(coords, nn, D, loss_function):
        """
        coords = pde_points
        nn = neural network
        D = diffusion coefficient
        
        """
        
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


def get_datadict():
    return datadict