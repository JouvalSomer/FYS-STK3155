import os, glob
import numpy as np
import torch

spatial_dim = 2

""" Check if there is a GPU available """
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU ", torch.cuda.get_device_name())
    using_gpu = True
else:
    device = torch.device("cpu")
    print("Using CPU")
    using_gpu = False


data_list = []
input_list = []

def import_data(dataset, train):
    path_to_data = os.getcwd() + "/data/"
    brainmask = np.load(path_to_data + dataset +  "/masks/mask.npy")
    box = np.load(path_to_data + dataset +  "/masks/box.npy")

    if train:
        roi = np.logical_not(box) * brainmask # Apply the inverse mask (Traning set)
    else:
        roi = box * brainmask # Apply the mask (Test set)

    return path_to_data, roi

def load_images(path_to_data, dataset):
    path_to_concentrations = path_to_data + dataset +  "/concentrations/"
    images = {}
    for cfile in os.listdir(path_to_concentrations):
        c = np.load(path_to_concentrations + cfile)
        images[cfile[:-4]] = c
    return images

def make_coordinate_grid(images):
    """ Create a (n x n x 2) array where arr[i,j, :] = (x_i, y_i) is the position of voxel (i,j)"""
    n = 256
    # We want to assign coordinates to every voxel, so the shape of the meshgrid has to be the same as the image
    assert n == images[next(iter(images.keys()))].shape[0]
    assert n == images[next(iter(images.keys()))].shape[1]

    coordinate_axis = np.linspace(-1, 1, n) # Range from -1 to 1 for activation function (Tanh)
    XX, YY = np.meshgrid(coordinate_axis, coordinate_axis, indexing='ij')
    arr = np.array([XX, YY])
    coordinate_grid = np.swapaxes(arr, 0, 1)
    coordinate_grid = np.swapaxes(coordinate_grid, 1, 2)

    return coordinate_grid

def get_input_output_pairs(coordinate_grid, mask, images):
    input_output_pairs = {}

    xy = coordinate_grid[mask] # The mast alters the min and max values. So the grid is scaled again below

    # Thise are made global because they are used to unscale the data outside this function
    global input_max 
    global input_min 
    input_max = np.max(xy)
    input_min = np.min(xy)
    max_min = input_max - input_min
    xy_scaled = 2*(xy - np.min(xy))/max_min - 1 # Scaling the data between -1 and 1

    true_min = np.min([np.min(images[key][mask]) for key in images.keys()])
    true_max = np.max([np.max(images[key][mask]) for key in images.keys()])
    max_min_images = true_max - true_min

    for timekey, image in images.items():

        xyt_true = image[mask]
        xyt_true_scaled = 2 * (xyt_true - true_min)/max_min_images - 1 # Scaling the true consentration

        timekey = float(timekey)
        scaled_timekey = round(2*(timekey)/(45.60) - 1, 4) # Scaling the time from 0 to 45.6 hours to -1, 1 
        xyt = np.zeros((xy_scaled.shape[0], 3))
        xyt[..., :2] = xy_scaled
        xyt[..., -1] = scaled_timekey

        input_output_pairs[scaled_timekey] = (xyt, xyt_true_scaled) 

    return input_output_pairs

def get_timedata(coordinate_grid, mask, images):
    """ Returns the scaled timekeys """
    input_output_pairs = get_input_output_pairs(coordinate_grid, mask, images)
    scaled_timekey = list(input_output_pairs.keys())
    return scaled_timekey 

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
path_to_data, roi = import_data(dataset, train=False)
images = load_images(path_to_data, dataset)
coordinate_grid = make_coordinate_grid(images)
datadict = get_input_output_pairs(coordinate_grid, mask=roi, images=images)
ts = get_timedata(coordinate_grid, roi, images)
data_list, input_list = create_space_time_tensor(ts, datadict, data_list, input_list, using_gpu, spatial_dim)




''' Residual points '''
def init_collocation_points(coords, t_max, t_min, num_points=int(1e5)):
    with torch.no_grad():
        assert len(coords.shape) == 2
        torch.manual_seed(155) #Seed for rand. functions
        random_ints = torch.randint(high=coords.size(0), size=(num_points,), device=coords.device)    
        coords = coords[random_ints, :] # Choose random coords.
        
        torch.manual_seed(155) #Seed for rand. functions
        a = (np.random.rand(coords.shape[0]))
        
        random_times = torch.from_numpy(a).to(coords.device)
        t = (random_times * (t_max - t_min) + t_min)

        coords[..., -1] = t

    return coords