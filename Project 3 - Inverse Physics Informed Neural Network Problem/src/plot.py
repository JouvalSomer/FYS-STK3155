import matplotlib.pyplot as plt
import torch

from data import *
from optim import *

os.makedirs('results', exist_ok=True)

""" Plot losses during traning """
def plot_total_losses(total_losses, dloss, pdeloss):
    plt.rcParams.update({'figure.max_open_warning': 0})
    plt.figure(dpi = 200)
    plt.semilogy(total_losses, label='total loss', linestyle='dashed', zorder=100)
    dloss = torch.tensor(dloss)
    dloss = dloss.cpu()
    pdeloss = torch.tensor(pdeloss)
    pdeloss = pdeloss.cpu()
    plt.semilogy(dloss, label='data')
    plt.semilogy(pdeloss, label='pde')
    plt.ylabel("Loss")
    plt.xlabel("Iteration")
    plt.legend()
    plt.savefig(f'results/loss_plot_A.png')
    plt.clf()

'''Plot D during training'''
def D_plot(D_during_train):
    plt.figure(dpi=200)
    plt.plot(D_during_train)
    plt.ylabel("D")
    plt.xlabel("Iteration")
    plt.savefig(f'results/D_plot_A.png')
    plt.clf()

''' Plot Simulated MRI images and predicted images '''
def plot_MRI_images(u_nn, train=False):
    os.makedirs('results/MRI_images', exist_ok=True)

    dataset = "brain2dsmooth10"
    path_to_data, roi = import_data(dataset, train)
    images = load_images(path_to_data, dataset)
    coordinate_grid = make_coordinate_grid(images)
    datadict = get_input_output_pairs(coordinate_grid, mask=roi, images=images)
    time_keys = get_timedata(coordinate_grid, roi, images)

    for i,t in enumerate(time_keys):
        xyt = torch.tensor(datadict[t][0]).float()
        xyt_cpu = xyt.cpu()

        """ Simulated MRI images """
        plt.figure(dpi=200)

        plt.plot(xyt_cpu[:, 0], xyt_cpu[:, 1], marker=".", linewidth=0, markersize=0.1, color="k")
        plt.scatter(xyt_cpu[:, 0], xyt_cpu[:, 1], c=datadict[t][1], vmin=0., vmax=1.)

        plt.colorbar()
        plt.xlabel("x")
        plt.ylabel("y")
        plt.savefig(f'results/MRI_images/{i}_true.png')
        plt.clf()


        """ Predicted MRI images """
        plt.figure(dpi=200)

        xyt[:, -1] = float(t)
        u_prediction=u_nn(xyt)
        
        xyt = xyt.cpu()
        u_prediction = u_prediction.cpu()

        plt.plot(xyt[:, 0], xyt[:, 1], marker=".", linewidth=0, markersize=0.1, color="k")
        plt.scatter(xyt[:, 0], xyt[:, 1], c=np.squeeze(u_prediction.detach().numpy(),1), vmin=0., vmax=1.)
        
        plt.colorbar()
        plt.xlabel("x")
        plt.ylabel("y")
        plt.savefig(f'results/MRI_images/{i}_pred.png')
        plt.clf()


