import matplotlib.pyplot as plt
import torch

from data import *
from optim import *

os.makedirs('results', exist_ok=True)

""" Plot losses during traning """
def train_plot_total_losses(total_losses, dloss, pdeloss):
    plt.rcParams.update({'figure.max_open_warning': 0})
    plt.figure(dpi = 200)
    total_losses = torch.tensor(total_losses)
    total_losses = total_losses.cpu()
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
    plt.savefig(f'results/train_loss_plot.png')
    plt.clf()

def test_plot_total_losses(total_losses, dloss, pdeloss):
    plt.rcParams.update({'figure.max_open_warning': 0})
    plt.figure(dpi = 200)
    total_losses = torch.tensor(total_losses)
    total_losses = total_losses.cpu()
    plt.semilogy(total_losses, label='total loss', linestyle='dashed', zorder=100)
    plt.ylabel("Loss")
    plt.xlabel("Iteration")
    plt.legend()
    plt.savefig(f'results/test_loss_plot.png')
    plt.clf()

def train_test_total_losses(train_total_losses, test_total_losses):
    plt.rcParams.update({'figure.max_open_warning': 0})
    plt.figure(dpi = 200)
    train_total_losses = torch.tensor(train_total_losses)
    test_total_losses = torch.tensor(test_total_losses)

    plt.semilogy(test_total_losses, label='test')
    plt.semilogy(train_total_losses, label='train')
    plt.ylabel("Loss")
    plt.xlabel("Iteration")
    plt.legend()
    plt.savefig(f'results/test_train_losses.png')
    plt.clf()


'''Plot D during training'''
def D_plot(D_during_train):
    plt.figure(dpi=200)
    plt.semilogy(D_during_train)
    plt.ylabel("D")
    plt.xlabel("Iteration")
    plt.savefig(f'results/D_plot.png')
    plt.clf()


""" Plot of the traning images """
def train_images():
    os.makedirs('results/train_images', exist_ok=True)
    dataset = "brain2dsmooth10"
    path_to_data, roi = import_data(dataset, mask=True)
    images = load_images(path_to_data, dataset)
    coordinate_grid = make_coordinate_grid(images)
    datadict, true_time_keys = get_input_output_pairs(coordinate_grid, mask=roi, images=images)
    train_time_keys = get_train_time_keys()

    for i,t in enumerate(train_time_keys):
        xyt = torch.tensor(datadict[t][0]).float()
        xyt_cpu = xyt.cpu()
        plt.figure(dpi=200)
        plt.plot(xyt_cpu[..., 0], xyt_cpu[..., 1], marker=".", linewidth=0, markersize=0.1, color="k")
        plt.scatter(xyt_cpu[..., 0], xyt_cpu[..., 1], c=datadict[t][1], vmin=-1., vmax=1.)
        plt.xlabel("x", fontsize=12)
        plt.ylabel("y", fontsize=12)
        plt.colorbar()
        plt.savefig(f'results/train_images/{str(int(true_time_keys[i]))}_true.png')
        plt.clf()


""" Comparison plot of the test prediction and the test data """
def test_data_NN_prediction(NN):
    os.makedirs('results/test_data_NN_prediction', exist_ok=True)

    dataset = "brain2dsmooth10"
    path_to_data, roi = import_data(dataset, mask=True)
    images = load_images(path_to_data, dataset)
    coordinate_grid = make_coordinate_grid(images)
    datadict, true_time_keys = get_input_output_pairs(coordinate_grid, mask=roi, images=images)

    test_time_keys = get_test_time_keys()

    fig, axs = plt.subplots(2, 2, figsize=[12,9], sharex=True)


    xyt1  = torch.tensor(datadict[test_time_keys[0]][0]).float()
    xyt2  = torch.tensor(datadict[test_time_keys[1]][0]).float()

    xyt1[:, -1] = float(test_time_keys[0])
    xyt2[:, -1] = float(test_time_keys[1])

    prediction1 = NN(xyt1)
    prediction2 = NN(xyt2)

    xyt1 = xyt1.cpu()
    xyt2 = xyt2.cpu()
    
    prediction1 = prediction1.cpu()
    prediction2 = prediction2.cpu()


    img1 = axs[0, 0].plot(xyt1[:, 0], xyt1[:, 1], marker=".", linewidth=0, markersize=0.1, color="k")
    img1 = axs[0, 0].scatter(xyt1[:, 0], xyt1[:, 1], c=np.squeeze(prediction1.detach().numpy(),1), vmin=-1., vmax=1.)
    axs[0, 0].set_title(f"PINN prediction at time = {true_time_keys[7]} hours", pad=7)

    img2 = axs[1, 0].plot(xyt2[:, 0], xyt2[:, 1], marker=".", linewidth=0, markersize=0.1, color="k")
    img2 = axs[1, 0].scatter(xyt2[:, 0], xyt2[:, 1], c=np.squeeze(prediction2.detach().numpy(),1), vmin=-1., vmax=1.)
    axs[1, 0].set_title(f"PINN prediction at time = {true_time_keys[14]} hours", pad=7)

    img3 = axs[0, 1].plot(xyt1[:, 0], xyt1[:, 1], marker=".", linewidth=0, markersize=0.1, color="k")
    img3 = axs[0, 1].scatter(xyt1[:, 0], xyt1[:, 1], c=datadict[test_time_keys[0]][1], vmin=-1., vmax=1.)
    axs[0, 1].set_title(f"Data at time = {true_time_keys[7]} hours", pad=7)

    img4 = axs[1, 1].plot(xyt2[:, 0], xyt2[:, 1], marker=".", linewidth=0, markersize=0.1, color="k")
    img4 = axs[1, 1].scatter(xyt2[:, 0], xyt2[:, 1], c=datadict[test_time_keys[1]][1], vmin=-1., vmax=1.)
    axs[1, 1].set_title(f"Data at time = {true_time_keys[14]} hours", pad=7)

    fig.suptitle('Test and data images', y=0.98)

    cbar = fig.colorbar(img4, ax=axs, orientation='vertical', fraction=0.046, pad=0.1)
    plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
    plt.savefig(f'results/test_data_NN_prediction/test_data_NN_prediction.png')
    plt.clf()
