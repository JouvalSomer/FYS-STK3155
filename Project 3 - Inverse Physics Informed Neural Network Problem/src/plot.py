import matplotlib.pyplot as plt
import torch

from data import *
from optim import *

def make_results_folder(optim):
    os.makedirs(f'results_{optim}', exist_ok=True)
# plt.rcParams.update({
#     "text.usetex": True},
#     "font.family": "serif",
#     "font.serif": ["ComputerModern"])

plt.rcParams['figure.figsize'] = (12,9)

plt.rc('axes', facecolor='whitesmoke', edgecolor='none',
    axisbelow=True, grid=True)
plt.rc('grid', color='w', linestyle='solid')
plt.rc('lines', linewidth=2)

plt.rcParams.update({'font.size': 20})

""" Plot losses during traning """
def train_plot_total_losses(total_losses, dloss, pdeloss, pde_w, optim):
    plt.rcParams.update({'figure.max_open_warning': 0})
    plt.figure(dpi = 200)
    total_losses = torch.tensor(total_losses)
    total_losses = total_losses.cpu()
    plt.semilogy(total_losses, label=f'Total loss = data_loss + {str(pde_w)} * pde_loss', linestyle='dashed', zorder=100)
    
    dloss = torch.tensor(dloss)
    dloss = dloss.cpu()
    plt.semilogy(dloss, label='Data loss')

    if len(pdeloss) >= 1:
        pdeloss = torch.tensor(pdeloss)
        pdeloss = pdeloss.cpu()
        plt.semilogy(pdeloss, label='PDE loss')

    plt.title('Train losses')
    plt.ylabel("Loss", fontsize=20 , labelpad=10)
    plt.xlabel("Iteration", fontsize=20 , labelpad=10)
    plt.legend(framealpha=0.9, facecolor=(1, 1, 1, 1))
    plt.tight_layout()
    plt.savefig(f'results_{optim}/train_loss_plot_{int(pde_w * 100)}.png')
    plt.clf()

def test_plot_total_losses(total_losses, pde_w, optim):
    plt.rcParams.update({'figure.max_open_warning': 0})
    plt.figure(dpi = 200)
    total_losses = torch.tensor(total_losses)
    total_losses = total_losses.cpu()
    plt.semilogy(total_losses, label='total loss')
    plt.title('Total loss on the test data', pad=15)
    plt.ylabel("Loss", fontsize=20 , labelpad=10)
    plt.xlabel("Iteration", fontsize=20 , labelpad=10)
    plt.legend(framealpha=0.9, facecolor=(1, 1, 1, 1))
    plt.tight_layout()
    plt.savefig(f'results_{optim}/test_loss_plot_{int(pde_w * 100)}.png')
    plt.clf()

def train_test_total_losses(train_total_losses, test_total_losses, pde_w, optim):
    plt.rcParams.update({'figure.max_open_warning': 0})
    plt.figure(dpi = 200)
    train_total_losses = torch.tensor(train_total_losses)
    test_total_losses = torch.tensor(test_total_losses)

    plt.semilogy(test_total_losses, label='test')
    plt.semilogy(train_total_losses, label='train')
    plt.title('Test/train loss', pad=15)
    plt.ylabel("Loss", fontsize=20 , labelpad=10)
    plt.xlabel("Iteration", fontsize=20 , labelpad=10)
    plt.legend(framealpha=0.9, facecolor=(1, 1, 1, 1))
    plt.savefig(f'results_{optim}/test_train_losses_{int(pde_w * 100)}.png')
    plt.tight_layout()
    plt.clf()


'''Plot D during training'''
def D_plot(D_during_train, pde_w, optim):
    plt.figure(dpi=200)
    plt.title("PINN's diffusion coefficient estimate against iterations", pad=15)
    plt.semilogy(D_during_train, label='The diffusion coefficient')
    plt.ylabel("Estimated diffusion coefficient", fontsize=20 , labelpad=10)
    plt.xlabel("Iterations", fontsize=20 , labelpad=10)
    plt.legend(framealpha=0.9, facecolor=(1, 1, 1, 1))
    plt.tight_layout()
    plt.savefig(f'results_{optim}/D_plot_{int(pde_w * 100)}.png')
    plt.clf()


""" Plot of the traning images """
def train_images():
    os.makedirs(f'train_images', exist_ok=True)
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
        plt.plot(xyt_cpu[..., 0], xyt_cpu[..., 1], marker=".", linewidth=0, markersize=0.2, color="k")
        plt.scatter(xyt_cpu[..., 0], xyt_cpu[..., 1], c=datadict[t][1], vmin=-1., vmax=1., linewidths=2.5)
        plt.xlabel("x", fontsize=20, labelpad=10)
        plt.ylabel("y", fontsize=20, labelpad=10)
        plt.title(f'Synthetic data at {true_time_keys[i]} hours', pad=15)
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(f'train_images/{str(int(true_time_keys[i]))}_true.png')
        plt.clf()


""" Comparison plot of the test prediction and the test data """
def test_data_NN_prediction(NN, pde_w, optim):
    os.makedirs(f'results_{optim}/test_data_NN_prediction', exist_ok=True)

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
    axs[0, 0].set_title(f"PINN prediction at time = {true_time_keys[7]} hours", fontsize=14, pad=7)

    img2 = axs[1, 0].plot(xyt2[:, 0], xyt2[:, 1], marker=".", linewidth=0, markersize=0.1, color="k")
    img2 = axs[1, 0].scatter(xyt2[:, 0], xyt2[:, 1], c=np.squeeze(prediction2.detach().numpy(),1), vmin=-1., vmax=1.)
    axs[1, 0].set_title(f"PINN prediction at time = {true_time_keys[14]} hours", fontsize=14, pad=7)

    img3 = axs[0, 1].plot(xyt1[:, 0], xyt1[:, 1], marker=".", linewidth=0, markersize=0.1, color="k")
    img3 = axs[0, 1].scatter(xyt1[:, 0], xyt1[:, 1], c=datadict[test_time_keys[0]][1], vmin=-1., vmax=1.)
    axs[0, 1].set_title(f"Data at time = {true_time_keys[7]} hours", fontsize=14, pad=7)

    img4 = axs[1, 1].plot(xyt2[:, 0], xyt2[:, 1], marker=".", linewidth=0, markersize=0.1, color="k")
    img4 = axs[1, 1].scatter(xyt2[:, 0], xyt2[:, 1], c=datadict[test_time_keys[1]][1], vmin=-1., vmax=1.)
    axs[1, 1].set_title(f"Data at time = {true_time_keys[14]} hours", fontsize=14, pad=7)

    fig.suptitle('Test and data images', y=0.98)

    cbar = fig.colorbar(img4, ax=axs, orientation='vertical', fraction=0.046, pad=0.1)
    plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
    plt.savefig(f'results_{optim}/test_data_NN_prediction/test_data_NN_prediction_{int(pde_w * 100)}.png')
    plt.clf()



def bvto(losses_train_ANN, losses_test_ANN, losses_train_PINN, losses_test_PINN, pde_w, optim):
    
    losses_train_ANN = torch.tensor(losses_train_ANN)
    losses_test_ANN = torch.tensor(losses_test_ANN)
    losses_train_ANN = torch.tensor(losses_train_PINN)
    losses_test_ANN = torch.tensor(losses_test_PINN)

    fig, ax = plt.subplots(2, 1, figsize=[12,9], sharex=True, sharey=True)
    fig.suptitle('Train/test loss - ANN vs PINN', y=0.98)

    # Plot something in the first subplot
    ax[0].plot(losses_train_ANN, label='ANN train loss')
    ax[0].plot(losses_test_ANN, label='ANN test loss')
    ax[0].set_yscale('log')
    ax[0].set_ylabel('Loss', fontsize=20 ,labelpad=10)
    ax[0].set_title('ANN', pad=5)
    ax[0].legend(framealpha=0.9, facecolor=(1, 1, 1, 1))


    # Plot something in the first subplot
    ax[1].plot(losses_train_PINN, label='PINN train loss')
    ax[1].plot(losses_test_PINN, label='PINN test loss')
    ax[1].set_yscale('log')
    ax[1].set_ylabel('Loss', fontsize=20 , labelpad=10)
    ax[1].set_title('PINN', pad=5)
    ax[1].set_xlabel('Iterations', fontsize=20 , labelpad=10)
    ax[1].legend(framealpha=0.9, facecolor=(1, 1, 1, 1))

    plt.savefig(f'results_{optim}/bvto_{int(pde_w * 100)}.png')
    plt.clf()
