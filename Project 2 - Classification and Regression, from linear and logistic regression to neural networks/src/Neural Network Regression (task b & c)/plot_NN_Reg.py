import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
from matplotlib import cm
from tqdm import tqdm

from activation import*
from layer import*
from train_NN import*
from func import*


def cost_heatmap(X, Y, Z, n_epochs, init_LR, decay, O_M, momentum, minibatch_size, n_minibatches, lambda_min, lambda_max, nlambdas, n_seeds, acti_func):
    plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["ComputerModern"]})

    fig, ax = plt.subplots(figsize=(14,8))
    # fig.subplots_adjust(top=0.2)

    plt.rcParams.update({'font.size': 26})

    lambdas = np.logspace(lambda_min, lambda_max, nlambdas)
    n_nodes = [5, 10, 50, 100]

    MSE_nodes_lambda_SGD = np.empty((len(n_nodes), nlambdas))
    method = 'Regg'
    seeds = np.arange(n_seeds)

    for n_idx, nodes in tqdm(enumerate(n_nodes)):
        for l_idx, lmb in enumerate(lambdas):
            mse_SGD = 0
            for seed in seeds:
                x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(X, Y, Z, test_size=0.2, random_state = seed)

                input_train = np.c_[x_train, y_train]
                input_test = np.c_[x_test, y_test]

                n_nodes_inputLayer = input_train.shape[1]
                n_nodes_outputLayer = z_train.shape[1]

                ANN_SGD = [
                Layer(n_nodes_inputLayer, nodes),
                acti_func(),
                Layer(nodes, nodes),
                acti_func(),
                Layer(nodes, n_nodes_outputLayer),
                Linear_Activation()
                ]

                mse_SGD_train, mse_SGD_test, R2_SGD_train, R2_SGD_test = train_NN_SGD(ANN_SGD, input_train, z_train, input_test, z_test, n_epochs, init_LR, decay, O_M, momentum, minibatch_size, n_minibatches, seed, lmb, method)

                y_pred_SGD = fwd(ANN_SGD, input_test)[0]

                mse_SGD += mse_SGD_test[-1]

                MSE_nodes_lambda_SGD[n_idx,l_idx] = mse_SGD #/len(seeds)

    """ Heatmap Plot for SGD """
    sns.heatmap(MSE_nodes_lambda_SGD.T, cmap="RdYlGn_r",
    annot=True, annot_kws={"size": 20},
    fmt="1.4f", linewidths=1, linecolor=(30/255,30/255,30/255,1),
    cbar_kws={"orientation": "horizontal", "shrink":0.8, "aspect":40, "label":r"Cost", "pad":0.05})
    x_idx = np.arange(len(n_nodes)) + 0.5
    y_idx = np.arange(nlambdas) + 0.5
    ax.set_xticks(x_idx, [str(deg) for deg in n_nodes], fontsize='medium')
    ax.set_yticks(y_idx, [str(f'{lam:1.1E}') for lam in lambdas], rotation=0, fontsize='medium')
    ax.set_xlabel("Nodes in the layer", labelpad=10,  fontsize='medium')
    ax.set_ylabel(r'$\log_{10} \lambda$', labelpad=10,  fontsize='medium')
    ax.set_title(r'\bf{MSE Heatmap - SGD}', pad=15)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    plt.tight_layout()
    plt.show()

    """ Finding the aramaters that gave the lowest cost """
    index = np.argwhere(MSE_nodes_lambda_SGD == np.min(MSE_nodes_lambda_SGD))
    best_n_nodes_cost = n_nodes[index[0,0]]
    best_lambda_cost = lambdas[index[0,1]]
    print(f'The lowest cost with SGD was achieved {best_n_nodes_cost} nodes, and with lambda = {best_lambda_cost}.')

    return best_n_nodes_cost, best_lambda_cost

def FF_plot(ANN_GD):
    plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["ComputerModern"]})
    plt.rcParams['figure.figsize'] = (16,12)
    plt.rcParams.update({'font.size': 20})
    plt.rc('axes', facecolor='none', edgecolor='none',
        axisbelow=True, grid=True)
    plt.rc('grid', color='w', linestyle='solid')
    plt.rc('lines', linewidth=1)

    """ Make Surface Data """
    x = np.linspace(0, 1, 50).reshape(-1,1)
    y = np.linspace(0, 1, 50).reshape(-1,1)
    X, Y = np.meshgrid(x, y)
    X_ = X.flatten().reshape(-1, 1)
    Y_ = Y.flatten().reshape(-1, 1)
    i = np.c_[X_, Y_]
    z = fwd(ANN_GD, i)[0]
    z = z.reshape(50, 50)
    z_diff = abs(z - Z_no_noise)

    # fig.patch.set_facecolor('whitesmoke')
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    
    ax.xaxis.set_pane_color((230/255, 230/255, 230/255, 1))
    ax.yaxis.set_pane_color((230/255, 230/255, 230/255, 1))
    ax.zaxis.set_pane_color((230/255, 230/255, 230/255, 1))

    # ax.scatter(x, y, z, c=z, cmap="winter")
    # cset = ax.contourf(X_no_noise, Y_no_noise, Z_no_noise, zdir='x', offset=-0.1, cmap=cm.Blues)
    # cset = ax.contourf(X_no_noise, Y_no_noise, Z_no_noise, zdir='y', offset=1.1, cmap=cm.Blues)
    # cset = ax.contourf(x, y, z, zdir='x', offset=-0.1, alpha=0.3, cmap=cm.Oranges)
    # cset = ax.contourf(x, y, z, zdir='y', offset=1.1, alpha=0.3, cmap=cm.Oranges)
    # ax.plot_surface(X_no_noise, Y_no_noise, Z_no_noise, rstride=8, cstride=8, alpha=0.7, cmap="Blues")

    ax.plot_surface(X, Y, z, alpha=1, alpha=0.8, cmap="cool")

    cset = ax.contourf(X, Y, z_diff, zdir='z', offset=-0.79, cmap='winter')
    
    cbaxes = fig.add_axes([0.21, 0.3, 0.03, 0.4]) 
    cbar = fig.colorbar(cset, pad=0.1, shrink=0.5, cax = cbaxes, ticks=[np.min(z_diff), np.max(z_diff)/2, np.max(z_diff)])
    cbar.ax.set_ylabel(r'Absolute Distance')
    cbar.ax.yaxis.set_label_position('left')
    ax.set_title(r'\bf{Frank Function Surface Plot - NN with GD}', y=0.96)
    ax.set_xlabel(r'x values', labelpad=15)
    ax.set_ylabel(r'y values', labelpad=15)
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_zticks([0.0, 0.5, 1.0])
    ax.set_zlim(-0.8, 2)
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.set_zlabel(r'Frank Funcrion', labelpad=10)
    ax.tick_params(axis='both', which='major')
    ax.view_init(elev=13, azim=-24)
    plt.show()

def cost_plot_SGD(n_epochs, n_minibatches, cost_SGD_train, cost_SGD_test):
    iters = np.arange(n_epochs*n_minibatches)
    plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["ComputerModern"]})
    plt.rc('axes', facecolor='whitesmoke', edgecolor='none',
       axisbelow=True, grid=True)
    plt.rc('grid', color='w', linestyle='solid')
    plt.rc('lines', linewidth=2)
    plt.rcParams['figure.figsize'] = (8,6)
    plt.rcParams.update({'font.size': 20})
    plt.yscale('log')
    plt.plot(iters, cost_SGD_test, zorder=100, label=r'Cost for test data - SGD')
    plt.plot(iters, cost_SGD_train, zorder=0, label=r'Cost for train data - SGD')
    plt.title(r'Cost as Function of Iteration', pad=15)
    plt.xlabel(fr'Iterations (number of epochs = {n_epochs}, number of batches {n_minibatches})', labelpad=10)
    plt.ylabel(r'Cost (log-scale)', labelpad=10)
    plt.legend(framealpha=0.9, facecolor=(1, 1, 1, 1))
    plt.show()

def R2_plot_SGD(n_epochs, n_minibatches, R2_SGD_train, R2_SGD_test):
    iters = np.arange(n_epochs*n_minibatches)
    plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["ComputerModern"]})
    plt.rc('axes', facecolor='whitesmoke', edgecolor='none',
       axisbelow=True, grid=True)
    plt.rc('grid', color='w', linestyle='solid')
    plt.rc('lines', linewidth=2)
    plt.rcParams['figure.figsize'] = (8,6)
    plt.rcParams.update({'font.size': 20})
    plt.plot(iters, R2_SGD_test, zorder=100, label=r'R2 score for test data - SGD')
    plt.plot(iters, R2_SGD_train, zorder=0, label=r'R2 score for train data - SGD')
    plt.yscale('log')
    plt.title(r'R2 score as Function of Iteration', pad=15)
    plt.xlabel(fr'Iterations: (number of epochs = {n_epochs}, number of batches {n_minibatches})', labelpad=10)
    plt.ylabel(r'R2 score (log-scale)', labelpad=10)
    plt.legend(framealpha=0.9, facecolor=(1, 1, 1, 1))
    plt.show()



if __name__=='__main__':

    """ DATA """
    seed = 55 # np.random.randint(0, 100)

    X_no_noise, Y_no_noise, Z_no_noise = data_FF(noise=False, step_size=0.02, alpha=0.05, reshape=False)

    X, Y, Z = data_FF(noise=True, step_size=0.02, alpha=0.1, reshape=True)

    x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(X, Y, Z, test_size=0.2, random_state = seed)
    input_train = np.c_[x_train, y_train]
    input_test = np.c_[x_test, y_test]

    n_nodes_inputLayer = input_train.shape[1]
    n_nodes_outputLayer = z_train.shape[1]

    """ Hyperparameters """
    n_epochs = 10                                     # Number of epochs
    init_LR = 0.05                                      # Initial learning rate (LR)
    decay = 0.0                                         # init_LR/n_epochs # LR decay rate (for fixed LR set decay to 0)
    momentum = 0.0                                      # Momentum value for GD and SGD.
    minibatch_size = 10                 
    n_minibatches =  x_train.shape[0]//minibatch_size   # number of minibatches
    # N_iter_GD = n_epochs*n_minibatches                # Number of iterations for GD
    lambda_min, lambda_max = -16, -2                    # Lambda search space
    nlambdas = 7                                        # Number of lambdas
    n_seeds = 1                                         # Number of seeds to achieve an average cost

    """ Optimization Method """
    # If you want plain GD without any optimization choose 'momentum' with momentum value of 0
    Optimizer_method = ['Adam','momentum'] # Adagrad and RMSprop not yet implemented 
    O_M = Optimizer_method[0] #Choose the optimization method

    """ Model Type """
    problem_method = ['Regg', 'Class']
    method = problem_method[0]

    """ PLOTS: """
    # Possible activation functions are Linear_Activation, Sigmoid, ReLU, LeakyReLU, Hyperbolic, ELU and Sin
    activation_functions = [Sigmoid, Hyperbolic, ReLU, LeakyReLU, ELU]
    for acti_func in activation_functions[:1]:

        best_n_nodes_cost, best_lambda_cost = cost_heatmap(X, Y, Z, n_epochs, init_LR, decay, O_M, momentum, minibatch_size, n_minibatches, lambda_min, lambda_max, nlambdas, n_seeds, acti_func)

        """ The Neural Network """
        ANN_SGD = [
        Layer(n_nodes_inputLayer, best_n_nodes_cost),
        acti_func(),
        Layer(best_n_nodes_cost, best_n_nodes_cost),
        acti_func(),
        Layer(best_n_nodes_cost, n_nodes_outputLayer),
        Linear_Activation()
        ]

        """ Traning the Network """
        cost_SGD_train, cost_SGD_test, R2_SGD_train, R2_SGD_test = train_NN_SGD(ANN_SGD, input_train, z_train, input_test, z_test, n_epochs, init_LR, decay, O_M, momentum, minibatch_size, n_minibatches, seed, best_lambda_cost, method)

        """ Forward => Applying Weights and Biases  """
        y_pred_SGD = fwd(ANN_SGD, input_test)[0]

        """ Plotting the Surface Plot """
        FF_plot(ANN_SGD)

        """ Plotting the cost """
        cost_plot_SGD(n_epochs, n_minibatches, cost_SGD_train, cost_SGD_test)

        """ Plotting the R2 score """
        R2_plot_SGD(n_epochs, n_minibatches, R2_SGD_train, R2_SGD_test)

