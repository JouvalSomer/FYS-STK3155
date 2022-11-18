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
    n_nodes = [10, 50, 100, 150]

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
                Layer(nodes, nodes),
                acti_func(),
                Layer(nodes, n_nodes_outputLayer),
                Linear_Activation()
            ]

                mse_SGD_train, mse_SGD_test, R2_SGD_train, R2_SGD_test = train_NN_SGD(ANN_SGD, input_train, z_train, input_test, z_test, n_epochs, init_LR, decay, O_M, momentum, minibatch_size, n_minibatches, seed, lmb, method)


                mse_SGD += mse_SGD_test[-1]

            MSE_nodes_lambda_SGD[n_idx,l_idx] = mse_SGD/n_seeds

    """ Heatmap Plot for SGD """
    sns.heatmap(MSE_nodes_lambda_SGD.T, cmap="RdYlGn_r",
    annot=True, annot_kws={"size": 20},
    fmt="1.4f", linewidths=1, linecolor=(30/255,30/255,30/255,1),
    cbar_kws={"orientation": "horizontal", "shrink":0.8, "aspect":40, "label":r"Cost", "pad":0.05})
    x_idx = np.arange(len(n_nodes)) + 0.5
    y_idx = np.arange(nlambdas) + 0.5
    ax.set_xticks(x_idx, [str(deg) for deg in n_nodes], fontsize='medium')
    ax.set_yticks(y_idx, [str(f'{lam:1.1E}') for lam in lambdas], rotation=0, fontsize='medium')
    ax.set_xlabel(r"Nodes in the layer", labelpad=10,  fontsize='medium')
    ax.set_ylabel(r'$\log_{10} \lambda$', labelpad=10,  fontsize='medium')
    ax.set_title(fr'Cost Heatmap for NN on Franke Data - SGD, Adam and {acti_func.__name__}', pad=15)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    plt.tight_layout()
    plt.savefig(f'cost_heatmap_{acti_func.__name__}_1.png', dpi=150)
    plt.clf()

    """ Finding the paramaters that give the lowest cost """
    index = np.argwhere(MSE_nodes_lambda_SGD == np.min(MSE_nodes_lambda_SGD))
    best_n_nodes_cost = n_nodes[index[0,0]]
    best_lambda_cost = lambdas[index[0,1]]
    print(f'\nThe lowest cost with SGD was achieved {best_n_nodes_cost} nodes, and with lambda = {best_lambda_cost}.\n')

    return best_n_nodes_cost, best_lambda_cost

def cost_plot_SGD(n_epochs, n_minibatches, cost_SGD_train, cost_SGD_test, acti_func):
    iters = np.arange(n_epochs*n_minibatches)

    plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["ComputerModern"]})
    plt.rcParams['figure.figsize'] = (8,6)
    # plt.rcParams.update({'font.size': 20})
    plt.rcParams.update({'font.size': 26})
    plt.rc('axes', facecolor='whitesmoke', edgecolor='none',
    axisbelow=True, grid=True)
    plt.rc('grid', color='w', linestyle='solid')
    plt.rc('lines', linewidth=2)
    print(f'The lowest cost was {np.min(cost_SGD_test)}.\n')

    plt.yscale('log')
    plt.plot(iters, cost_SGD_test, zorder=100, label=r'Test Cost')
    plt.plot(iters, cost_SGD_train, zorder=5, label=r'Train Cost')
    plt.title(fr'Cost as Function of Iteration - SGD, Adam and {acti_func.__name__}', pad=15)
    plt.xlabel(fr'Iterations (number of epochs = {n_epochs}, number of batches {n_minibatches})', labelpad=10)
    plt.ylabel(r'Cost (log-scale)', labelpad=10)
    plt.legend(framealpha=0.9, facecolor=(1, 1, 1, 1))
    plt.tight_layout()
    plt.savefig(f'cost_plot_{acti_func.__name__}_1.png', dpi=150)
    plt.clf()

def R2_plot_SGD(n_epochs, n_minibatches, R2_SGD_train, R2_SGD_test, acti_func):
    iters = np.arange(n_epochs*n_minibatches)

    plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["ComputerModern"]})
    plt.rcParams['figure.figsize'] = (8,6)
    # plt.rcParams.update({'font.size': 20})
    plt.rcParams.update({'font.size': 26})
    plt.rc('axes', facecolor='whitesmoke', edgecolor='none',
    axisbelow=True, grid=True)
    plt.rc('grid', color='w', linestyle='solid')
    plt.rc('lines', linewidth=2)
    print(f'The highest R2 score was {np.max(R2_SGD_test)}.\n')
    plt.plot(iters, R2_SGD_test, zorder=100, label=r'Test R2 score')
    plt.plot(iters, R2_SGD_train, zorder=5, label=r'Train R2 score')
    plt.yscale('log')
    plt.title(fr'R2 score as Function of Iteration - SGD, Adam and {acti_func.__name__}', pad=15)
    plt.xlabel(fr'Iterations: (number of epochs = {n_epochs}, number of batches {n_minibatches})', labelpad=10)
    plt.ylabel(r'R2 score (log-scale)', labelpad=10)
    plt.legend(framealpha=0.9, facecolor=(1, 1, 1, 1))
    plt.tight_layout()
    plt.savefig(f'R2_plot_{acti_func.__name__}_1.png', dpi=150)
    plt.clf()

def FF_plot(X, Y, z, z_diff, acti_func):
    plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["ComputerModern"]})
    plt.rcParams['figure.figsize'] = (12,6)
    plt.rcParams.update({'font.size': 20})
    plt.rc('axes', facecolor='none', edgecolor='none',
        axisbelow=True, grid=True)
    plt.rc('grid', color='w', linestyle='solid')
    plt.rc('lines', linewidth=1)

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

    ax.plot_surface(X, Y, z, alpha=1, cmap="cool")

    cset = ax.contourf(X, Y, z_diff, zdir='z', offset=-0.79, cmap='winter')
    
    cbaxes = fig.add_axes([0.17, 0.3, 0.03, 0.4]) 
    cbar = fig.colorbar(cset, pad=0.1, shrink=0.5, cax = cbaxes, ticks=[np.min(z_diff), np.max(z_diff)/2, np.max(z_diff)])
    cbar.ax.set_ylabel(r'Absolute Distance')
    cbar.ax.yaxis.set_label_position('left')
    ax.set_title(fr'Franke Function Surface Plot - NN w/ SGD, Adam and {acti_func.__name__}', y=0.96)
    ax.set_xlabel(r'x values', labelpad=15)
    ax.set_ylabel(r'y values', labelpad=15)
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_zticks([0.0, 0.5, 1.0])
    ax.set_zlim(-0.8, 2)
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.set_zlabel(r'Franke Funcrion', labelpad=10)
    ax.tick_params(axis='both', which='major')
    ax.view_init(elev=15, azim=-30)
    plt.savefig(f'surface_plot_{acti_func.__name__}_1.png', dpi=150)
    # plt.show()
    plt.clf()

""" Func to find best minibatch size """
def find_minibatch_size_SDG(input_train, z_train, input_test, z_test, n_nodes_inputLayer, n_nodes_outputLayer, best_n_nodes_cost, best_lambda_cost, n_epochs, init_LR, decay, O_M, momentum, seed, method):

    """ Model Type """
    problem_method = ['Regg', 'Class']
    method = problem_method[0]

    minibatchs = [5, 10, 20, 30, 40, 50]
    cost_minibatch = np.zeros(len(minibatchs))

    """ Optimization method """
    # To get plain GD without any optimization choose 'momentum' with momentum value of 0
    Optimizer_method = ['Adagrad', 'RMSprop', 'Adam','momentum']
    O_M = Optimizer_method[2] # Choose the optimization method


    for mb_idx, size in enumerate(minibatchs):
        minibatch_size = np.shape(x_train)[0]//size
        n_minibatches = np.shape(x_train)[0]//minibatch_size #number of minibatches

        ANN_SGD = [
        Layer(n_nodes_inputLayer, best_n_nodes_cost),
        acti_func(),
        Layer(best_n_nodes_cost, best_n_nodes_cost),
        acti_func(),
        Layer(best_n_nodes_cost, best_n_nodes_cost),
        acti_func(),
        Layer(best_n_nodes_cost, n_nodes_outputLayer),
        Linear_Activation()
        ]

        mse_SGD_train, mse_SGD_test, R2_SGD_train, R2_SGD_test = train_NN_SGD(ANN_SGD, input_train, z_train, input_test, z_test, n_epochs, init_LR, decay, O_M, momentum, minibatch_size, n_minibatches, seed, best_lambda_cost, method)


        cost_minibatch[mb_idx] = mse_SGD_test[-1]

    index = np.argwhere(cost_minibatch == np.min(cost_minibatch))
    best_minibatch_size = minibatchs[index[0,0]]
    
    minibatch_size = np.shape(x_train)[0]//best_minibatch_size
    n_minibatches = np.shape(x_train)[0]//minibatch_size #number of minibatches

    print(f'The lowest cost with SGD was achieved with minibatch size = {minibatch_size}. Thus, {n_minibatches} minibatches.')
    return best_minibatch_size

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
    print(np.shape(x_train)[0])
    
    """ Hyperparameters """
    n_epochs = 10                                     # Number of epochs
    init_LR = 0.01                                      # Initial learning rate (LR)
    decay = 0.0                                         # init_LR/n_epochs # LR decay rate (for fixed LR set decay to 0)
    momentum = 0.0                                      # Momentum value for GD and SGD.
    minibatch_size = int(np.floor(x_train.shape[0] * 0.005)) # 0.5% of the train data-set
    n_minibatches =  x_train.shape[0]//minibatch_size   # number of minibatches
    lambda_min, lambda_max = -15, 1                    # Lambda search space
    nlambdas = 9                                        # Number of lambdas
    n_seeds = 3                                         # Number of seeds to achieve an average cost
    B = 100

    iters = n_minibatches*n_epochs
    cost_SGD_train_mean = np.empty((iters, B))
    cost_SGD_test_mean = np.empty((iters, B))
    R2_SGD_train_mean = np.empty((iters, B))
    R2_SGD_test_mean = np.empty((iters, B))


    print('\n')
    print('n_epochs = ', n_epochs)
    print('init_LR = ', init_LR)
    print('minibatch_size = ', minibatch_size)
    print('n_minibatches = ', n_minibatches)
    print('\n')
    
    """ Optimization Method """
    # If you want plain GD without any optimization choose 'momentum' with momentum value of 0
    Optimizer_method = ['Adam','momentum'] # Adagrad and RMSprop not yet implemented 
    O_M = Optimizer_method[0] #Choose the optimization method

    """ Model Type """
    problem_method = ['Regg', 'Class']
    method = problem_method[0]


    """ PLOTS: """
    # Possible activation functions are Linear_Activation, Sigmoid, ReLU, LeakyReLU, Hyperbolic, ELU and Sin
    acti_func = LeakyReLU
    best_n_nodes_cost, best_lambda_cost = cost_heatmap(X, Y, Z, n_epochs, init_LR, decay, O_M, momentum, minibatch_size, n_minibatches, lambda_min, lambda_max, nlambdas, n_seeds, acti_func)


    """ Make Surface Data """
    x = np.linspace(0, 1, 50).reshape(-1,1)
    y = np.linspace(0, 1, 50).reshape(-1,1)
    X, Y = np.meshgrid(x, y)
    X_ = X.flatten().reshape(-1, 1)
    Y_ = Y.flatten().reshape(-1, 1)
    i = np.c_[X_, Y_]
    z = np.zeros(50*50).reshape(-1, 1)

    """ The Neural Network """
    ANN_SGD = [
        Layer(n_nodes_inputLayer, best_n_nodes_cost),
        acti_func(),
        Layer(best_n_nodes_cost, best_n_nodes_cost),
        acti_func(),
        Layer(best_n_nodes_cost, best_n_nodes_cost),
        acti_func(),
        Layer(best_n_nodes_cost, n_nodes_outputLayer),
        Linear_Activation()
    ]

    # best_minibatch_size = find_minibatch_size_SDG(input_train, z_train, input_test, z_test, n_nodes_inputLayer, n_nodes_outputLayer, best_n_nodes_cost, best_lambda_cost, n_epochs, init_LR, decay, O_M, momentum, seed, method)
    for b in range(B):
        """ Traning the Network """
        cost_SGD_train_, cost_SGD_test_, R2_SGD_train_, R2_SGD_test_ = train_NN_SGD(ANN_SGD, input_train, z_train, input_test, z_test, n_epochs, init_LR, decay, O_M, momentum, minibatch_size, n_minibatches, seed, best_lambda_cost, method)
        # cost_SGD_train, cost_SGD_test, R2_SGD_train, R2_SGD_test = train_NN_GD(ANN_SGD, x_train, y_train, x_test, y_test, n_epochs, init_LR, decay, O_M, momentum, best_lambda_cost, method)
        
        cost_SGD_train_mean[:,b] += cost_SGD_train_
        cost_SGD_test_mean[:,b] += cost_SGD_test_
        R2_SGD_train_mean[:,b] += R2_SGD_train_
        R2_SGD_test_mean[:,b] += R2_SGD_test_
        
        output = fwd(ANN_SGD, i)[0]
        z += output

    z /= B
    z = z.reshape(50, 50)
    z_diff = abs(z - Z_no_noise)

    cost_SGD_train = np.mean(cost_SGD_train_mean, axis=1)
    cost_SGD_test = np.mean(cost_SGD_test_mean, axis=1) 
    R2_SGD_train = np.mean(R2_SGD_train_mean, axis=1)
    R2_SGD_test = np.mean(R2_SGD_test_mean, axis=1)

    """ Forward => Applying Weights and Biases  """
    y_pred_SGD = fwd(ANN_SGD, input_test)[0]

    """ Plotting the cost """
    cost_plot_SGD(n_epochs, n_minibatches, cost_SGD_train, cost_SGD_test, acti_func)

    """ Plotting the R2 score """
    R2_plot_SGD(n_epochs, n_minibatches, R2_SGD_train, R2_SGD_test, acti_func)

    """ Plotting the Surface Plot """
    FF_plot(X, Y, z, z_diff, acti_func)

