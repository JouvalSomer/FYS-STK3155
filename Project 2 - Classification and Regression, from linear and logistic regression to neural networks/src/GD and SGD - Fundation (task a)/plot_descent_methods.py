import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
# from tqdm import tqdm

from func import *
from optimizers import *
from descent_methods import *


""" PLOTS: """
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["ComputerModern"]})

""" (1/8) Heatmap of GD to find best lambda and model complexity """
def find_lambda_DG(x_train, x_test, y_train, y_test, x, y_true, G_M, lambda_min, lambda_max, nlambdas, max_polydeg, plot, n_epochs, seed):

    lambdas = np.logspace(lambda_min, lambda_max, nlambdas)
    polydeg = np.arange(max_polydeg)
    cost_lambda_degree = np.empty((nlambdas, max_polydeg))

    """ Hyperparameters """
    init_LR = 0.1                          # Initial learning rate (LR)
    decay = 0.0                             # init_LR/n_epochs # LR decay rate (for fixed LR set decay to 0)
    momentum = 0.0                          # Momentum value for GD.

    """ Optimization method """
    # To get plain GD without any optimization choose 'momentum' with momentum value of 0
    Optimizer_method = ['Adagrad', 'RMSprop', 'Adam','momentum']
    O_M = Optimizer_method[3] # Choose the optimization method

    for d_idx, deg in enumerate(polydeg):

        X_train = designMatrix_1D(x_train, deg + 1)
        X_test = designMatrix_1D(x_test, deg + 1)

        for l_idx, lmb in enumerate(lambdas):
            y_predict_GD, theta, test_cost_GD, train_cost_GD = GD(X_train, X_test, y_train, y_test, G_M, O_M, n_epochs, init_LR, decay, momentum, seed, lmb)
            cost_lambda_degree[l_idx, d_idx] = test_cost_GD[-1]
    
    index = np.argwhere(cost_lambda_degree == np.min(cost_lambda_degree))
    best_poly_deg_cost = polydeg[index[0,1]]
    best_lambda_cost = lambdas[index[0,0]]

    print(f'The lowest cost with GD was achieved at polynomial degree = {best_poly_deg_cost}, and with lambda = {best_lambda_cost}.')

    if plot:
        fig, ax = plt.subplots(figsize=(14,8))
        plt.rcParams.update({'font.size': 26})
        sns.heatmap(cost_lambda_degree[:,1:], cmap="RdYlGn_r", 
        annot=True, annot_kws={"size": 20},
        fmt="1.4f", linewidths=1, linecolor=(30/255,30/255,30/255,1),
        cbar_kws={"orientation": "horizontal", "shrink":0.8, "aspect":40, "label":r"Cost", "pad":0.05})
        x_idx = np.arange(max_polydeg-1) + 0.5
        y_idx = np.arange(nlambdas) + 0.5
        ax.set_xticks(x_idx, [deg for deg in polydeg[1:]], fontsize='medium')
        ax.set_yticks(y_idx, [float(f'{lam:1.1E}') for lam in lambdas], rotation=0, fontsize='medium')
        ax.set_xlabel(r"Polynomial degree", labelpad=10, fontsize='medium')
        ax.set_ylabel(r'$\log_{10} \lambda$', labelpad=10, fontsize='medium')
        ax.set_title(r'\bf{Cost Heatmap for plain GD}', pad=15)
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')
        plt.tight_layout()
        plt.savefig('cost_heatmap_plain_GD_LR_0_1.png', dpi=150)
        plt.clf()

        plt.rc('axes', facecolor='whitesmoke', edgecolor='none',
        axisbelow=True, grid=True)
        plt.rc('grid', color='w', linestyle='solid')
        plt.rc('lines', linewidth=2)

        X_train = designMatrix_1D(x_train, best_poly_deg_cost + 1)
        X_test = designMatrix_1D(x_test, best_poly_deg_cost + 1)

        y_predict_GD, theta, test_cost_GD, train_cost_GD = GD(X_train, X_test, y_train, y_test, G_M, O_M, n_epochs, init_LR, decay, momentum, seed, lmb)

        fig, ax = plt.subplots(figsize=(16,9))
        fig.subplots_adjust(bottom=0.22)
        """ Regression line plot """
        ax.scatter(x_test, y_predict_GD, c='limegreen', s=5, label=r'GD')
        # ax.scatter(x_test, y_predict_OLS, c='dodgerblue', s=5, label='OLS')
        ax.plot(x, y_true, zorder=100, c='black', label='True y')
        ax.scatter(x_train, y_train, c='indigo', marker='o', s=3, alpha=0.3, label='Data') # Data
        ax.set_title(r'\bf{Regression line plot for plain GD}', pad=15)
        ax.set_xlabel(r'$x$', labelpad=10)
        ax.set_ylabel(r'$y$',  labelpad=10)
        ax.legend(framealpha=0.9, facecolor=(1, 1, 1, 1))
        string = f'init-LR = {init_LR}, decay = {decay}, momentum = {momentum}, n\_epochs = {n_epochs}'
        plt.figtext(0.5, 0.05, string, ha="center", fontsize=18, bbox={'facecolor':'white', 'edgecolor':'black', 'lw':0.5, 'boxstyle':'round'})
        plt.savefig('regression_line_plot_plain_GD.png', dpi=150)
        plt.clf()

        iters = np.arange(n_epochs)
        fig, ax = plt.subplots(figsize=(16,9))
        fig.subplots_adjust(bottom=0.22)
        ax.plot(iters, train_cost_GD, color='crimson', zorder=100, lw=2, label=r"Train Cost for GD") #zorder=0,
        ax.plot(iters, test_cost_GD, color='royalblue', lw=2, label=r"Test Cost for GD") #zorder=0,
        # plt.scatter(best_poly_deg_GD, np.min(MSE_test), color='forestgreen', marker='x', zorder=100, s=150, label='Lowest MSE')
        ax.set_xlabel(r"Iterations", labelpad=10)
        ax.set_ylabel(r"Cost", labelpad=10)
        ax.set_title(r"\bf{Cost as function of iterations for plain GD}", pad=15)
        ax.legend(framealpha=0.9, facecolor=(1, 1, 1, 1))
        ax.set_yscale('log')
        string = f'init-LR = {init_LR}, decay = {decay}, momentum = {momentum}, n\_epochs = {n_epochs}'
        plt.figtext(0.5, 0.05, string, ha="center", fontsize=18, bbox={'facecolor':'white', 'edgecolor':'black', 'lw':0.5, 'boxstyle':'round'})
        plt.savefig('cost_plot_plain_GD.png', dpi=150)
        plt.clf()

    return best_poly_deg_cost, best_lambda_cost

""" (2/8) Heatmap of SGD to find best lambda and model complexity """
def find_lambda_SDG(x_train, x_test, y_train, y_test, x, y_true, G_M, lambda_min, lambda_max, nlambdas, max_polydeg, plot, n_epochs, seed):

    lambdas = np.logspace(lambda_min, lambda_max, nlambdas)
    polydeg = np.arange(max_polydeg)
    cost_lambda_degree = np.empty((nlambdas, max_polydeg))

    """ Hyperparameters """
    init_LR = 0.1                          # Initial learning rate (LR)
    decay = 0.0                             # init_LR/n_epochs # LR decay rate (for fixed LR set decay to 0)
    momentum = 0.0                          # Momentum value for GD.
    minibatch_size = np.shape(x_train)[0]//20
    n_minibatches = np.shape(x_train)[0]//minibatch_size #number of minibatches

    """ Optimization method """
    # To get plain GD without any optimization choose 'momentum' with momentum value of 0
    Optimizer_method = ['Adagrad', 'RMSprop', 'Adam','momentum']
    O_M = Optimizer_method[3] # Choose the optimization method

    for d_idx, deg in enumerate(polydeg):

        X_train = designMatrix_1D(x_train, deg + 1)
        X_test = designMatrix_1D(x_test, deg + 1)

        for l_idx, lmb in enumerate(lambdas):
            y_predict_SGD, theta, test_cost_SGD, train_cost_SGD = SGD(X_train, X_test, y_train, y_test, O_M, G_M, minibatch_size, n_minibatches, n_epochs, init_LR, decay, momentum, seed, lmb)
            cost_lambda_degree[l_idx, d_idx] = test_cost_SGD[-1]
    
    index = np.argwhere(cost_lambda_degree == np.min(cost_lambda_degree))
    best_poly_deg_cost = polydeg[index[0,1]]
    best_lambda_cost = lambdas[index[0,0]]

    print(f'The lowest cost with SGD was achieved at polynomial degree = {best_poly_deg_cost}, and with lambda = {best_lambda_cost}.')

    if plot:
        fig, ax = plt.subplots(figsize=(14,8))
        plt.rcParams.update({'font.size': 26})
        sns.heatmap(cost_lambda_degree[:,1:], cmap="RdYlGn_r", 
        annot=True, annot_kws={"size": 20},
        fmt="1.4f", linewidths=1, linecolor=(30/255,30/255,30/255,1),
        cbar_kws={"orientation": "horizontal", "shrink":0.8, "aspect":40, "label":r"Cost", "pad":0.05})
        x_idx = np.arange(max_polydeg-1) + 0.5
        y_idx = np.arange(nlambdas) + 0.5
        ax.set_xticks(x_idx, [deg for deg in polydeg[1:]], fontsize='medium')
        ax.set_yticks(y_idx, [float(f'{lam:1.1E}') for lam in lambdas], rotation=0, fontsize='medium')
        ax.set_xlabel(r"Polynomial degree", labelpad=10, fontsize='medium')
        ax.set_ylabel(r'$\log_{10} \lambda$', labelpad=10, fontsize='medium')
        ax.set_title(r'\bf{Cost Heatmap for plain SGD}', pad=15)
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')
        plt.tight_layout()
        plt.savefig('cost_heatmap_plain_SGD_LR_0_1.png', dpi=150)
        plt.clf()

        plt.rc('axes', facecolor='whitesmoke', edgecolor='none',
        axisbelow=True, grid=True)
        plt.rc('grid', color='w', linestyle='solid')
        plt.rc('lines', linewidth=2)

        X_train = designMatrix_1D(x_train, best_poly_deg_cost + 1)
        X_test = designMatrix_1D(x_test, best_poly_deg_cost + 1)

        y_predict_SGD, theta, test_cost_SGD, train_cost_SGD = SGD(X_train, X_test, y_train, y_test, O_M, G_M, minibatch_size, n_minibatches, n_epochs, init_LR, decay, momentum, seed, best_lambda_cost)
        
        """ Regression line plot """
        fig, ax = plt.subplots(figsize=(16,9))
        fig.subplots_adjust(bottom=0.22)
        ax.scatter(x_test, y_predict_SGD, c='limegreen', s=5, label=r'SGD')
        # ax.scatter(x_test, y_predict_OLS, c='dodgerblue', s=5, label='OLS')
        ax.plot(x, y_true, zorder=100, c='black', label='True y')
        ax.scatter(x_train, y_train, c='indigo', marker='o', s=3, alpha=0.3, label='Data') # Data
        ax.set_title(r'\bf{Regression line plot for plain SGD}', pad=15)
        ax.set_xlabel(r'$x$', labelpad=10)
        ax.set_ylabel(r'$y$',  labelpad=10)
        ax.legend(framealpha=0.9, facecolor=(1, 1, 1, 1))
        string = f'init-LR = {init_LR}, decay = {decay}, momentum = {momentum}, n\_epochs = {n_epochs}, minibatch\_size = {minibatch_size}, n\_minibatches = {n_minibatches}'
        plt.figtext(0.5, 0.05, string, ha="center", fontsize=18, bbox={'facecolor':'white', 'edgecolor':'black', 'lw':0.5, 'boxstyle':'round'})
        plt.savefig('regression_line_plot_plain_SGD.png', dpi=150)
        plt.clf()

        iters = np.arange(n_epochs*n_minibatches)
        fig, ax = plt.subplots(figsize=(16,9))
        fig.subplots_adjust(bottom=0.22)
        ax.plot(iters, train_cost_SGD, color='crimson', zorder=100, lw=2, label=r"Train Cost for SGD") #zorder=0,
        ax.plot(iters, test_cost_SGD, color='royalblue', lw=2, label=r"Test Cost for SGD") #zorder=0,
        # plt.scatter(best_poly_deg_GD, np.min(MSE_test), color='forestgreen', marker='x', zorder=100, s=150, label='Lowest MSE')
        ax.set_xlabel(r"Iterations", labelpad=10)
        ax.set_ylabel(r"Cost", labelpad=10)
        ax.set_title(r"\bf{Cost as function of iterations for plain SGD}", pad=15)
        ax.set_yscale('log')
        ax.legend(framealpha=0.9, facecolor=(1, 1, 1, 1))
        string = f'init-LR = {init_LR}, decay = {decay}, momentum = {momentum}, n\_epochs = {n_epochs}, minibatch\_size = {minibatch_size}, n\_minibatches = {n_minibatches}'
        plt.figtext(0.5, 0.05, string, ha="center", fontsize=18, bbox={'facecolor':'white', 'edgecolor':'black', 'lw':0.5, 'boxstyle':'round'})
        plt.savefig('cost_plot_plain_SGD.png', dpi=150)
        plt.clf()

    return best_poly_deg_cost, best_lambda_cost

""" Func to find best minibatch size """
def find_minibatch_size_SDG(x_train, x_test, y_train, y_test, best_poly_deg_SGD, best_lambda_SGD, G_M, n_epochs, seed):

    minibatchs = [5, 10, 20, 30, 40, 50]

    X_train = designMatrix_1D(x_train, best_poly_deg_SGD)
    X_test = designMatrix_1D(x_test, best_poly_deg_SGD)

    """ Hyperparameters """
    init_LR = 0.1                          # Initial learning rate (LR)
    decay = 0.0                             # init_LR/n_epochs # LR decay rate (for fixed LR set decay to 0)
    momentum = 0.0                          # Momentum value for GD.

    cost_minibatch = np.zeros(len(minibatchs))

    """ Optimization method """
    # To get plain GD without any optimization choose 'momentum' with momentum value of 0
    Optimizer_method = ['Adagrad', 'RMSprop', 'Adam','momentum']
    O_M = Optimizer_method[3] # Choose the optimization method

    for mb_idx, size in enumerate(minibatchs):
        minibatch_size = np.shape(x_train)[0]//size
        n_minibatches = np.shape(x_train)[0]//minibatch_size #number of minibatches

        y_predict_SGD, theta, test_cost_SGD, train_cost_SGD = SGD(X_train, X_test, y_train, y_test, O_M, G_M, minibatch_size, n_minibatches, n_epochs, init_LR, decay, momentum, seed, best_lambda_SGD)
        cost_minibatch[mb_idx] = test_cost_SGD[-1]

    index = np.argwhere(cost_minibatch == np.min(cost_minibatch))
    best_minibatch_size = minibatchs[index[0,0]]

    minibatch_size = np.shape(x_train)[0]//best_minibatch_size
    n_minibatches = np.shape(x_train)[0]//minibatch_size #number of minibatches

    print(f'The lowest cost with SGD was achieved with minibatch size = {minibatch_size}. Thus, {n_minibatches} minibatches.')
    return best_minibatch_size

""" (3/8) cost-plot for GD and SGD with a fixed learning rate using the chosen lambda """
def fixed_LR(x_train, x_test, y_train, y_test, x, y_true, best_poly_deg_GD, best_lambda_GD, best_poly_deg_SGD, best_lambda_SGD, G_M, n_epochs, seed, best_minibatch_size):
    """ Hyperparameters """
    init_LR = 0.1                          # Initial learning rate (LR)
    decay = 0.0                             # init_LR/n_epochs # LR decay rate (for fixed LR set decay to 0)
    momentum = 0.0                          # Momentum value for GD.
    minibatch_size = np.shape(x_train)[0]//best_minibatch_size
    n_minibatches = np.shape(x_train)[0]//minibatch_size #number of minibatches

    """ Optimization method """
    # To get plain GD without any optimization choose 'momentum' with momentum value of 0
    Optimizer_method = ['Adagrad', 'RMSprop', 'Adam','momentum']
    O_M = Optimizer_method[3] # Choose the optimization method

    X_train_GD = designMatrix_1D(x_train, best_poly_deg_GD) # Train design matrix for GD 
    X_test_GD = designMatrix_1D(x_test, best_poly_deg_GD) # Test design matrix for GD

    X_train_SGD = designMatrix_1D(x_train, best_poly_deg_SGD) # Train design matrix for SGD
    X_test_SGD = designMatrix_1D(x_test, best_poly_deg_SGD) # Test design matrix for SGD

    y_predict_GD, theta, test_cost_GD, train_cost_GD = GD(X_train_GD, X_test_GD, y_train, y_test, G_M, O_M, n_epochs*n_minibatches, init_LR, decay, momentum, seed, best_lambda_GD)
    y_predict_SGD, theta, test_cost_SGD, train_cost_SGD = SGD(X_train_SGD, X_test_SGD, y_train, y_test, O_M, G_M, minibatch_size, n_minibatches, n_epochs, init_LR, decay, momentum, seed, best_lambda_SGD)

    plt.rc('axes', facecolor='whitesmoke', edgecolor='none',
    axisbelow=True, grid=True)
    plt.rc('grid', color='w', linestyle='solid')
    plt.rc('lines', linewidth=2)

    iters = np.arange(n_epochs*n_minibatches)
    fig, ax = plt.subplots(figsize=(16,9))
    fig.subplots_adjust(bottom=0.22)
    ax.plot(iters, test_cost_GD, color='crimson', lw=2, zorder=100, label=r"Cost for the test data - GD") #zorder=0,
    ax.plot(iters, test_cost_SGD, color='royalblue', lw=2, label=r"Cost for the test data - SGD") #zorder=0,
    # plt.scatter(best_poly_deg_GD, np.min(MSE_test), color='forestgreen', marker='x', zorder=100, s=150, label='Lowest MSE')
    ax.set_xlabel(r"Iterations", labelpad=10)
    ax.set_ylabel(r"Cost", labelpad=10)
    ax.set_title(r"\bf{Cost as function of iterations for fixed LR}", pad=15)
    ax.set_yscale('log')
    ax.legend(framealpha=0.9, facecolor=(1, 1, 1, 1))
    string = f'init-LR = {init_LR}, decay = {decay}, momentum = {momentum}, n\_epochs = {n_epochs}, minibatch\_size = {minibatch_size}, n\_minibatches = {n_minibatches}'
    plt.figtext(0.5, 0.05, string, ha="center", fontsize=18, bbox={'facecolor':'white', 'edgecolor':'black', 'lw':0.5, 'boxstyle':'round'})
    plt.savefig('cost_plot_fixed_LR.png', dpi=150)
    plt.clf()

    """ Regression line plot """
    fig, ax = plt.subplots(figsize=(16,9))
    fig.subplots_adjust(bottom=0.22)
    ax.scatter(x_test, y_predict_SGD, c='limegreen', s=5, label='SGD')
    ax.scatter(x_test, y_predict_GD, c='crimson', s=5,label='GD')
    # ax.scatter(x_test, y_predict_OLS, c='dodgerblue', s=5, label='OLS')
    ax.plot(x, y_true, zorder=100, c='black', label='True y')
    ax.scatter(x_train, y_train, c='indigo', marker='o', s=3, alpha=0.3, label='Data') # Data
    ax.set_title(r'\bf{Regression line plot for fixed LR}', pad=15)
    ax.set_xlabel(r'$x$', labelpad=10)
    ax.set_ylabel(r'$y$',  labelpad=10)
    ax.legend(framealpha=0.9, facecolor=(1, 1, 1, 1))
    string = f'init-LR = {init_LR}, decay = {decay}, momentum = {momentum}, n\_epochs = {n_epochs}, minibatch\_size = {minibatch_size}, n\_minibatches = {n_minibatches}'
    plt.figtext(0.5, 0.05, string, ha="center", fontsize=18, bbox={'facecolor':'white', 'edgecolor':'black', 'lw':0.5, 'boxstyle':'round'})
    plt.savefig('regression_line_plot_fixed_LR.png', dpi=150)
    plt.clf()

""" (4/8) cost-plot for GD and SGD with a fixed learning rate and momentum using the chosen lambda """
def fixed_LR_momentum(x_train, x_test, y_train, y_test, x, y_true, best_poly_deg_GD, best_lambda_GD, best_poly_deg_SGD, best_lambda_SGD, G_M, n_epochs, seed, best_minibatch_size):
    """ Hyperparameters """
    init_LR = 0.1                          # Initial learning rate (LR)
    decay = 0.0                             # init_LR/n_epochs # LR decay rate (for fixed LR set decay to 0)
    momentum = 0.9                          # Momentum value for GD.
    minibatch_size = np.shape(x_train)[0]//best_minibatch_size
    n_minibatches = np.shape(x_train)[0]//minibatch_size #number of minibatches

    """ Optimization method """
    # To get plain GD without any optimization choose 'momentum' with momentum value of 0
    Optimizer_method = ['Adagrad', 'RMSprop', 'Adam','momentum']
    O_M = Optimizer_method[3] # Choose the optimization method

    X_train_GD = designMatrix_1D(x_train, best_poly_deg_GD) # Train design matrix for GD 
    X_test_GD = designMatrix_1D(x_test, best_poly_deg_GD) # Test design matrix for GD

    X_train_SGD = designMatrix_1D(x_train, best_poly_deg_SGD) # Train design matrix for SGD
    X_test_SGD = designMatrix_1D(x_test, best_poly_deg_SGD) # Test design matrix for SGD

    y_predict_GD, theta, test_cost_GD, train_cost_GD = GD(X_train_GD, X_test_GD, y_train, y_test, G_M, O_M, n_epochs*n_minibatches, init_LR, decay, momentum, seed, best_lambda_GD)
    y_predict_SGD, theta, test_cost_SGD, train_cost_SGD = SGD(X_train_SGD, X_test_SGD, y_train, y_test, O_M, G_M, minibatch_size, n_minibatches, n_epochs, init_LR, decay, momentum, seed, best_lambda_SGD)

    plt.rc('axes', facecolor='whitesmoke', edgecolor='none',
    axisbelow=True, grid=True)
    plt.rc('grid', color='w', linestyle='solid')
    plt.rc('lines', linewidth=2)

    iters = np.arange(n_epochs*n_minibatches)
    fig, ax = plt.subplots(figsize=(16,9))
    fig.subplots_adjust(bottom=0.22)
    ax.plot(iters, test_cost_GD, color='crimson', lw=2, zorder=100, label=r"Cost for the test data - GD") #zorder=0,
    ax.plot(iters, test_cost_SGD, color='royalblue', lw=2, label=r"Cost for the test data - SGD") #zorder=0,
    # plt.scatter(best_poly_deg_GD, np.min(MSE_test), color='forestgreen', marker='x', zorder=100, s=150, label='Lowest MSE')
    ax.set_xlabel(r"Iterations", labelpad=10)
    ax.set_ylabel(r"Cost", labelpad=10)
    ax.set_title(r"\bf{Cost as function of iterations for fixed LR and momentum}", pad=15)
    ax.set_yscale('log')
    ax.legend(framealpha=0.9, facecolor=(1, 1, 1, 1))
    string = f'init-LR = {init_LR}, decay = {decay}, momentum = {momentum}, n\_epochs = {n_epochs}, minibatch\_size = {minibatch_size}, n\_minibatches = {n_minibatches}'
    plt.figtext(0.5, 0.05, string, ha="center", fontsize=18, bbox={'facecolor':'white', 'edgecolor':'black', 'lw':0.5, 'boxstyle':'round'})
    plt.savefig('cost_plot_fixed_LR_momentum.png', dpi=150)
    plt.clf()

    """ Regression line plot """
    fig, ax = plt.subplots(figsize=(16,9))
    fig.subplots_adjust(bottom=0.22)
    ax.scatter(x_test, y_predict_SGD, c='limegreen', s=5, label='SGD')
    ax.scatter(x_test, y_predict_GD, c='crimson', s=5,label='GD')
    # ax.scatter(x_test, y_predict_OLS, c='dodgerblue', s=5, label='OLS')
    ax.plot(x, y_true, zorder=100, c='black', label='True y')
    ax.scatter(x_train, y_train, c='indigo', marker='o', s=3, alpha=0.3, label='Data') # Data
    ax.set_title(r'\bf{Regression line plot for fixed LR and momentum}', pad=15)
    ax.set_xlabel(r'$x$', labelpad=10)
    ax.set_ylabel(r'$y$',  labelpad=10)
    ax.legend(framealpha=0.9, facecolor=(1, 1, 1, 1))
    string = f'init-LR = {init_LR}, decay = {decay}, momentum = {momentum}, n\_epochs = {n_epochs}, minibatch\_size = {minibatch_size}, n\_minibatches = {n_minibatches}'
    plt.figtext(0.5, 0.05, string, ha="center", fontsize=18, bbox={'facecolor':'white', 'edgecolor':'black', 'lw':0.5, 'boxstyle':'round'})
    plt.savefig('regression_line_plot_fixed_LR_momentum.png', dpi=150)
    plt.clf()

""" (5/8) cost-plot for GD and SGD with an adaptive learning rate and momentum using the chosen lambda """
def adaptive_LR_momentum(x_train, x_test, y_train, y_test, x, y_true, best_poly_deg_GD, best_lambda_GD, best_poly_deg_SGD, best_lambda_SGD, G_M, n_epochs, seed, best_minibatch_size):
    """ Hyperparameters """
    init_LR = 0.1                          # Initial learning rate (LR)
    decay = 0.01                             # init_LR/n_epochs # LR decay rate (for fixed LR set decay to 0)
    momentum = 0.9                        # Momentum value for GD.
    minibatch_size = np.shape(x_train)[0]//best_minibatch_size
    n_minibatches = np.shape(x_train)[0]//minibatch_size #number of minibatches

    """ Optimization method """
    # To get plain GD without any optimization choose 'momentum' with momentum value of 0
    Optimizer_method = ['Adagrad', 'RMSprop', 'Adam','momentum']
    O_M = Optimizer_method[3] # Choose the optimization method

    X_train_GD = designMatrix_1D(x_train, best_poly_deg_GD) # Train design matrix for GD 
    X_test_GD = designMatrix_1D(x_test, best_poly_deg_GD) # Test design matrix for GD

    X_train_SGD = designMatrix_1D(x_train, best_poly_deg_SGD) # Train design matrix for SGD
    X_test_SGD = designMatrix_1D(x_test, best_poly_deg_SGD) # Test design matrix for SGD

    y_predict_GD, theta, test_cost_GD, train_cost_GD = GD(X_train_GD, X_test_GD, y_train, y_test, G_M, O_M, n_epochs*n_minibatches, init_LR, decay, momentum, seed, best_lambda_GD)
    y_predict_SGD, theta, test_cost_SGD, train_cost_SGD = SGD(X_train_SGD, X_test_SGD, y_train, y_test, O_M, G_M, minibatch_size, n_minibatches, n_epochs, init_LR, decay, momentum, seed, best_lambda_SGD)

    plt.rc('axes', facecolor='whitesmoke', edgecolor='none',
    axisbelow=True, grid=True)
    plt.rc('grid', color='w', linestyle='solid')
    plt.rc('lines', linewidth=2)

    iters = np.arange(n_epochs*n_minibatches)
    fig, ax = plt.subplots(figsize=(16,9))
    fig.subplots_adjust(bottom=0.22)
    ax.plot(iters, test_cost_SGD, color='royalblue', lw=2, label=r"Cost for the test data - SGD") #zorder=0,
    ax.plot(iters, test_cost_GD, color='crimson', lw=2, zorder=100, label=r"Cost for the test data - GD") #zorder=0,
    # plt.scatter(best_poly_deg_GD, np.min(MSE_test), color='forestgreen', marker='x', zorder=100, s=150, label='Lowest MSE')
    ax.set_xlabel(r"Iterations", labelpad=10)
    ax.set_ylabel(r"Cost", labelpad=10)
    ax.set_title(r"\bf{Cost as function of iterations for adaptive LR and momentum}", pad=15)
    ax.set_yscale('log')
    ax.legend(framealpha=0.9, facecolor=(1, 1, 1, 1))
    string = f'init-LR = {init_LR}, decay = {decay}, momentum = {momentum}, n\_epochs = {n_epochs}, minibatch\_size = {minibatch_size}, n\_minibatches = {n_minibatches}'
    plt.figtext(0.5, 0.05, string, ha="center", fontsize=18, bbox={'facecolor':'white', 'edgecolor':'black', 'lw':0.5, 'boxstyle':'round'})
    plt.savefig('cost_plot_adaptive_LR_momentum.png', dpi=150)
    plt.clf()

    """ Regression line plot """
    fig, ax = plt.subplots(figsize=(16,9))
    fig.subplots_adjust(bottom=0.22)
    ax.scatter(x_test, y_predict_SGD, c='limegreen', s=5, label='SGD')
    ax.scatter(x_test, y_predict_GD, c='crimson', s=5,label='GD')
    # ax.scatter(x_test, y_predict_OLS, c='dodgerblue', s=5, label='OLS')
    ax.plot(x, y_true, zorder=100, c='black', label='True y')
    ax.scatter(x_train, y_train, c='indigo', marker='o', s=3, alpha=0.3, label='Data') # Data
    ax.set_title(r'\bf{Regression line plot for adaptive LR and momentum}', pad=15)
    ax.set_xlabel(r'$x$', labelpad=10)
    ax.set_ylabel(r'$y$',  labelpad=10)
    ax.legend(framealpha=0.9, facecolor=(1, 1, 1, 1))
    string = f'init-LR = {init_LR}, decay = {decay}, momentum = {momentum}, n\_epochs = {n_epochs}, minibatch\_size = {minibatch_size}, n\_minibatches = {n_minibatches}'
    plt.figtext(0.5, 0.05, string, ha="center", fontsize=18, bbox={'facecolor':'white', 'edgecolor':'black', 'lw':0.5, 'boxstyle':'round'})
    plt.savefig('regression_line_plot_adaptive_LR_momentum.png', dpi=150)
    plt.clf()

""" (6/8) cost-plot for GD and SGD with Adagrad with momentum using the chosen lambda """
def Adagrad_w_momentum(x_train, x_test, y_train, y_test, x, y_true, best_poly_deg_GD, best_lambda_GD, best_poly_deg_SGD, best_lambda_SGD, G_M, n_epochs, seed, best_minibatch_size):
    """ Hyperparameters """
    init_LR = 0.1                          # Initial learning rate (LR)
    decay = 0.01                             # init_LR/n_epochs # LR decay rate (for fixed LR set decay to 0)
    momentum = 0.9                        # Momentum value for GD.
    minibatch_size = np.shape(x_train)[0]//best_minibatch_size
    n_minibatches = np.shape(x_train)[0]//minibatch_size #number of minibatches

    """ Optimization method """
    # To get plain GD without any optimization choose 'momentum' with momentum value of 0
    Optimizer_method = ['Adagrad', 'RMSprop', 'Adam','momentum']
    O_M = Optimizer_method[0] # Choose the optimization method

    X_train_GD = designMatrix_1D(x_train, best_poly_deg_GD) # Train design matrix for GD 
    X_test_GD = designMatrix_1D(x_test, best_poly_deg_GD) # Test design matrix for GD

    X_train_SGD = designMatrix_1D(x_train, best_poly_deg_SGD) # Train design matrix for SGD
    X_test_SGD = designMatrix_1D(x_test, best_poly_deg_SGD) # Test design matrix for SGD

    y_predict_GD, theta, test_cost_GD, train_cost_GD = GD(X_train_GD, X_test_GD, y_train, y_test, G_M, O_M, n_epochs*n_minibatches, init_LR, decay, momentum, seed, best_lambda_GD)
    y_predict_SGD, theta, test_cost_SGD, train_cost_SGD = SGD(X_train_SGD, X_test_SGD, y_train, y_test, O_M, G_M, minibatch_size, n_minibatches, n_epochs, init_LR, decay, momentum, seed, best_lambda_SGD)

    plt.rc('axes', facecolor='whitesmoke', edgecolor='none',
    axisbelow=True, grid=True)
    plt.rc('grid', color='w', linestyle='solid')
    plt.rc('lines', linewidth=2)

    iters = np.arange(n_epochs*n_minibatches)
    fig, ax = plt.subplots(figsize=(16,9))
    fig.subplots_adjust(bottom=0.22)
    ax.plot(iters, test_cost_SGD, color='royalblue', lw=2, label=r"Cost for the test data - SGD") #zorder=0,
    ax.plot(iters, test_cost_GD, color='crimson', lw=2, zorder=100, label=r"Cost for the test data - GD") #zorder=0,
    # plt.scatter(best_poly_deg_GD, np.min(MSE_test), color='forestgreen', marker='x', zorder=100, s=150, label='Lowest MSE')
    ax.set_xlabel(r"Iterations", labelpad=10)
    ax.set_ylabel(r"Cost", labelpad=10)
    ax.set_title(r"\bf{Cost as function of iterations for Adagrad w/ momentum}", pad=15)
    ax.set_yscale('log')
    ax.legend(framealpha=0.9, facecolor=(1, 1, 1, 1))
    string = f'init-LR = {init_LR}, decay = {decay}, momentum = {momentum}, n\_epochs = {n_epochs}, minibatch\_size = {minibatch_size}, n\_minibatches = {n_minibatches}'
    plt.figtext(0.5, 0.05, string, ha="center", fontsize=18, bbox={'facecolor':'white', 'edgecolor':'black', 'lw':0.5, 'boxstyle':'round'})
    plt.savefig('cost_plot_Adagrad_w_momentum.png', dpi=150)
    plt.clf()

    """ Regression line plot """
    fig, ax = plt.subplots(figsize=(16,9))
    fig.subplots_adjust(bottom=0.22)
    ax.scatter(x_test, y_predict_SGD, c='limegreen', s=5, label='SGD')
    ax.scatter(x_test, y_predict_GD, c='crimson', s=5,label='GD')
    # ax.scatter(x_test, y_predict_OLS, c='dodgerblue', s=5, label='OLS')
    ax.plot(x, y_true, zorder=100, c='black', label='True y')
    ax.scatter(x_train, y_train, c='indigo', marker='o', s=3, alpha=0.3, label='Data') # Data
    ax.set_title(r'\bf{Regression line plot for Adagrad w/ momentum}', pad=15)
    ax.set_xlabel(r'$x$', labelpad=10)
    ax.set_ylabel(r'$y$',  labelpad=10)
    ax.legend(framealpha=0.9, facecolor=(1, 1, 1, 1))
    string = f'init-LR = {init_LR}, decay = {decay}, momentum = {momentum}, n\_epochs = {n_epochs}, minibatch\_size = {minibatch_size}, n\_minibatches = {n_minibatches}'
    plt.figtext(0.5, 0.05, string, ha="center", fontsize=18, bbox={'facecolor':'white', 'edgecolor':'black', 'lw':0.5, 'boxstyle':'round'})
    plt.savefig('regression_line_plot_Adagrad_w_momentum.png', dpi=150)
    plt.clf()
  
""" (7/8) cost-plot for GD and SGD with Adagrad without momentum using the chosen lambda """
def Adagrad_w_o_momentum(x_train, x_test, y_train, y_test, x, y_true, best_poly_deg_GD, best_lambda_GD, best_poly_deg_SGD, best_lambda_SGD, G_M, n_epochs, seed, best_minibatch_size):
    """ Hyperparameters """
    init_LR = 0.1                          # Initial learning rate (LR)
    decay = 0.01                             # init_LR/n_epochs # LR decay rate (for fixed LR set decay to 0)
    momentum = 0.0                        # Momentum value for GD.
    minibatch_size = np.shape(x_train)[0]//best_minibatch_size
    n_minibatches = np.shape(x_train)[0]//minibatch_size #number of minibatches

    """ Optimization method """
    # To get plain GD without any optimization choose 'momentum' with momentum value of 0
    Optimizer_method = ['Adagrad', 'RMSprop', 'Adam','momentum']
    O_M = Optimizer_method[0] # Choose the optimization method

    X_train_GD = designMatrix_1D(x_train, best_poly_deg_GD) # Train design matrix for GD 
    X_test_GD = designMatrix_1D(x_test, best_poly_deg_GD) # Test design matrix for GD

    X_train_SGD = designMatrix_1D(x_train, best_poly_deg_SGD) # Train design matrix for SGD
    X_test_SGD = designMatrix_1D(x_test, best_poly_deg_SGD) # Test design matrix for SGD

    y_predict_GD, theta, test_cost_GD, train_cost_GD = GD(X_train_GD, X_test_GD, y_train, y_test, G_M, O_M, n_epochs*n_minibatches, init_LR, decay, momentum, seed, best_lambda_GD)
    y_predict_SGD, theta, test_cost_SGD, train_cost_SGD = SGD(X_train_SGD, X_test_SGD, y_train, y_test, O_M, G_M, minibatch_size, n_minibatches, n_epochs, init_LR, decay, momentum, seed, best_lambda_SGD)

    plt.rc('axes', facecolor='whitesmoke', edgecolor='none',
    axisbelow=True, grid=True)
    plt.rc('grid', color='w', linestyle='solid')
    plt.rc('lines', linewidth=2)

    iters = np.arange(n_epochs*n_minibatches)
    fig, ax = plt.subplots(figsize=(16,9))
    fig.subplots_adjust(bottom=0.22)
    ax.plot(iters, test_cost_SGD, color='royalblue', lw=2, label=r"Cost for the test data - SGD") #zorder=0,
    ax.plot(iters, test_cost_GD, color='crimson', lw=2, zorder=100, label=r"Cost for the test data - GD") #zorder=0,
    # plt.scatter(best_poly_deg_GD, np.min(MSE_test), color='forestgreen', marker='x', zorder=100, s=150, label='Lowest MSE')
    ax.set_xlabel(r"Iterations", labelpad=10)
    ax.set_ylabel(r"Cost", labelpad=10)
    ax.set_title(r"\bf{Cost as function of iterations for Adagrad w/ no momentum}", pad=15)
    ax.set_yscale('log')
    ax.legend(framealpha=0.9, facecolor=(1, 1, 1, 1))
    string = f'init-LR = {init_LR}, decay = {decay}, momentum = {momentum}, n\_epochs = {n_epochs}, minibatch\_size = {minibatch_size}, n\_minibatches = {n_minibatches}'
    plt.figtext(0.5, 0.05, string, ha="center", fontsize=18, bbox={'facecolor':'white', 'edgecolor':'black', 'lw':0.5, 'boxstyle':'round'})
    plt.savefig('cost_plot_Adagrad_w_o_momentum.png', dpi=150)
    plt.clf()

    """ Regression line plot """
    fig, ax = plt.subplots(figsize=(16,9))
    fig.subplots_adjust(bottom=0.22)
    ax.scatter(x_test, y_predict_SGD, c='limegreen', s=5, label='SGD')
    ax.scatter(x_test, y_predict_GD, c='crimson', s=5,label='GD')
    # ax.scatter(x_test, y_predict_OLS, c='dodgerblue', s=5, label='OLS')
    ax.plot(x, y_true, zorder=100, c='black', label='True y')
    ax.scatter(x_train, y_train, c='indigo', marker='o', s=3, alpha=0.3, label='Data') # Data
    ax.set_title(r'\bf{Regression line plot for Adagrad w/ no momentum}', pad=15)
    ax.set_xlabel(r'$x$', labelpad=10)
    ax.set_ylabel(r'$y$',  labelpad=10)
    ax.legend(framealpha=0.9, facecolor=(1, 1, 1, 1))
    string = f'init-LR = {init_LR}, decay = {decay}, momentum = {momentum}, n\_epochs = {n_epochs}, minibatch\_size = {minibatch_size}, n\_minibatches = {n_minibatches}'
    plt.figtext(0.5, 0.05, string, ha="center", fontsize=18, bbox={'facecolor':'white', 'edgecolor':'black', 'lw':0.5, 'boxstyle':'round'})
    plt.savefig('regression_line_plot_Adagrad_w_0_momentum.png', dpi=150)
    plt.clf()

""" (8/8) cost-plot for either GD or SGD with Adagrad, Adam, RMSprop and with momentum using the chosen lambda """
def optim_plot_SGD(x_train, x_test, y_train, y_test, x, y_true, best_poly_deg_SGD, best_lambda_SGD, G_M, n_epochs, seed, best_minibatch_size):

    """ Hyperparameters """
    init_LR = 0.1                          # Initial learning rate (LR)
    decay = 0.01                             # init_LR/n_epochs # LR decay rate (for fixed LR set decay to 0)
    momentum = 0.9                        # Momentum value for GD.
    minibatch_size = np.shape(x_train)[0]//best_minibatch_size
    n_minibatches = np.shape(x_train)[0]//minibatch_size #number of minibatches

    X_train_SGD = designMatrix_1D(x_train, best_poly_deg_SGD) # Train design matrix for SGD
    X_test_SGD = designMatrix_1D(x_test, best_poly_deg_SGD) # Test design matrix for SGD


    # Optimizer_method = ['Adagrad', 'RMSprop', 'Adam','momentum']
    # colors_SGD = ['forestgreen', 'crimson', 'royalblue', 'darkorange']

    Optimizer_method = ['RMSprop', 'Adagrad', 'Adam', 'momentum']
    colors_SGD = ['crimson', 'forestgreen', 'royalblue', 'darkorange']
    zor = [5, 10, 20, 50]

    iters = np.arange(n_epochs*n_minibatches)

    plt.rc('axes', facecolor='whitesmoke', edgecolor='none',
    axisbelow=True, grid=True)
    plt.rc('grid', color='w', linestyle='solid')
    plt.rc('lines', linewidth=2)

    fig, ax = plt.subplots(figsize=(16,9))
    fig.subplots_adjust(bottom=0.22)

    for idx, O_M in enumerate(Optimizer_method):

        y_predict_SGD, theta, test_cost_SGD, train_cost_SGD = SGD(X_train_SGD, X_test_SGD, y_train, y_test, O_M, G_M, minibatch_size, n_minibatches, n_epochs, init_LR, decay, momentum, seed, best_lambda_SGD)

        ax.plot(iters, test_cost_SGD, c=colors_SGD[idx], zorder=zor[idx], lw=2, label=fr"Cost - SGD using {O_M}") #zorder=0,

    # ax.plot(iters, test_cost_SGD, color='royalblue', lw=2, label=r"Cost for the test data - SGD") #zorder=0,
    # plt.scatter(best_poly_deg_GD, np.min(MSE_test), color='forestgreen', marker='x', zorder=100, s=150, label='Lowest MSE')
    ax.set_xlabel(r"Iterations", labelpad=10)
    ax.set_ylabel(r"Cost", labelpad=10)
    ax.set_title(r"\bf{Cost as function of iterations for different optimizers - SGD}", pad=15)
    ax.set_yscale('log')
    ax.legend(framealpha=0.9, facecolor=(1, 1, 1, 1))
    string = f'init-LR = {init_LR}, decay = {decay}, momentum = {momentum}, n\_epochs = {n_epochs}, minibatch\_size = {minibatch_size}, n\_minibatches = {n_minibatches}'
    plt.figtext(0.5, 0.05, string, ha="center", fontsize=18, bbox={'facecolor':'white', 'edgecolor':'black', 'lw':0.5, 'boxstyle':'round'})
    plt.savefig('cost_plot_optims_SGD.png', dpi=150)
    plt.clf()

    """ Regression line plot """
    fig, ax = plt.subplots(figsize=(16,9))
    fig.subplots_adjust(bottom=0.22)
    for idx, O_M in enumerate(Optimizer_method):

        y_predict_SGD, theta, test_cost_SGD, train_cost_SGD = SGD(X_train_SGD, X_test_SGD, y_train, y_test, O_M, G_M, minibatch_size, n_minibatches, n_epochs, init_LR, decay, momentum, seed, best_lambda_SGD)

        ax.scatter(x_test, y_predict_SGD, c=colors_SGD[idx], s=5, label=fr'SGD using {O_M}')

    # ax.scatter(x_test, y_predict_OLS, c='dodgerblue', s=5, label='OLS')
    ax.plot(x, y_true, zorder=100, c='black', label='True y')
    ax.scatter(x_train, y_train, c='indigo', marker='o', s=3, alpha=0.3, label='Data') # Data
    ax.set_title(r'\bf{Regression line plot for different optimizers - SGD}', pad=15)
    ax.set_xlabel(r'$x$', labelpad=10)
    ax.set_ylabel(r'$y$',  labelpad=10)
    ax.legend(framealpha=0.9, facecolor=(1, 1, 1, 1))
    string = f'init-LR = {init_LR}, decay = {decay}, momentum = {momentum}, n\_epochs = {n_epochs}, minibatch\_size = {minibatch_size}, n\_minibatches = {n_minibatches}'
    plt.figtext(0.5, 0.05, string, ha="center", fontsize=18, bbox={'facecolor':'white', 'edgecolor':'black', 'lw':0.5, 'boxstyle':'round'})
    plt.savefig('regression_line_plot_optims_SGD.png', dpi=150)
    plt.clf()

def optim_plot_GD(x_train, x_test, y_train, y_test, x, y_true, best_poly_deg_GD, best_lambda_GD, G_M, n_epochs, seed, best_minibatch_size):

    """ Hyperparameters """
    init_LR = 0.1                          # Initial learning rate (LR)
    decay = 0.01                             # init_LR/n_epochs # LR decay rate (for fixed LR set decay to 0)
    momentum = 0.9                        # Momentum value for GD.
    minibatch_size = np.shape(x_train)[0]//best_minibatch_size
    n_minibatches = np.shape(x_train)[0]//minibatch_size #number of minibatches

    X_train_GD = designMatrix_1D(x_train, best_poly_deg_GD) # Train design matrix for GD
    X_test_GD = designMatrix_1D(x_test, best_poly_deg_GD) # Test design matrix for GD


    Optimizer_method = ['RMSprop', 'Adagrad', 'Adam','momentum']
    colors_GD = ['crimson', 'forestgreen', 'royalblue', 'darkorange']


    iters = np.arange(n_epochs*n_minibatches)


    plt.rc('axes', facecolor='whitesmoke', edgecolor='none',
    axisbelow=True, grid=True)
    plt.rc('grid', color='w', linestyle='solid')
    plt.rc('lines', linewidth=2)

    fig, ax = plt.subplots(figsize=(16,9))
    fig.subplots_adjust(bottom=0.22)

    for idx, O_M in enumerate(Optimizer_method):

        y_predict_GD, theta, test_cost_GD, train_cost_GD = GD(X_train_GD, X_test_GD, y_train, y_test, G_M, O_M, n_epochs*n_minibatches, init_LR, decay, momentum, seed, best_lambda_GD)

        ax.plot(iters, test_cost_GD, c=colors_GD[idx], lw=2, label=fr"Cost - GD using {O_M}") #zorder=0,

    ax.set_xlabel(r"Iterations", labelpad=10)
    ax.set_ylabel(r"Cost", labelpad=10)
    ax.set_title(r"\bf{Cost as function of iterations for different optimizers - GD}", pad=15)
    ax.set_yscale('log')
    ax.legend(framealpha=0.9, facecolor=(1, 1, 1, 1))
    string = f'init-LR = {init_LR}, decay = {decay}, momentum = {momentum}, n\_epochs = {n_epochs}, minibatch\_size = {minibatch_size}, n\_minibatches = {n_minibatches}'
    plt.figtext(0.5, 0.05, string, ha="center", fontsize=18, bbox={'facecolor':'white', 'edgecolor':'black', 'lw':0.5, 'boxstyle':'round'})
    plt.savefig('cost_plot_optims_GD.png', dpi=150)
    plt.clf()

    """ Regression line plot """
    fig, ax = plt.subplots(figsize=(16,9))
    fig.subplots_adjust(bottom=0.22)
    for idx, O_M in enumerate(Optimizer_method):

        y_predict_GD, theta, test_cost_GD, train_cost_GD = GD(X_train_GD, X_test_GD, y_train, y_test, G_M, O_M, n_epochs*n_minibatches, init_LR, decay, momentum, seed, best_lambda_GD)

        ax.scatter(x_test, y_predict_GD, c=colors_GD[idx], s=5, label=fr'GD using {O_M}')

    ax.plot(x, y_true, zorder=100, c='black', label='True y')
    ax.scatter(x_train, y_train, c='indigo', marker='o', s=3, alpha=0.3, label='Data') # Data
    ax.set_title(r'\bf{Regression line plot for different optimizers - GD}', pad=15)
    ax.set_xlabel(r'$x$', labelpad=10)
    ax.set_ylabel(r'$y$',  labelpad=10)
    ax.legend(framealpha=0.9, facecolor=(1, 1, 1, 1))
    string = f'init-LR = {init_LR}, decay = {decay}, momentum = {momentum}, n\_epochs = {n_epochs}, minibatch\_size = {minibatch_size}, n\_minibatches = {n_minibatches}'
    plt.figtext(0.5, 0.05, string, ha="center", fontsize=18, bbox={'facecolor':'white', 'edgecolor':'black', 'lw':0.5, 'boxstyle':'round'})
    plt.savefig('regression_line_plot_optims_GD.png', dpi=150)
    plt.clf()

if __name__=='__main__':

    """ Data: """
    coeff = [1.0, 1.0, 1.0]
    n = 1000 # Number of datapoints
    noise = True
    alpha = 0.1 # Noise scaling
    seed = 55 # Seed for noise (used to replicate results)
    x, y_noise, y_true = polynomial(coeff, n, noise, alpha, seed)

    x_train, x_test, y_train, y_test = train_test_split(x, y_noise, test_size=0.2, random_state = seed)

    """ Gradient Gradient_method """
    Gradient_method = ['auto', 'anal']
    G_M = Gradient_method[1] #Choose the Gradient Gradient_method

    lambda_min = -15
    lambda_max = -1
    nlambdas = 15
    max_polydeg = 3

    plot = True
    n_epochs = 1000
    seed = 55

    """ Plots: """
    best_poly_deg_GD, best_lambda_GD = find_lambda_DG(x_train, x_test, y_train, y_test, x, y_true, G_M, lambda_min, lambda_max, nlambdas, max_polydeg, plot, n_epochs, seed)
    best_poly_deg_SGD, best_lambda_SGD = find_lambda_SDG(x_train, x_test, y_train, y_test, x, y_true,G_M, lambda_min, lambda_max, nlambdas, max_polydeg, plot, n_epochs, seed)
    best_poly_deg_GD += 1
    best_poly_deg_SGD += 1

    best_minibatch_size = find_minibatch_size_SDG(x_train, x_test, y_train, y_test, best_poly_deg_SGD, best_lambda_SGD, G_M, n_epochs, seed)

    fixed_LR(x_train, x_test, y_train, y_test, x, y_true, best_poly_deg_GD, best_lambda_GD, best_poly_deg_SGD, best_lambda_SGD, G_M, n_epochs, seed, best_minibatch_size)
    fixed_LR_momentum(x_train, x_test, y_train, y_test, x, y_true, best_poly_deg_GD, best_lambda_GD, best_poly_deg_SGD, best_lambda_SGD, G_M, n_epochs, seed, best_minibatch_size)
    adaptive_LR_momentum(x_train, x_test, y_train, y_test, x, y_true, best_poly_deg_GD, best_lambda_GD, best_poly_deg_SGD, best_lambda_SGD, G_M, n_epochs, seed, best_minibatch_size)
    Adagrad_w_momentum(x_train, x_test, y_train, y_test, x, y_true, best_poly_deg_GD, best_lambda_GD, best_poly_deg_SGD, best_lambda_SGD, G_M, n_epochs, seed, best_minibatch_size)
    Adagrad_w_o_momentum(x_train, x_test, y_train, y_test, x, y_true, best_poly_deg_GD, best_lambda_GD, best_poly_deg_SGD, best_lambda_SGD, G_M, n_epochs, seed, best_minibatch_size)
    optim_plot_SGD(x_train, x_test, y_train, y_test, x, y_true, best_poly_deg_SGD, best_lambda_SGD, G_M, n_epochs, seed, best_minibatch_size)
    optim_plot_GD(x_train, x_test, y_train, y_test, x, y_true, best_poly_deg_GD, best_lambda_GD, G_M, n_epochs, seed, best_minibatch_size)

    print('\n\n                      FERDIG!!!:))\n\n')