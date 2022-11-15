
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
from matplotlib import cm


from activation import*
from layer import*
from train_NN import*
from func import*


# coeff = [1.0, 1.0, 1.0]
# n = 1000
# alpha = 0.1
# noise = True
seed = 55

X_no_noise, Y_no_noise, Z_no_noise = data_FF(noise=False, step_size=0.02, alpha=0.05, reshape=False)

# X, Y, Y_true = polynomial(coeff, n, noise, alpha, seed)

X, Y, Z = data_FF(noise=True, step_size=0.02, alpha=0.05, reshape=True)

x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(X, Y, Z, test_size=0.25, random_state = seed)


input_train = np.c_[x_train, y_train]
input_test = np.c_[x_test, y_test]


n_nodes_inputLayer = input_train.shape[1]
n_nodes_outputLayer = z_train.shape[1]

ANN_GD = [
    Layer(n_nodes_inputLayer, 50),
    Hyperbolic(),
    Layer(50, 50),
    Hyperbolic(),
    Layer(50, n_nodes_outputLayer),
    Linear_Activation()
]

ANN_SGD = [
    Layer(n_nodes_inputLayer, 50),
    Hyperbolic(),
    Layer(50, n_nodes_outputLayer),
    Linear_Activation()
]


""" Hyperparameters """
n_epochs = 100          # Number of epochs
init_LR = 0.01          # Initial learning rate (LR)
decay = 0.0             # init_LR/n_epochs # LR decay rate (for fixed LR set decay to 0)
momentum = 0.8          # Momentum value for GD and SGD.
minibatch_size = 50
n_minibatches =  10 # x_train.shape[0]//minibatch_size #number of minibatches
N_iter_GD = n_epochs*n_minibatches
lmb = 0 # 1e-1
seed = 55 #np.random.randint(0, 100)

""" Optimization method """
# If you want plain GD without any optimization choose 'momentum' with momentum value of 0
Optimizer_method = ['Adam','momentum'] # Adagrad and RMSprop not yet implemented 
O_M = Optimizer_method[0] #Choose the optimization method

problem_method = ['Regg', 'Class']
method = problem_method[0]

# mse_SGD_train, mse_SGD_test = train_NN_SGD(ANN_SGD, input_train, z_train, input_test, z_test, n_epochs, init_LR, decay, O_M, momentum, minibatch_size, n_minibatches, seed, lmb, method)
mse_GD_train, mse_GD_test = train_NN_GD(ANN_GD, input_train, z_train, input_test, z_test, N_iter_GD, init_LR, decay, O_M, momentum, lmb, method)

# y_pred_SGD = fwd(ANN_SGD, input_test)[0]
# y_pred_GD = fwd(ANN_GD, input_test)[0]


plt.rcParams.update({
"text.usetex": True,
"font.family": "serif",
"font.serif": ["ComputerModern"]})


x = np.linspace(0, 1, 50).reshape(-1,1)
y = np.linspace(0, 1, 50).reshape(-1,1)
X, Y = np.meshgrid(x, y)
x = X.flatten().reshape(-1, 1)
y = Y.flatten().reshape(-1, 1)
i = np.c_[x, y]
z = fwd(ANN_GD, i)[0]
x = X
y = Y
z = z.reshape(50, 50)

z_diff = abs(z - Z_no_noise)


def FF_plot():
    plt.rcParams['figure.figsize'] = (16,12)
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

    ax.plot_surface(x, y, z, alpha=1, cmap="autumn")

    # ax.plot_surface(X_no_noise, Y_no_noise, Z_no_noise, rstride=8, cstride=8, alpha=0.7, cmap="Blues")

    cset = ax.contourf(X, Y, z_diff, zdir='z', offset=-0.79, cmap='winter')
    cbaxes = fig.add_axes([0.21, 0.3, 0.03, 0.4]) 
    cbar = fig.colorbar(cset, pad=0.1, shrink=0.5, cax = cbaxes, ticks=[np.min(z_diff), np.max(z_diff)/2, np.max(z_diff)])
    cbar.ax.set_ylabel(r'Difference')
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

def poly_line_plot():
    plt.rcParams['figure.figsize'] = (8,6)
    plt.rcParams.update({'font.size': 20})
    fig, ax = plt.subplots()
    fig.subplots_adjust(bottom=0.22)
    """ Regression line plot """
    ax.scatter(x_test, y_pred_SGD, c='crimson', s=5, label='SGD')
    # print(x_train.shape, y_pred.shape)
    ax.scatter(x_test, y_pred_GD, c='limegreen', s=5, label='GD')
    # ax.scatter(x_test, y_predict_OLS, c='dodgerblue', s=5, label='OLS')
    ax.scatter(X, Y_true, zorder=100, c='black', s=4, label='True y')
    ax.scatter(X, Y, c='indigo', marker='o', s=3, alpha=0.2, label='Data') # Data
    ax.set_title(r'Regression line plot', pad=15)
    ax.set_xlabel(r'$x$', labelpad=10)
    ax.set_ylabel(r'$y$',  labelpad=10)
    ax.legend(framealpha=0.9, facecolor=(1, 1, 1, 1))
    string = f'init-LR = {init_LR}, decay = {decay}, momentum = {momentum}, Niterations = {N_iter_GD}, n\_epochs = {n_epochs}, minibatch\_size = {minibatch_size}, n\_minibatches = {n_minibatches}'
    plt.figtext(0.5, 0.04, string, ha="center", fontsize=18, bbox={'facecolor':'white', 'edgecolor':'black', 'lw':0.5, 'boxstyle':'round'})
    plt.show()

iters = np.arange(n_epochs*n_minibatches)
def MSE_plot_GD():
    plt.rcParams['figure.figsize'] = (8,6)
    plt.rcParams.update({'font.size': 20})
    plt.rc('axes', facecolor='whitesmoke', edgecolor='none',
        axisbelow=True, grid=True)
    plt.rc('grid', color='w', linestyle='solid')
    plt.rc('lines', linewidth=2)

    plt.yscale('log')
    plt.plot(iters, mse_GD_test, label='GD MSE test')
    plt.plot(iters, mse_GD_train, label='GD MSE train')
    plt.title('MSE as Function of Iteration')
    plt.xlabel('Iterations')
    plt.ylabel('MSE')
    plt.legend(framealpha=0.9, facecolor=(1, 1, 1, 1))
    plt.show()

def MSE_plot_SGD():
    plt.rcParams['figure.figsize'] = (8,6)
    plt.rcParams.update({'font.size': 20})
    plt.rc('axes', facecolor='whitesmoke', edgecolor='none',
        axisbelow=True, grid=True)
    plt.rc('grid', color='w', linestyle='solid')
    plt.rc('lines', linewidth=2)
    plt.yscale('log')
    plt.plot(iters, mse_SGD_test, label='SGD MSE test')
    plt.plot(iters, mse_SGD_train, label='SGD MSE train')
    plt.title('MSE as Function of Iteration')
    plt.xlabel('Iterations')
    plt.ylabel('MSE')
    plt.legend(framealpha=0.9, facecolor=(1, 1, 1, 1))
    plt.show()

def MSE_plot_diff_GD_SGD():
    plt.rcParams['figure.figsize'] = (8,6)
    plt.rcParams.update({'font.size': 20})
    plt.rc('axes', facecolor='whitesmoke', edgecolor='none',
        axisbelow=True, grid=True)
    plt.rc('grid', color='w', linestyle='solid')
    plt.rc('lines', linewidth=2)
    plt.yscale('log')
    plt.plot(iters, mse_SGD_test, label='SGD MSE test')
    plt.plot(iters, mse_GD_test, label='GD MSE test')
    plt.title('MSE as Function of Iteration')
    plt.xlabel('Iterations')
    plt.ylabel('MSE')
    plt.legend(framealpha=0.9, facecolor=(1, 1, 1, 1))
    plt.show()


""" Heatmap for RIDGE"""
# def MSE_heatmap_RIDGE(input_train, z_train, input_test, z_test, n_epochs, init_LR, decay, O_M, momentum, minibatch_size, n_minibatches, seed, lmb, lambda_min, lambda_max, nlambdas):
def MSE_heatmap_RIDGE(X, Y, Z, n_epochs, init_LR, decay, O_M, momentum, minibatch_size, n_minibatches, lmb, lambda_min, lambda_max, nlambdas):
    
    fig, ax = plt.subplots(figsize=(16,12))
    fig.subplots_adjust(top=0.2)

    plt.rcParams.update({'font.size': 26})
    plt.rcParams['axes.titlepad'] = 40 # Space between the title and graph

    lambdas = np.logspace(lambda_min, lambda_max, nlambdas)
    n_nodes = [5, 10, 50, 100]

    MSE_nodes_lambda_GD = np.empty((len(n_nodes), nlambdas))
    MSE_nodes_lambda_SGD = np.empty((len(n_nodes), nlambdas))
    seeds = np.arange(3)
    for n_idx, nodes in enumerate(n_nodes):
        for l_idx, lmb in enumerate(lambdas):
            mse_SGD = 0
            mse_GD = 0
            for seed in seeds:
                print(seed)
                x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(X, Y, Z, test_size=0.2, random_state = seed)

                input_train = np.c_[x_train, y_train]
                input_test = np.c_[x_test, y_test]

                n_nodes_inputLayer = input_train.shape[1]
                n_nodes_outputLayer = z_train.shape[1]

                ANN_GD = [
                Layer(n_nodes_inputLayer, nodes),
                Hyperbolic(),
                Layer(nodes, nodes),
                Hyperbolic(),
                Layer(nodes, nodes),
                Hyperbolic(),
                Layer(nodes, n_nodes_outputLayer),
                Linear_Activation()
                ]

                ANN_SGD = [
                Layer(n_nodes_inputLayer, nodes),
                Hyperbolic(),
                Layer(nodes, nodes),
                Hyperbolic(),
                Layer(nodes, nodes),
                Hyperbolic(),
                Layer(nodes, n_nodes_outputLayer),
                Linear_Activation()
                ]

                mse_SGD_train, mse_SGD_test = train_NN_SGD(ANN_SGD, input_train, z_train, input_test, z_test, n_epochs, init_LR, decay, O_M, momentum, minibatch_size, n_minibatches, seed, lmb)
                mse_GD_train, mse_GD_test = train_NN_GD(ANN_GD, input_train, z_train, input_test, z_test, N_iter_GD, init_LR, decay, O_M, momentum, lmb)

                # y_pred_SGD = fwd(ANN_SGD, input_test)[0]
                # y_pred_GD = fwd(ANN_GD, input_test)[0]

                mse_SGD += mse_SGD_test[-1]
                mse_GD += mse_GD_test[-1]

            MSE_nodes_lambda_GD[n_idx,l_idx] = mse_GD/len(seeds)
            MSE_nodes_lambda_SGD[n_idx,l_idx] = mse_SGD/len(seeds)

    # index = np.argwhere(MSE_degree_lambda == np.min(MSE_degree_lambda))

    # if index[0,0] == 0:
    #     index[0,0] += 1

    # if index[0,1] == 0:
    #     index[0,1] += 1

    # best_poly_deg_MSE = poly_deg[index[0,0]]
    # best_lambda_MSE_index = index[0,1]


    # max_order = int(np.ceil(1.5*best_poly_deg_MSE))
    # min_order = int(np.floor(0.5*best_poly_deg_MSE))

    # if (max_order - min_order) <= 2:
    #     max_order += 1

    # max_lambda = int(np.ceil(1.5*best_lambda_MSE_index))
    # min_lambda = int(np.floor(0.5*best_lambda_MSE_index))

    # if (max_lambda - min_lambda) <= 2:
    #     max_lambda += 1

    # if max_order > order:
    #     max_order = order

    # if max_lambda > len(lambdas):
    #     max_lambda = len(lambdas)

    # print('Poly. grad. =', min_order, max_order)
    # print('lambda = ', min_lambda, max_lambda)
    # print(np.shape(MSE_degree_lambda[min_order:max_order, min_lambda:max_lambda]))

    sns.heatmap(MSE_nodes_lambda_GD.T, cmap="RdYlGn_r",
    xticklabels=[str(deg) for deg in n_nodes],
    yticklabels=[str(f'{lam:1.1E}') for lam in lambdas],
    annot=True, annot_kws={"size": 12},
    fmt="1.4f", linewidths=1, linecolor=(30/255,30/255,30/255,1),
    cbar_kws={"orientation": "horizontal", "shrink":0.8, "aspect":40, "label": "MSE", "pad":0.05})
    ax.set_xlabel("Nodes in the layer", labelpad=10)
    ax.set_ylabel(r'$\log_{10} \lambda$', labelpad=10)
    ax.set_title(r'\bf{MSE Heatmap}')
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    plt.tight_layout()
    plt.show()

if __name__=='__main__':
    FF_plot()
    # poly_line_plot()
    # MSE_plot_GD()
    # MSE_plot_SGD()
    # MSE_plot_diff_GD_SGD()
    lambda_min, lambda_max = -15, -1
    nlambdas = 7
    # MSE_heatmap_RIDGE(X, Y, Z, n_epochs, init_LR, decay, O_M, momentum, minibatch_size, n_minibatches, lmb, lambda_min, lambda_max, nlambdas)
    # MSE_heatmap_RIDGE(input_train, z_train, input_test, z_test, n_epochs, init_LR, decay, O_M, momentum, minibatch_size, n_minibatches, lmb, lambda_min, lambda_max, nlambdas)