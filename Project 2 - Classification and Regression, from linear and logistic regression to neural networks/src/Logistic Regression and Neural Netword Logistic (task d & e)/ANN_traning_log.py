"""Neural Network for Binary Classification"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor

from activation import*
from layer import*
from train_NN_Class import*
from func import*

#Breast Cancer Data set:
from sklearn.datasets import load_breast_cancer

#PCA:
from sklearn.decomposition import PCA


"""Load the data"""
cancer = load_breast_cancer()
X = cancer.data
Y = cancer.target.reshape(-1,1)

"""Corr matrix:"""
# cancerpd = pd.DataFrame(cancer.data, columns=cancer.feature_names)
# correlation_matrix = cancerpd.corr().round(1)

"""Bootstrapping over B boots to find mean Cost and Accuracy:"""
B=50
costboot = np.zeros((B)).reshape(-1,1)
acc_boot = np.zeros((B)).reshape(-1,1)

for boot in range (B):

    """Split Randomly"""
    X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.20)

    """The Standardscaler"""
    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)


    """PCA"""
    # pca = PCA(n_components=20)
    # X_train = pca.fit_transform(X_train)
    # X_test = pca.transform(X_test)

    # print("PCA explained variance:")
    # print(pca.explained_variance_ratio_)
    # print("PCA components:")
    # print(abs( pca.components_ ))

    """Nodes:"""
    n_nodes_inputLayer = X_train.shape[1]
    n_nodes_outputLayer = y_train.shape[1]

    ANN_GD = [
        Layer(n_nodes_inputLayer, 18),
        Sigmoid(),
        Layer(18, 18),
        Sigmoid(),
        Layer(18, n_nodes_outputLayer),
        Sigmoid()
    ]

    ANN_SGD = [
        Layer(n_nodes_inputLayer, 18),
        Sigmoid(),
        Layer(18, 18),
        Sigmoid(),
        Layer(18, n_nodes_outputLayer),
        Sigmoid()
    ]

    """ Hyperparameters """
    n_epochs = 200          # Number of epochs
    init_LR = 0.0025        # Initial learning rate (LR)
    decay = 0.0             # init_LR/n_epochs # LR decay rate (for fixed LR set decay to 0)
    momentum = 0.8          # Momentum value for GD and SGD.
    minibatch_size = 200
    n_minibatches =  (X_train.shape[0]//minibatch_size) #number of minibatches
    N_iter_GD = n_epochs*n_minibatches
    lmb =  1e-12
    seed = np.random.randint(0, 100000)
    #seed = 55 #for validation.


    """ Optimization method """
    # If you want plain GD without any optimization choose 'momentum' with momentum value of 0
    Optimizer_method = ['Adam','momentum'] # Adagrad and RMSprop not yet implemented
    O_M = Optimizer_method[0] #Choose the optimization method

    mse_SGD = train_NN_SGD(ANN_SGD, X_train, y_train, n_epochs, init_LR, decay, O_M, momentum, minibatch_size, n_minibatches, seed, lmb)
    mse_GD = train_NN_GD(ANN_GD, X_train, y_train, N_iter_GD, init_LR, decay, O_M, momentum, lmb)
    #print(mse_SGD[-1])

    y_pred_SGD = fwd(ANN_SGD, X_test)[0]
    y_pred_GD = fwd(ANN_GD, X_test)[0]

    """TBD: Lage TEST COST PLOT"""


    costboot[boot] = mse_SGD[-1] #stores last train MSE.

    tol = 0.50
    """Classsfication:"""
    y_predicted = y_pred_SGD.copy()
    y_predicted[y_predicted >= tol] = 1
    y_predicted[y_predicted < tol] = 0

    """Accuracy Score Based on the test set:"""
    acc_score = accuracy(y_predicted,y_test)
    acc_boot[boot] = acc_score

"""The mean acc score and mean cost of the B bootstraps:"""
print(" ")
print("RESULTS FROM THE BOOTSTRAP: ")

acc_confint = np.percentile(acc_boot,[2.5,97.5])

print(f"The mean accuracy score is {np.mean(acc_boot):.6f}  and its standard dev is {np.std(acc_boot):.6f} based on {B} boot runs.")

print(" ")
# Print the confidence interval
print('Accuracy score within 95% interval of =', acc_confint, 'percent')
print(" ")
print(f"The mean train entropy cost and its standard dev from {B} boot runs:")
#print(costboot)
print(f"{np.mean(costboot):.8f},{np.std(costboot):.8f}")
# Print the confidence interval
print(" ")
cost_confint = np.percentile(costboot,[2.5,97.5])
print(f"95% confidence interval of the train cost for {B} boot runs is {cost_confint}")


# Plot the histogram of the boot accuracies:
plt.hist(acc_boot, bins=20, density=True)
plt.xlabel(r'Accuracy Score')
plt.ylabel('PDF')
plt.savefig("accuracy.jpg")
plt.show()


# Plot the histogram of the boot cost:
plt.hist(costboot, bins=20, density=True)
plt.xlabel(r' Cost')
plt.ylabel('PDF')
plt.savefig("cost.jpg")
plt.show()


print(" ")
print("RESULT FROM THE LAST RUN IN THE BOOTSTRAP:")
print(" ")
print(f"Correctly scored {np.sum(y_predicted==y_test)} out of {y_test.shape[0]}")

print(f"Incorrectly scored {np.sum(y_predicted!=y_test)} out of {y_test.shape[0]}")

print("Scored positive by the NN:")
print(int(np.sum(y_predicted)))

print("Scored negative by the NN:")
print(int(np.sum(1 - y_predicted)))


print("Correctly scored positive by NN out of total positive:")
print(int(np.sum(y_predicted*y_test)),np.sum(y_test))
#
print("Correctly scored negative by NN out of total negative:")
print(int(np.sum((1-y_predicted)*(1-y_test))), np.sum(1-y_test))




# plt.rcParams.update({
# "text.usetex": True,
# "font.family": "serif",
# "font.serif": ["ComputerModern"]})
# plt.rcParams['figure.figsize'] = (8,6)
# plt.rcParams.update({'font.size': 20})
# plt.rc('axes', facecolor='whitesmoke', edgecolor='none',
#     axisbelow=True, grid=True)
# plt.rc('grid', color='w', linestyle='solid')
# plt.rc('lines', linewidth=2)
#
# fig, ax = plt.subplots()
# fig.subplots_adjust(bottom=0.22)
# """ Regression line plot """
# ax.scatter(X, y_pred_SGD, c='crimson', s=5, label='SGD')
# # print(x_train.shape, y_pred.shape)
# ax.scatter(X, y_pred_GD, c='limegreen', s=5, label='GD')
# # ax.scatter(x_test, y_predict_OLS, c='dodgerblue', s=5, label='OLS')
# ax.scatter(X, Y_true, zorder=100, c='black', s=4, label='True y')
# ax.scatter(X, Y, c='indigo', marker='o', s=3, alpha=0.2, label='Data') # Data
# ax.set_title(r'Regression line plot', pad=15)
# ax.set_xlabel(r'$x$', labelpad=10)
# ax.set_ylabel(r'$y$',  labelpad=10)
# ax.legend(framealpha=0.9, facecolor=(1, 1, 1, 1))
# string = f'init-LR = {init_LR}, decay = {decay}, momentum = {momentum}, Niterations = {N_iter_GD}, n\_epochs = {n_epochs}, minibatch\_size = {minibatch_size}, n\_minibatches = {n_minibatches}'
# plt.figtext(0.5, 0.04, string, ha="center", fontsize=18, bbox={'facecolor':'white', 'edgecolor':'black', 'lw':0.5, 'boxstyle':'round'})
# plt.show()

iters = np.arange(n_epochs*n_minibatches)
plt.yscale('log')
plt.plot(iters, mse_GD, label='GD Cost')
plt.plot(iters, mse_SGD, label='SGD Cost')
plt.title('Cost as Function of Iteration')
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.legend(framealpha=0.9, facecolor=(1, 1, 1, 1))
plt.savefig("costplot.jpg")
plt.show()
