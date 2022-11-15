"""Plot functions"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from optimizers import *
from descent_methods import *
from func import *

""" PLOTS """
plt.rcParams.update({
"text.usetex": True,
"font.family": "serif",
"font.serif": ["ComputerModern"]})
plt.rcParams['figure.figsize'] = (8,6)
plt.rcParams.update({'font.size': 20})
plt.rc('axes', facecolor='whitesmoke', edgecolor='none',
    axisbelow=True, grid=True)
plt.rc('grid', color='w', linestyle='solid')
plt.rc('lines', linewidth=2)


""" Data """
seed = 55 #STIG: seed has an effect..
n = 10000
x = np.linspace(0, 1, n)
# exact_theta = [1.0, 4.5, 8.2]
exact_theta = [1.0, 1.0, 1.0]

alpha = 0.1 # Noise scaling set to 0 for test.
y = y_func(x, exact_theta)
y = (y + alpha*np.random.normal(0, 1, x.shape)).reshape(-1, 1)


""" Train Test Split """
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state = seed)


features = len(exact_theta)
X_train = designMatrix_1D(x_train, features)
X_test = designMatrix_1D(x_test, features)


""" Hyperparameters """
n_epochs = 100 #number of epochs
init_LR = 0.01 # Initial learning rate (LR)
decay = 0.01 #init_LR/n_epochs # LR decay rate (for fixed LR set decay to 0)
momentum = 0.9 # Momentum value for GD.
minibatch_size = 100
n_minibatches = int(np.shape(X_train)[0]/minibatch_size) #number of minibatches
Niterations = n_epochs*n_minibatches# Number of GD iterations
lmb = 0

""" Gradient method """
Gradient_method = ['auto', 'anal']
G_M = Gradient_method[1] #Choose the Gradient method

""" Optimization method """
# If you want plain GD without any optimization choose 'momentum' with momentum value of 0
Optimizer_method = ['Adagrad', 'RMSprop', 'Adam','momentum']
O_M = Optimizer_method[3] #Choose the optimization method


""" Results """
# y_predict_GD, w_GD, b_GD, mse_GD = GD(X_train, X_test, y_train, y_test, G_M, Niterations, init_LR, decay, momentum, seed)
y_predict_SGD, w_SGD, b_SGD, mse_SGD = SGD(X_train, X_test, y_train, y_test, O_M, G_M, minibatch_size, n_minibatches, n_epochs, init_LR, decay, momentum, seed, lmb)
# y_predict_OLS, theta_OLS = OLS(X_train, X_test, y_train)



# y_predict_GD_test = np.dot(w_GD, X_test) + b_GD
# print(f'MSE_GD  = {MSE(y_test, y_predict_GD_test):.5f}')
y_predict_SGD_test = X_test @ w_SGD + b_SGD
print(f'MSE_SGD = {MSE(y_test, y_predict_SGD_test):.5f}')

# y_predict_OLS_test = X_test @ theta_OLS
# print(f'MSE_OLS = {MSE(y_test, y_predict_OLS_test):.5f} \n')



fig, ax = plt.subplots()
fig.subplots_adjust(bottom=0.22)
""" Regression line plot """
# ax.scatter(x_test, y_predict_GD, c='crimson', s=5,label='GD')
ax.scatter(x_train, y_predict_SGD, c='limegreen', s=5, label='SGD')
# ax.scatter(x_test, y_predict_OLS, c='dodgerblue', s=5, label='OLS')
ax.plot(x, y_func(x, exact_theta), zorder=100, c='black', label='True y')
ax.scatter(x_train, y_train, c='indigo', marker='o', s=3, alpha=0.05, label='Data') # Data
ax.set_title(r'Regression line plot', pad=15)
ax.set_xlabel(r'$x$', labelpad=10)
ax.set_ylabel(r'$y$',  labelpad=10)
ax.legend(framealpha=0.9, facecolor=(1, 1, 1, 1))
string = f'init-LR = {init_LR}, decay = {decay}, momentum = {momentum}, Niterations = {Niterations}' # n\_epochs = {n_epochs}, minibatch\_size = {minibatch_size}, n\_minibatches = {n_minibatches}
plt.figtext(0.5, 0.04, string, ha="center", fontsize=18, bbox={'facecolor':'white', 'edgecolor':'black', 'lw':0.5, 'boxstyle':'round'})
plt.show()

""" MSE plot """
fig, ax = plt.subplots()
fig.subplots_adjust(bottom=0.22)
plt.yscale('log')
# plt.plot(np.arange(Niterations), mse_GD, label='MSE for GD')
plt.plot(range(n_epochs*n_minibatches), mse_SGD, label='MSE for SGD')
plt.title(r'MSE plot', pad=15)
plt.xlabel(r'Iterations',  labelpad=10)
plt.ylabel(r'MSE',  labelpad=10)
plt.legend(framealpha=0.9, facecolor=(1, 1, 1, 1))
string = f'init-LR = {init_LR}, decay = {decay}, momentum = {momentum}, Niterations = {Niterations}'
plt.figtext(0.5, 0.04, string, ha="center", fontsize=18, bbox={'facecolor':'white', 'edgecolor':'black', 'lw':0.5})
plt.show()
