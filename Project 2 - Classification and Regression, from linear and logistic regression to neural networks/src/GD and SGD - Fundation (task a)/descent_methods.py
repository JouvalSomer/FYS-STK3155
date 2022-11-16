from func import *
from optimizers import *

import numpy as np
from sklearn.utils import shuffle


def GD(X_train, X_test, y_train, y_test, Gradient_method, Optimizer_method, Niterations, init_LR, decay, momentum, seed, lmb):

    """ Gradient Decent """
    np.random.seed(seed)
    theta = np.random.randn(np.shape(X_train)[1],1) # Initial thetas/betas

    change = 0.0
    mse_test = np.zeros(Niterations)
    mse_train = np.zeros(Niterations)
    
    """ Optimizer method """
    if Optimizer_method == 'Adagrad':
        optim = Adagrad(init_LR)
    if Optimizer_method == 'RMSprop':
        optim = RMSprop(init_LR)
    if Optimizer_method == 'Adam':
        optim = Adam(init_LR)

    # Gradient decent:
    Giter = np.zeros(shape=(X_train.shape[1],X_train.shape[1]))
    for i in range(Niterations):

        """ Gradient method """
        if Gradient_method == 'auto':  
            gradients =  auto_gradient(theta, X_train, y_train, lmb) # Autograd
        if Gradient_method == 'anal':
            gradients = cost_theta_diff(theta, X_train, y_train, lmb) # Analytical

        """ Optimizer method """
        if Optimizer_method == 'Adagrad':
            update = optim(gradients, Giter)#uses class
            theta -= update

        if Optimizer_method == 'RMSprop':
            update = optim(gradients, Giter)#uses class
            theta -= update

        if Optimizer_method == 'Adam':
            update = optim(gradients)
            theta -= update

        if Optimizer_method == 'momentum':
            eta = learning_schedule(i, init_LR, decay) # LR
            update = eta * gradients + momentum * change # Update to the thetas

            theta -= update
            change = update # Update the amount the momentum gets added

        y_predict_GD_test = X_test @ theta
        mse_test[i] = cost(y_test, y_predict_GD_test, lmb, theta)

        y_predict_GD_train = X_train @ theta
        mse_train[i] = cost(y_train, y_predict_GD_train, lmb, theta)

    y_predict_GD = X_test @ theta

    return y_predict_GD, theta, mse_test, mse_train

def SGD(X_train, X_test, y_train, y_test, Optimizer_method, Gradient_method, minibatch_size, n_minibatches, n_epochs, init_LR, decay, momentum, seed, lmb):
    """ Stochastic Gradient Decent """

    np.random.seed(seed)
    theta = np.random.randn(np.shape(X_train)[1],1) # Initial thetas/betas

    mse_test = np.zeros(n_epochs*n_minibatches)
    mse_train = np.zeros(n_epochs*n_minibatches)

    count = 0
    change = 0.0

    """ Optimizer method """
    if Optimizer_method == 'Adagrad':
        optim = Adagrad(init_LR)
    if Optimizer_method == 'RMSprop':
        optim = RMSprop(init_LR)
    if Optimizer_method == 'Adam':
        optim = Adam(init_LR)

    for epoch in range(n_epochs):
        Giter = np.zeros(shape=(X_train.shape[1],X_train.shape[1]))
        for batch in range(n_minibatches):

            random_index = minibatch_size*np.random.randint(n_minibatches)
            X_batch = X_train[random_index:random_index+minibatch_size]
            y_batch = y_train[random_index:random_index+minibatch_size]


            """ Gradient method """
            if Gradient_method == 'auto':  
                gradients =  auto_gradient(theta, X_batch, y_batch, lmb) # Autograd
            if Gradient_method == 'anal':
                gradients = cost_theta_diff(theta, X_batch, y_batch, lmb) # Analytical

            """ Optimizer method """
            if Optimizer_method == 'Adagrad':
                update = optim(gradients, Giter)#uses class
                theta -= update

            if Optimizer_method == 'RMSprop':
                update = optim(gradients, Giter)#uses class
                theta -= update

            if Optimizer_method == 'Adam':
                update = optim(gradients)
                theta -= update

            if Optimizer_method == 'momentum':
                eta = learning_schedule(epoch, init_LR, decay) # LR
                update = eta * gradients + momentum * change # Update to the thetas
                theta -= update
                change = update


            y_predict_SGD_test = X_test @ theta
            mse_test[count] = cost(y_test, y_predict_SGD_test, lmb, theta)

            y_predict_SGD_train = X_train @ theta
            mse_train[count] = cost(y_train, y_predict_SGD_train, lmb, theta)
            count += 1

    y_predict_GD = X_test @ theta
    return y_predict_GD, theta, mse_test, mse_train
