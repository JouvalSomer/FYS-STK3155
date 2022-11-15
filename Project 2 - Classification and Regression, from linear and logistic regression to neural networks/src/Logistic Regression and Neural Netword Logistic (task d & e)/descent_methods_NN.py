"""Descent Gradient Methods"""

import numpy as np
import autograd.numpy as np
from autograd import grad

from func import *
from optimizers import *


def GD(input, output, method, Niterations, init_LR, decay, momentum, seed, lmb):

    """ Gradient Descent """
    np.random.seed(seed)
    w = np.random.randn(np.shape(input)[1],1)
    np.random.seed(seed)
    b = np.random.randn(len(output), 1)
    change_w = change_b = 0.0

    mse = np.zeros(Niterations)
    # Gradient decent:
    for i in range(Niterations):

        if method == 'auto':
            grad_w, grad_b =  auto_gradient_NN(w, b, input, output, lmb) # Autograd
        if method == 'anal':
            grad_w, grad_b = cost_w_b_diff(w, b, input, output, lmb) # Analytical

        eta = learning_schedule(i, init_LR, decay) # LR
        update_w = eta * grad_w + momentum * change_w # Update to the thetas
        update_b = eta * grad_b + momentum * change_b
        if i % 400 == 0:
            print(w.flatten(), b.flatten(), f'iter = {i}')
        w -= update_w
        b -= update_b
        change_w = update_w
        change_b = update_b

        y_predict_GD_test = np.dot(w, input) + b
        mse[i] = MSE(output, y_predict_GD_test)

    y_predict_SGD = np.dot(w, input) + b
    return y_predict_SGD, w, b, mse

def SGD(input, output, Optimizer_method, Gradient_method, minibatch_size, n_minibatches, n_epochs, init_LR, decay, momentum, seed, lmb):

    """ Stochastic Gradient Descent """
    np.random.seed(seed)
    w = np.random.randn(input.shape[1], output.shape[1])
    np.random.seed(seed)
    b = np.random.randn(output.shape[1], 1)

    change_w = change_b = 0.0

    np.random.seed(seed)
    random_index = minibatch_size*np.random.randint(n_minibatches)
    X_batch = input[random_index:random_index + minibatch_size]
    y_batch = output[random_index:random_index + minibatch_size]


    mse = np.zeros(n_epochs*n_minibatches)
    count = 0
    eta = init_LR

    """ Optimizer method """
    if Optimizer_method == 'Adagrad':
        optim_w = Adagrad(eta)
        optim_b = Adagrad(eta)
    if Optimizer_method == 'RMSprop':
        optim_w = RMSprop(eta)
        optim_b = RMSprop(eta)
    if Optimizer_method == 'Adam':
        optim_w = Adam(eta)
        optim_b = Adam(eta)


    for epoch in range(n_epochs):
        Giter_w = np.zeros(shape=(input.shape[1],input.shape[1]))
        Giter_b = np.zeros(shape=(input.shape[1],input.shape[1]))

        for batch in range(n_minibatches):

            """ Gradient method """
            if Gradient_method == 'auto':
                grad_w, grad_b =  auto_gradient_NN(w, b, X_batch, y_batch, lmb) # Autograd
            if Gradient_method == 'anal':
                grad_w, grad_b = cost_w_b_diff(w, b, X_batch, y_batch, lmb) # Analytical

            """ Optimizer method """
            if Optimizer_method == 'Adagrad':
                update_w = optim_w(grad_w, Giter_w)
                update_b = optim_b(grad_b, Giter_b)
                if count % 400 == 0:
                    print(w.flatten(), b.flatten(), f'iter = {count}')
                w -= update_w
                b -= update_b

            if Optimizer_method == 'RMSprop':
                update_w = optim_w(grad_w, Giter_w)
                update_b = optim_b(grad_b, Giter_b)
                if count % 400 == 0:
                    print(w.flatten(), b.flatten(), f'iter = {count}')
                w -= update_w
                b -= update_b

            if Optimizer_method == 'Adam':
                update_w = optim_w(grad_w)
                update_b = optim_b(grad_b)
                if count % 400 == 0:
                    print(w.flatten(), b.flatten(), f'iter = {count}')
                w -= update_w
                b -= update_b

            if Optimizer_method == 'momentum':
                eta = learning_schedule(epoch, init_LR, decay) # LR
                update_w = eta * grad_w + momentum * change_w # Update to the thetas
                update_b = eta * grad_b + momentum * change_b
                if count % 400 == 0:
                    print(w.flatten(), b.flatten(), f'iter = {count}')
                w -= update_w
                b -= update_b
                change_w = update_w
                change_b = update_b
            
            y_predict_GD_test = input @ w + b
            mse[count] = MSE(output, y_predict_GD_test)
            count +=1

    y_predict_SGD = input @ w + b
    return y_predict_SGD, w, b, mse
