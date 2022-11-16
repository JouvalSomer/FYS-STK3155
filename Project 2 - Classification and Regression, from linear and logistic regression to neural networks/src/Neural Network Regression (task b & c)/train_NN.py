""" The neural FFNN"""
from sklearn.metrics import r2_score
import numpy as np

from func import*
from optimizers import *

def fwd(ANN, input):
    output = input
    for idx, layer in enumerate(ANN):
        output, weights = layer.forward(output)
        if idx == (len(ANN) - 2):
            wei = weights
    return output, wei

def fwd_test(ANN, input):
    output = input
    for idx, layer in enumerate(ANN):
        output, weights = layer.forward(output)
        if idx == (len(ANN) - 2):
            wei = weights
    return output, wei

                
def train_NN_GD(ANN, x_train, y_train, x_test, y_test, epochs, learning_rate, decay, O_M, momentum, lmb, method):

    """ Optimizer method """
    if O_M == 'Adagrad':
        Layer_optim = []
        for _ in range(int(len(ANN)/2)):
            optim_w = Adagrad(learning_rate)
            optim_b = Adagrad(learning_rate)
            Layer_optim.append([optim_w, optim_b])

    if O_M == 'RMSprop':
        Layer_optim = []
        for _ in range(int(len(ANN)/2)):
            optim_w = RMSprop(learning_rate)
            optim_b = RMSprop(learning_rate)
            Layer_optim.append([optim_w, optim_b])

    if O_M == 'Adam':
        Layer_optim = []
        for _ in range(int(len(ANN)/2)):
            optim_w = Adam(learning_rate)
            optim_b = Adam(learning_rate)
            Layer_optim.append([optim_w, optim_b])
    
    if O_M == 'momentum':
        optim_w = 0
        optim_b = 0

    if method == 'Regg':
        cost_ = cost
        cost_grad = cost_diff
    if method == 'Class':
        cost_ = cross_entropy_cost
        cost_grad = diff_cross_entropy_cost

    cost_train = np.zeros(epochs)
    cost_test = np.zeros(epochs)

    R2_train = np.zeros(epochs)
    R2_test = np.zeros(epochs)
    
    for epoch in range(epochs):
        change_w = change_b = 0.0
        count = 0

        # Forward:
        output_test, _ = fwd_test(ANN, x_test)
        output, weights = fwd(ANN, x_train)

        # Current mse
        cost_test[epoch] = cost_(y_test, output_test, lmb, weights)
        cost_train[epoch] = cost_(y_train, output, lmb, weights)
        
        R2_train[epoch] = r2_score(y_train, output)
        R2_test[epoch] = r2_score(y_test, output_test)


        # Backward:
        grad = cost_grad(y_train, output, lmb, weights)

        for idx,layer in enumerate(reversed(ANN)):
            if O_M == 'momentum':
                grad = layer.backward(grad, learning_rate, decay, O_M, momentum, epoch, change_w, change_b, optim_w, optim_b, lmb)
                if (idx+1) % 2 == 0:
                    count += 1
            else:
                grad = layer.backward(grad, learning_rate, decay, O_M, momentum, epoch, change_w, change_b, Layer_optim[count][0], Layer_optim[count][1], lmb)
                if (idx+1) % 2 == 0:
                    count += 1
    return cost_train, cost_test, R2_train, R2_test


def train_NN_SGD(ANN, input_train, z_train, input_test, z_test, epochs, learning_rate, decay, O_M, momentum, minibatch_size, n_minibatches, seed, lmb, method):

    """ Optimizer method """
    if O_M == 'Adagrad':
        Layer_optim = []
        for _ in range(int(len(ANN)/2)):
            optim_w = Adagrad(learning_rate)
            optim_b = Adagrad(learning_rate)
            Layer_optim.append([optim_w, optim_b])

    if O_M == 'RMSprop':
        Layer_optim = []
        for _ in range(int(len(ANN)/2)):
            optim_w = RMSprop(learning_rate)
            optim_b = RMSprop(learning_rate)
            Layer_optim.append([optim_w, optim_b])

    if O_M == 'Adam':
        Layer_optim = []
        for _ in range(int(len(ANN)/2)):
            optim_w = Adam(learning_rate)
            optim_b = Adam(learning_rate)
            Layer_optim.append([optim_w, optim_b])
    
    if O_M == 'momentum':
        optim_w = 0
        optim_b = 0

    iters = epochs*n_minibatches

    cost_train = np.zeros(iters)
    cost_test = np.zeros(iters)
    R2_train = np.zeros(iters)
    R2_test = np.zeros(iters)
    i = 0

    if method == 'Regg':
        cost_ = cost
        cost_grad = cost_diff
    if method == 'Class':
        cost_ = cross_entropy_cost
        cost_grad = diff_cross_entropy_cost

    for epoch in range(epochs):

        for batch in range(n_minibatches):

            random_index = minibatch_size*np.random.randint(n_minibatches)
            input_batch = input_train[random_index:random_index+minibatch_size]
            z_batch = z_train[random_index:random_index+minibatch_size]

            # Forward:
            output_test, _ = fwd_test(ANN, input_test)
            output, weights = fwd(ANN, input_batch)

            # Current mse
            cost_train[i] = cost_(z_batch, output, lmb, weights)
            cost_test[i] = cost_(z_test, output_test, lmb, weights)
            R2_train[i] = r2_score(z_batch, output)
            R2_test[i] = r2_score(z_test, output_test)

            i += 1

            # Backward:
            grad = cost_grad(z_batch, output, lmb, weights)

            change_w = change_b = 0.0
            count = 0
            for idx,layer in enumerate(reversed(ANN)):
                if O_M == 'momentum':
                    grad = layer.backward(grad, learning_rate, decay, O_M, momentum, epoch, change_w, change_b, optim_w, optim_b, lmb)
                    if (idx+1) % 2 == 0:
                        count += 1
                else:
                    grad = layer.backward(grad, learning_rate, decay, O_M, momentum, epoch, change_w, change_b, Layer_optim[count][0], Layer_optim[count][1], lmb)
                    if (idx+1) % 2 == 0:
                        count += 1

    return cost_train, cost_test, R2_train, R2_test

