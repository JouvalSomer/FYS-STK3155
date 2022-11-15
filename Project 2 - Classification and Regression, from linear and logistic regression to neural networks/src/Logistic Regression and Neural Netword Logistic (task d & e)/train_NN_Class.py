""" The neural FFNN for Classification
updated 15.11.2022: Cost functions and gradient."""

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

def train_NN_GD(ANN, x_train, y_train, epochs, learning_rate, decay, O_M, momentum, lmb):

    """ Optimizer method """
    if O_M == 'Adagrad':
        Layer_optim = []
        # Layer_Giter = []
        for _ in range(int(len(ANN)/2)):
            optim_w = Adagrad(learning_rate)
            optim_b = Adagrad(learning_rate)
            Layer_optim.append([optim_w, optim_b])

            # Giter_w = np.zeros(shape=(,))
            # Giter_b = np.zeros(shape=(,))
            # Layer_Giter.append([Giter_w, Giter_b])

    if O_M == 'RMSprop':
        Layer_optim = []
        # Layer_Giter = []
        for _ in range(int(len(ANN)/2)):
            optim_w = RMSprop(learning_rate)
            optim_b = RMSprop(learning_rate)
            Layer_optim.append([optim_w, optim_b])

            # Giter_w = np.zeros(shape=(,))
            # Giter_b = np.zeros(shape=(,))
            # Layer_Giter.append([Giter_w, Giter_b])

    if O_M == 'Adam':
        Layer_optim = []
        for _ in range(int(len(ANN)/2)):
            optim_w = Adam(learning_rate)
            optim_b = Adam(learning_rate)
            Layer_optim.append([optim_w, optim_b])

    if O_M == 'momentum':
        optim_w = 0
        optim_b = 0


    mse = np.zeros(epochs)
    for epoch in range(epochs):
        change_w = change_b = 0.0
        count = 0
        # Forward:
        output, weights = fwd(ANN, x_train)
        # Current COST:
        mse[epoch] = cross_entropy_cost(y_train, output, lmb, weights)

        # Backward:
        grad = diff_cross_entropy_cost(y_train, output, lmb, weights)

        for idx,layer in enumerate(reversed(ANN)):
            grad = layer.backward(grad, learning_rate, decay, O_M, momentum, epoch, change_w, change_b, Layer_optim[count][0], Layer_optim[count][1], lmb)

            if (idx+1) % 2 == 0:
                count += 1
    return mse


def train_NN_SGD(ANN, X_train, y_train, epochs, learning_rate, decay, O_M, momentum, minibatch_size, n_minibatches, seed, lmb):

    """ Optimizer method """
    if O_M == 'Adagrad':
        Layer_optim = []
        # Layer_Giter = []
        for _ in range(int(len(ANN)/2)):
            optim_w = Adagrad(learning_rate)
            optim_b = Adagrad(learning_rate)
            Layer_optim.append([optim_w, optim_b])

            # Giter_w = np.zeros(shape=(,))
            # Giter_b = np.zeros(shape=(,))
            # Layer_Giter.append([Giter_w, Giter_b])

    if O_M == 'RMSprop':
        Layer_optim = []
        # Layer_Giter = []
        for _ in range(int(len(ANN)/2)):
            optim_w = RMSprop(learning_rate)
            optim_b = RMSprop(learning_rate)
            Layer_optim.append([optim_w, optim_b])

            # Giter_w = np.zeros(shape=(,))
            # Giter_b = np.zeros(shape=(,))
            # Layer_Giter.append([Giter_w, Giter_b])

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
    mse = np.zeros(iters)
    i = 0

    # np.random.seed(seed)
    # random_index = np.random.randint(n_minibatches) # minibatch_size*np.random.randint(n_minibatches)
    # X_batch = x_train[random_index:random_index + minibatch_size]
    # y_batch = y_train[random_index:random_index + minibatch_size]


    """ Stochastic Gradient Decent """
    np.random.seed(seed)
    #theta = np.random.randn(X_train.shape[1], 1) # Initial thetas/betas
    #change = 0.0
    """List of mini batches:"""
    X_Batcharray = [] #empty list of batch array
    Y_Batcharray = [] #empty list of batch array

    for bat in range(n_minibatches):
        random_index = minibatch_size*np.random.randint(n_minibatches)
        #print(n_minibatches)
        X_Batcharray.append(X_train[random_index:random_index + minibatch_size])
        Y_Batcharray.append(y_train[random_index:random_index + minibatch_size])


    for epoch in range(epochs):
        for batch in range(n_minibatches):

            X_batch = X_Batcharray[batch]
            y_batch = Y_Batcharray[batch]

            # Forward:
            output, weights = fwd(ANN, X_batch)

            # Current cost:
            mse[i] = cross_entropy_cost(y_batch, output, lmb, weights)
            i += 1

            # Backward:
            grad = diff_cross_entropy_cost(y_batch, output, lmb, weights)

            change_w = change_b = 0.0
            count = 0
            for idx,layer in enumerate(reversed(ANN)):
                grad = layer.backward(grad, learning_rate, decay, O_M, momentum, epoch, change_w, change_b, Layer_optim[count][0], Layer_optim[count][1], lmb)
                if (idx+1) % 2 == 0:
                    count += 1
    return mse
