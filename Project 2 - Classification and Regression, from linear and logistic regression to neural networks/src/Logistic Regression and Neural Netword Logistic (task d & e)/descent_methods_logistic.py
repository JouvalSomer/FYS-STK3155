"""Descent Gradient Methods for the Logistic regression
major update 15.11.2022"""

import numpy as np
import autograd.numpy as np
from autograd import grad

from func import*
from optimizers import *

#
# def GD(X_train, X_test, y_train, y_test, method, Niterations, init_LR, decay, momentum, seed):
#
#     """ Gradient Decent """
#     np.random.seed(seed)
#     theta = np.random.randn(np.shape(X_train)[1],1) # Initial thetas/betas
#     change = 0.0
#     mse = np.zeros(Niterations)
#     # Gradient decent:
#     for i in range(Niterations):
#
#         if method == 'auto':
#             gradients =  auto_gradient(theta, X_train, y_train) # Autograd
#         if method == 'anal':
#             gradients = gradient_CostOLS(theta, X_train, y_train) # Analytical
#
#         eta = learning_schedule(i, init_LR, decay) # LR
#         update = eta * gradients + momentum * change # Update to the thetas
#         theta -= update # Updating the thetas
#
#         y_predict_GD_test = X_test @ theta
#         mse[i] = MSE(y_test, y_predict_GD_test)
#         change = update # Update the amount the momentum gets added
#
#     y_predict_GD = X_test @ theta
#
#     return y_predict_GD, theta, mse

def SGD(X_train, X_test, y_train, y_test, Optimizer_method, Gradient_method, minibatch_size, n_minibatches, n_epochs, init_LR, decay, momentum, seed, lmb):

    """ Stochastic Gradient Decent """
    np.random.seed(seed)
    theta = np.random.randn(X_train.shape[1], 1) # Initial thetas/betas
    change = 0.0
    """List of mini batches:"""
    X_Batcharray = [] #empty list of batch array
    Y_Batcharray = [] #empty list of batch array
    for bat in range(n_minibatches):
        random_index = minibatch_size*np.random.randint(n_minibatches)
        #print(n_minibatches)
        X_Batcharray.append(X_train[random_index:random_index + minibatch_size])
        Y_Batcharray.append(y_train[random_index:random_index + minibatch_size])

    cost = np.zeros(n_epochs*n_minibatches) #Logistic regression cost.
    count = 0
    eta = init_LR

    """ Optimizer method """
    if Optimizer_method == 'Adagrad':
        optim = Adagrad(eta)
    if Optimizer_method == 'RMSprop':
        optim = RMSprop(eta)
    if Optimizer_method == 'Adam':
        optim = Adam(eta)

    for epoch in range(n_epochs):
        Giter = np.zeros(shape=(X_train.shape[1],X_train.shape[1]))

        for batch in range(n_minibatches):
            X_batch = X_Batcharray[batch]
            y_batch = Y_Batcharray[batch]

            """ Gradient method for logistic"""
            if Gradient_method == 'log':
                gradients = grad_cost_func_logregression(theta, X_batch, y_batch, lmb)

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
                #gradients = grad_cost_func_logreg(theta, X_batch, y_batch)

                eta = learning_schedule(epoch, init_LR, decay) # LR
                update = eta * gradients + momentum * change # Update to the thetas
                # if count == 0:
                #     print('\nThetas for SGD:')
                # if count % 200 == 0:
                #     print(theta.flatten(), f'iter = {count}')
                theta -= update
                change = update

            y_predict_GD_test = 1/(1+np.exp(-X_test @ theta))

            cost[count] = cross_entropy_cost(y_test, y_predict_GD_test, lmb, theta)

            count +=1

    """logistic classification on test:"""
    Prob = 1/(1+np.exp(- X_test @ theta))
    y_pred = Prob.copy()
    tol = 0.50
    y_pred[y_pred >= tol] = 1
    y_pred[y_pred < tol] = 0
    acc_score = accuracy(y_pred,y_test)

    return y_pred, Prob, theta, cost, acc_score
