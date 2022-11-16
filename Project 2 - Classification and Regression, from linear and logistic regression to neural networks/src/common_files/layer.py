"""Feed Forward and Backwards Propagation Class"""

import numpy as np
from func import *
from optimizers import*

class Layer:
    def __init__(self, input_size, output_size):
        self.input = None
        self.output = None

        init = np.sqrt(6)/np.sqrt(input_size + output_size)
        self.weights = np.random.uniform(low=-init, high=init, size=(input_size, output_size))
        # self.weights = np.random.randn(input_size, output_size)

        self.bias = np.ones((1, output_size)) * 0.01

    def forward(self, input):
        self.input = input
    
        fwd = (self.input @ self.weights)
        fwd = fwd + self.bias

        return fwd, self.weights

    def backward(self, output_gradient, learning_rate, decay, O_M, momentum, epoch, change_w, change_b, optim_w, optim_b, lmb):

        """ Optimizer method """
        if O_M == 'Adam':
            weights_gradient = self.input.T @ output_gradient + self.weights*lmb 
            bias_gradient = np.mean(output_gradient)

            update_w = optim_w(weights_gradient) 
            update_b = optim_b(bias_gradient)

            self.weights -= update_w
            self.bias -= update_b

        if O_M == 'momentum':
            eta = learning_schedule(epoch, learning_rate, decay)
            
            weights_gradient = self.input.T @ output_gradient 
            bias_gradient = np.mean(output_gradient)

            update_w = eta * weights_gradient + momentum * change_w
            update_b = eta * bias_gradient + momentum * change_b

            self.weights -= update_w
            self.bias -= update_b

            change_w = update_w
            change_b = update_b
            
        input_gradient = (self.weights @ output_gradient.T).T
        return input_gradient
