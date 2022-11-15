"""Activation methods

Note from Week 40: The sigmoid function are more biologically plausible
because the output of inactive neurons are zero.
Such activation function are called one-sided.
However, it has been shown that the hyperbolic tangent
performs better than the sigmoid for training MLPs and has become the
most popular for deep neural networks"""

import numpy as np

class Activation:
    def __init__(self, activation, activation_diff):
        self.input = None
        self.output = None

        self.activation = activation
        self.activation_diff = activation_diff

    def forward(self, input):
        self.input = input
        return self.activation(self.input), 0

    def backward(self, output_gradient, learning_rate, decay, O_M, momentum, epoch, change_w, change_b, optim_w, optim_b, lmb):
        return np.multiply(output_gradient, self.activation_diff(self.input))


class Hyperbolic(Activation):
    def __init__(self):
        def tanh(x):
            return np.tanh(x)

        def tanh_diff(x):
            return 1 - np.tanh(x) ** 2

        super().__init__(tanh, tanh_diff)

class ReLU(Activation):
    def __init__(self):
        def relu(x):
            #return np.max(0,x) funker ikke pga x er array....
            return x*(x>=0) #triks.

        def relu_diff(x):
            #derivate not defined at zero so we will sneak in:
            return 1*(x>=0)

        super().__init__(relu, relu_diff)

class LeakyReLU(Activation):
    def __init__(self):
        alpha = 0.01 #leak hyperparameter.
        def leakyrelu(x):
            return x*(x>=0) + alpha*x*(x<0)

        def leakyrelu_diff(x):
            return 1*(x>=0) + alpha*(x<0)
        super().__init__(leakyrelu, leakyrelu_diff)

class Sigmoid(Activation):
    def __init__(self):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        def sigmoid_diff(x):
            s = sigmoid(x)
            return s * (1 - s)

        super().__init__(sigmoid, sigmoid_diff)

class ELU(Activation):
    def __init__(self):
        alpha = 1 #leak hyperparameter.

        def elu(x):
            return alpha*(np.exp(x) - 1)*(x<=0) + x*(x>0)

        def elu_diff(x):
            return alpha*np.exp(x)*(x<0) + 1*(x>0)

        super().__init__(elu, elu_diff)

class Linear_Activation(Activation):
    def __init__(self):

        def NA(x):
            return x

        def NA_diff(x):
            return 1

        super().__init__(NA, NA_diff)

class Sin(Activation):
    def __init__(self):

        def sin(x):
            return np.sin(x)

        def sin_diff(x):
            return np.cos(x)

        super().__init__(sin, sin_diff)
