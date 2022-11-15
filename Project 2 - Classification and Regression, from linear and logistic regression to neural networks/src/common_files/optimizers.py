"""Optimizers for SGD descent methods"""


import numpy as np
import autograd.numpy as np
from autograd import grad

class Optimizer:
    """A super class for three optimizers."""
    def __init__(self, eta):
        self.eta = eta
        self.delta = 1e-7 #to avoid division by zero.

    def __call__(self,gradients):
        raise TypeError("You need to specify which Optimizer")

class Adagrad(Optimizer):
    def __call__(self,gradients, Giter):
        Giter += gradients @ gradients.T
        self.Ginverse = np.c_[self.eta/(self.delta + np.sqrt(np.diagonal(Giter)))]
        return np.multiply(self.Ginverse,gradients)

class RMSprop(Optimizer):
    def __call__(self,gradients, Giter):
        beta = 0.90 #Ref Geron boka.
        Previous = Giter.copy() #stores the current Giter.
        Giter += gradients @ gradients.T
        Giter = (beta*Previous + (1 - beta)*Giter)
        self.Ginverse = np.c_[self.eta/(self.delta + np.sqrt(np.diagonal(Giter)))]
        return np.multiply(self.Ginverse,gradients)

class Adam(Optimizer):
    """https://towardsdatascience.com/how-to-implement-an-adam-optimizer-from-scratch-76e7b217f1cc rand
    Algoritm 8.7 Adam in Chapter 8 of Ian Goodfellow"""
    def __init__(self,eta):
        super().__init__(eta) # Optimizer stores these.
        self.m = 0
        self.s = 0
        self.t = 1
        self.beta_1 = 0.90 #Ref Geron and Goodfellow bøkene.
        self.beta_2 = 0.999 #Ref Geron and Goodfellow bøkene.

    def __call__(self, gradients):

        #Update of 1st and 2nd moment:
        m = (self.beta_1*self.m + (1 - self.beta_1)*gradients)
        s = (self.beta_2*self.s + (1 - self.beta_2)*gradients**2)

        #Bias correction:
        self.mHat = m/(1 - self.beta_1**self.t) #med tidsteg t.
        self.sHat = s/(1 - self.beta_2**self.t)

        #Compute update:
        self.Ginverse = self.eta/(self.delta + np.sqrt(self.sHat))
        self.m = m
        self.s = s
        self.t += 1
        
        return np.multiply(self.Ginverse,self.mHat)
