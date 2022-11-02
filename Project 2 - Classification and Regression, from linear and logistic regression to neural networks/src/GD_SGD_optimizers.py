import numpy as np
import autograd.numpy as np
import matplotlib.pyplot as plt
# from autograd import elementwise_grad as egrad
from autograd import grad
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


def FrankeFunction(x,y, noice, alpha, seed):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    if noice:
        np.random.seed(seed)
        return term1 + term2 + term3 + term4 + alpha*np.random.normal(0, 1, x.shape)
    else:
        return term1 + term2 + term3 + term4

def data_FF(noise=True, step_size=0.05, alpha=0.05):
    x = np.arange(0, 1, step_size)
    y = np.arange(0, 1, step_size)

    X, Y = np.meshgrid(x, y)
    x = X.flatten().reshape(-1, 1)
    y = Y.flatten().reshape(-1, 1)
    Z = FrankeFunction(X, Y, noise, alpha, seed=3155)
    z = Z.flatten().reshape(-1, 1)
    return x, y, z

def MSE(z, z_tilde):
    return mean_squared_error(z, z_tilde)

def r2(z, z_tilde):
    return r2_score(z, z_tilde)

def scaling(X):
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    X[:,0] = 1
    return X

def designMatrix_1D(x, polygrad):
    n = len(x)
    X = np.ones((n,polygrad))
    for i in range(1,polygrad):
        X[:,i] = (x**i).ravel()
    return X

def y_func(x, exact_theta):
    y = 0
    for i, theta in enumerate(exact_theta):
        y += theta*x**i
    return y

def CostOLS(theta, X, y):
    m = X.shape[0]
    return (1.0/m)*np.sum((y - (X @ theta))**2)

def gradient_CostOLS(theta, X, y):
    m = X.shape[0]
    grad = 2/m * X.T @ (X @ theta - y)
    return np.mean(grad, axis=1)

def auto_gradient(theta, X, y):
    #training_gradient = grad(CostOLS,0) #STIG TEST
    training_gradient = grad(CostOLS)
    return training_gradient(theta, X, y)

def learning_schedule(epoch, init_LR, decay):
    return init_LR * 1/(1 + decay*epoch)


class Optimizer:
    """A super class for three optimizers."""
    def __init__(self, eta):
        #self.gradients = gradients
        self.eta = eta
        #self.Giter = Giter #for Adagrad and RMS prop
        self.delta = 1e-7 #to avoid division by zero.

    def __call__(self,gradients):
        pass
        # self.gradients = gradients
        # self.Ginverse = np.c_[self.eta/(self.delta + np.sqrt(np.diagonal(self.Giter)))]
        # return np.multiply(self.Ginverse,self.gradients)

class Adagrad(Optimizer):
    def __call__(self,gradients,Giter):
        Giter += gradients @ gradients.T
        self.Ginverse = np.c_[self.eta/(self.delta + np.sqrt(np.diagonal(Giter)))]
        return np.multiply(self.Ginverse,gradients)

class RMSprop(Optimizer):
    def __call__(self,gradients,Giter):
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

    def __call__(self,gradients):

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
        #Note: Also return m and s for succesive iterations in SGD.
        return np.multiply(self.Ginverse,self.mHat)


def OLS(X_train, X_test, y):
    """ OLS """
    XT_X = X_train.T @ X_train
    theta_linreg = np.linalg.pinv(XT_X) @ (X_train.T @ y)
    y_predict_OLS = X_test @ theta_linreg
    return y_predict_OLS, theta_linreg

def RIDGE(x_train, scale, lambda_min, lambda_max, nlambdas):

    lambdas = np.logspace(lambda_min, lambda_max, nlambdas)

    for degree in range(order):

        D_M_train = designMatrix_1D(x_train, degree)
        D_M_test = designMatrix_1D(x_train, degree)
        if scale:
            """Scaling the design matrices based on train matrix"""
            D_M_train, D_M_test = scaling(D_M_train, D_M_test)

        for l in range(nlambdas):
            lmb = lambdas[l]

            I = np.eye(np.shape(D_M_train)[1],np.shape(D_M_train)[1])

            Ridgebeta = np.linalg.inv(D_M_train.T @ D_M_train+lmb*I) @ D_M_train.T @ z_train

            z_tilde_test = D_M_test @ Ridgebeta



def GD(X_train, X_test, y_train, y_test, method, Niterations, init_LR, decay, momentum, seed):

    """ Gradient Decent """
    np.random.seed(seed)
    theta = np.random.randn(np.shape(X_train)[1],1) # Initial thetas/betas
    change = 0.0
    mse = np.zeros(Niterations)
    # Gradient decent:
    for i in range(Niterations):

        if method == 'auto':
            gradients =  auto_gradient(theta, X_train, y_train) # Autograd
        if method == 'anal':
            gradients = gradient_CostOLS(theta, X_train, y_train) # Analytical

        eta = learning_schedule(i, init_LR, decay) # LR
        update = eta * gradients + momentum * change # Update to the thetas
        theta -= update # Updating the thetas

        y_predict_GD_test = X_test @ theta
        mse[i] = MSE(y_test, y_predict_GD_test)
        change = update # Update the amount the momentum gets added

    y_predict_GD = X_test @ theta

    return y_predict_GD, theta, mse

def SGD(X_train, X_test, y_train, y_test, Optimizer_method, Gradient_method, minibatch_size, n_minibatches, n_epochs, init_LR, decay, momentum, seed):

    """ Stochastic Gradient Decent """
    np.random.seed(seed)
    theta = np.random.randn(np.shape(X_train)[1],1) # Initial thetas/betas
    change = 0.0

    np.random.seed(seed)
    random_index = minibatch_size*np.random.randint(n_minibatches)
    X_batch = X_train[random_index:random_index + minibatch_size]
    y_batch = y_train[random_index,:random_index + minibatch_size]

    mse = np.zeros(n_epochs*n_minibatches)
    count = 0
    #i = 0
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

            """ Gradient method """
            if Gradient_method == 'auto':
                gradients =  auto_gradient(theta, X_batch, y_batch) # Autograd
            if Gradient_method == 'anal':
                gradients = gradient_CostOLS(theta, X_batch, y_batch) # Analytical

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

            y_predict_GD_test = X_test @ theta
            mse[count] = MSE(y_test, y_predict_GD_test)
            count += 1

    y_predict_SGD = X_test @ theta
    return y_predict_SGD, theta, mse