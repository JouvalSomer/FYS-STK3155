import numpy as np
import autograd.numpy as np
import matplotlib.pyplot as plt
# from autograd import elementwise_grad as egrad
from autograd import grad
from sklearn.preprocessing import StandardScaler

def CostOLS(theta, X, y):
    m = X.shape[0]
    return (1.0/m)*np.sum((y - (X @ theta))**2)

def gradient_CostOLS(theta, X, y):
    m = X.shape[0]
    grad = 2/m * X.T @ (X @ theta - y)
    return np.mean(grad, axis=1)

def auto_gradient(theta, X, y):
    training_gradient = grad(CostOLS)
    return training_gradient(theta, X, y)

def learning_schedule(iter, init_LR, decay):
    return init_LR * 1/(1 + decay*iter)

def scaling(X):
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    X[:,0] = 1
    return X

def GD(method, Niterations, iterprint, init_LR, decay, plot):

    """ Design matrix and OLS """
    X = np.c_[np.ones((n,1)), x, x**2]
    XT_X = X.T @ X
    theta_linreg = np.linalg.pinv(XT_X) @ (X.T @ y)
    print("OLS betas:")
    print(theta_linreg, '\n')

    """ Scaling """
    # X = scaling(X)

    """ Gradient decent """
    theta = np.random.randn(3,1) # Initial thetas/betas

    # Gradient decent:
    for iter in range(Niterations):

        if method == 'auto':
            gradients =  auto_gradient(theta, X, y) # Autograd
        if method == 'anal':
            gradients = gradient_CostOLS(theta, X, y) # Analytical 

        eta = learning_schedule(iter, init_LR, decay)

        if iter % iterprint == 0: # Print theta and gradient every 50th iteration
            print('iter = ', iter)
            print('theta = \n', theta)
            print('gradients = \n', gradients, '\n')
            print('eta = ', eta)

        theta -= eta*gradients

    y_predict_OLS = X @ theta_linreg 
    y_predict_GD = X @ theta

    if plot:
        plt.plot(x, y_predict_OLS, "b-", label='OLS')
        plt.plot(x, y_predict_GD, "r-", label='GD') 
        plt.plot(x, y ,'g.', label='Data') # Data
        plt.xlabel(r'$x$')
        plt.ylabel(r'$y$')
        plt.legend()
        plt.show()


if __name__=='__main__':
    """ Data """
    n = 15
    x = np.linspace(0, 1, n)
    y = 3 + 2*x + 3*x**2
    alpha = 0.1 # Noise scaling
    np.random.seed(3155)
    y = (y + alpha*np.random.normal(0, 1, x.shape)).reshape(-1, 1)

    
    init_LR = 0.01 # Initial learning rate
    decay = 0.001 # Learning rate decay rate

    """ Gradient method """
    method = 'auto'
    # method = 'anal'

    plot = True
    Niterations = 400 # Number of GD iterations
    iterprint = 10 # Print theta and gradients every iterprint-th iteration
    GD(method, Niterations, iterprint, init_LR, decay, plot)