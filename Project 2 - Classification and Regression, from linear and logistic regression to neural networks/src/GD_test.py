import numpy as np
import autograd.numpy as np
import matplotlib.pyplot as plt
# from autograd import elementwise_grad as egrad
from autograd import grad
from sklearn.preprocessing import StandardScaler

def CostOLS(beta, X, y):
    m = X.shape[0]
    return (1.0/m)*np.sum((y - (X @ beta))**2)

def gradient_CostOLS(X, y, beta):
    m = X.shape[0]
    grad = 2/m * X.T @ (X @ beta - y)
    return np.mean(grad, axis=1)

def scaling(X):
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    return X

def GD(method, Niterations, iterprint, plot):
    """ Design matrix and OLS """
    X = np.c_[np.ones((n,1)), x, x**2]
    XT_X = X.T @ X
    theta_linreg = np.linalg.pinv(XT_X) @ (X.T @ y)
    print("OLS betas:")
    print(theta_linreg, '\n')

    """ Scaling """
    # X = scaling(X)
    # X[:,0] = 1

    """ Gradient decent """
    theta = np.random.randn(3,1) # Initial thetas/betas
    eta = 0.1 # Learning rate

    # Define the gradient:
    if method == 'auto':
        training_gradient = grad(CostOLS) # Autograd of the cost function 

    # Gradient decent:
    for iter in range(Niterations):
        if method == 'auto':
            gradients = training_gradient(theta, X, y) # Autograd
        else:
            gradients = gradient_CostOLS(X, y, theta).reshape(-1, 1)
        if iter % iterprint == 0: # Print theta and gradient every 50th iteration
            print('iter = ', iter)
            print('theta = \n', theta)
            print('gradients = \n', gradients, '\n')
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
    x = np.linspace(0, 2, n)
    y = 3 + 2*x + 3*x**2
    alpha = 0.1 # Noise scaling
    np.random.seed(3155)
    y = y + alpha*np.random.normal(0, 1, x.shape)

    """ Gradient method """
    method = 'auto'
    # method = 'manually'

    plot = True
    Niterations = 100 # Number of GD iterations
    iterprint = 10 # Print theta and gradients every iterprint-th iteration
    GD(method, Niterations, iterprint, plot)