"""General functions used"""

import numpy as np
import autograd.numpy as np
from autograd import grad
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

""" Data """
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

def data_FF(noise=True, step_size=0.05, alpha=0.05, reshape=True):
    x = np.arange(0, 1, step_size)
    y = np.arange(0, 1, step_size)

    X, Y = np.meshgrid(x, y)
    if reshape:
        x = X.flatten().reshape(-1, 1)
        y = Y.flatten().reshape(-1, 1)
        Z = FrankeFunction(X, Y, noise, alpha, seed=3155)
        z = Z.flatten().reshape(-1, 1)
        return x, y, z
    if not reshape:
        Z = FrankeFunction(X, Y, noise, alpha, seed=3155)
        return X, Y, Z


def y_func(x, exact_theta):
    y = 0
    for i, theta in enumerate(exact_theta):
        y += theta*x**i
    return y

def polynomial(coeff, n, noise, alpha, seed):
    X = np.linspace(0, 1, n).reshape(-1, 1)    
    Y_true = y_func(X, coeff)
    if noise:
        np.random.seed(seed)
        Y_noise = (Y_true + alpha*np.random.normal(0, 1, X.shape))
    else:
        Y_noise = Y_true
    return X, Y_noise, Y_true

def xor():
    X = np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4, 2))
    Y = np.reshape([[0], [1], [1], [0]], (4, 1))
    return X, Y



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

def learning_schedule(epoch, init_LR, decay):
    return init_LR * 1/(1 + decay*epoch)



""" Task a) """
def OLS(X_train, X_test, y):
    """ OLS """
    XT_X = X_train.T @ X_train
    theta_linreg = np.linalg.pinv(XT_X) @ (X_train.T @ y)
    y_predict_OLS = X_test @ theta_linreg
    return y_predict_OLS, theta_linreg

def cost_theta(theta, X, y, lmb):
    n = X.shape[0]
    return (1.0/n)*np.sum((y - (X @ theta))**2) + (lmb/2) * (theta.T@theta)

def cost_theta_diff(theta, X, y, lmb):
    n = X.shape[0]
    return lmb*theta - 2*X.T @(y - X@theta)/n


def auto_gradient(theta, X, y, lmb):
    print(theta.shape, X.shape, y.shape, lmb)
    y.shape += (1,)
    print(theta.shape, X.shape, y.shape, lmb)
    gradient = grad(cost_theta)
    return gradient(theta, X, y, lmb)


""" NN """
def cost_w_b(w, b, X, y, lmb):
    n = X.shape[0]
    return (1.0/n)*np.sum((y - (X @ w + b))**2) + (lmb/2)*(w.T@w)

def cost_w_b_diff(w, b, X, y, lmb):
    n = X.shape[0]
    grad_w = lmb*w + np.sum(2*X*(b + w*X - y))/n
    grad_b = (2/n)*np.sum(b + w*X - y)
    return grad_w, grad_b

def cost(z, z_tilde, lmb, w):
    return mean_squared_error(z, z_tilde) + (lmb/2)*(w.T@w)

def cost_diff(z, z_tilde, lmb, w):
    return 2 * (z_tilde - z) / np.size(z_tilde) # + lmb*w

def MSE(z, z_tilde):
    return mean_squared_error(z, z_tilde)

def MSE_diff(z, z_tilde):
    return 2 * (z_tilde - z) / np.size(z_tilde)

def auto_gradient_NN(w, b, X, y, lmb):
    w_gradient = grad(cost_w_b,0)
    b_gradient = grad(cost_w_b,1)
    return w_gradient(w, b, X, y, lmb), b_gradient(w, b, X, y, lmb)

def r2(z, z_tilde):
    return r2_score(z, z_tilde)

# def Cost_GD(w, b, X, y, lmb):
#     n = X.shape[0]
#     return (1.0/n)*np.sum((y - (X @ w + b))**2) + (lmb/2) * w**2

# def Cost_GD_diff(w, b, X, y, lmb):
#     n = X.shape[0]
#     grad_w = lmb*w + (2*X*(b + w*X - y))/n # (2/n)*X*(b + w*X - y) # 2.0/n * X.T @ ((X @ w + b) - y)
#     grad_b = (2/n)*(b + w*X - y)
#     return np.mean(grad_w, axis=1), np.mean(grad_b, axis=1)


""" Logistic Regression """

def accuracy(y_pred,y):
    """Logistic:"""
    accuracy = np.mean(y_pred.flatten()==y.flatten())
    return accuracy

def cross_entropy_cost(y_train, output, lmb, weights):
    """the binary cross entropy cost function
    lmb is L2 reg. w is for the regularization of NN"""
    eps =1e-15 # small epsilon for numeric stability.
    p = output #prediction.
    y = y_train
    w= weights
    cost = - np.sum(y*np.log(p+eps) + (1-y)*np.log(1-p+eps))+ (lmb/2)* w.T @ w
    cost = cost/y.shape[0]
    return cost

def diff_cross_entropy_cost(y_train, output, lmb, weights):
    """The derivative of the cost function for cross entropy NN classifier.
    Note that L2 regularisation happens in the layer.py backprop"""
    p = output
    y = y_train
    w = weights

    grad_cross_entropi =  (p - y) #Simplified for first layer backwards.
    mean = np.mean(grad_cross_entropi,axis =1)
    gradient_CE = np.expand_dims(mean, axis=1)
    return gradient_CE

def grad_cost_func_logregression(beta, X, y, lmb):
    """The derivative of the cost function for logistic regression"""
    p = 1/(1+np.exp(-X@beta))
    grad_cross_entropi = - X.T @ (y - p) +lmb*beta
    mean = np.mean(grad_cross_entropi,axis =1)
    gradient_CE = np.expand_dims(mean, axis=1)
    return gradient_CE




