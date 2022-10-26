import numpy as np
import autograd.numpy as np
import matplotlib.pyplot as plt
# from autograd import elementwise_grad as egrad
from autograd import grad
from sklearn.preprocessing import StandardScaler

def scaling(X):
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    X[:,0] = 1
    return X

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

def learning_schedule(epoch, init_LR, decay):
    return init_LR * 1/(1 + decay*epoch)

def Adagrad(gradients, eta, Giter):
    delta = 1e-8
    Giter += gradients @ gradients.T
    Ginverse = np.c_[eta/(delta + np.sqrt(np.diagonal(Giter)))]
    return np.multiply(Ginverse,gradients)

def RMSprop(gradients, eta, Giter):
    beta = 0.99
    delta = 1e-8
    Previous = Giter
    Giter += gradients @ gradients.T
    Gnew = (beta*Previous + (1 - beta)*Giter)
    Ginverse = np.c_[eta/(np.sqrt(np.diagonal(Gnew) + delta))]
    return np.multiply(Ginverse,gradients)

def Adam(gradients, eta, Giter):

    beta_1 = 0.9
    beta_2 = 0.99
    delta = 1e-8

    Previous = Giter
    
    Giter += gradients @ gradients.T

    m = (beta_1*m_priv + (1 - beta_1)*gradients)
    s = (beta_2*s_priv + (1 - beta_2)*Giter)

    m = m/(1 - beta_1)
    s = s/(1 - beta_2)

    Ginverse = np.c_[eta/(np.sqrt(np.diagonal(s)) + delta)]

    return np.multiply(Ginverse,m)

    # m = np.zeros(n) # two m for each parameter
    # v = np.zeros(n) # two v for each parameter
    # g = np.zeros(n) # two gradient
    
    # for t in range(1,max_iteration):

    #     # Update the m and v parameter
    #     m = [b1*m_i + (1 - b1)*g_i for m_i, g_i in zip(m, g)]
    #     v = [b2*v_i + (1 - b2)*(g_i**2) for v_i, g_i in zip(v, g)]

    #     # Bias correction for m and v
    #     m_cor = [m_i / (1 - (b1**t)) for m_i in m]
    #     v_cor = [v_i / (1 - (b2**t)) for v_i in v]

    #     # Update the parameter
    #     model.weights = [weight - (learning_rate / (np.sqrt(v_cor_i) + epsilon))*m_cor_i for weight, v_cor_i, m_cor_i in zip(model.weights, v_cor, m_cor)]
        
    return 

def OLS(X, y):
    """ OLS """
    XT_X = X.T @ X
    theta_linreg = np.linalg.pinv(XT_X) @ (X.T @ y)
    print("OLS thetas:")
    print(theta_linreg, '\n')
    y_predict_OLS = X @ theta_linreg 
    return y_predict_OLS
        
def GD(X, y, method, Niterations, init_LR, decay, momentum, seed):

    """ Gradient Decent """
    np.random.seed(seed)
    theta = np.random.randn(np.shape(X)[1],1) # Initial thetas/betas
    change = 0.0
    # Gradient decent:
    for i in range(Niterations):

        if method == 'auto':
            gradients =  auto_gradient(theta, X, y) # Autograd
        if method == 'anal':
            gradients = gradient_CostOLS(theta, X, y) # Analytical 

        eta = learning_schedule(i, init_LR, decay) # LR
        update = eta * gradients + momentum * change # Update to the thetas
        theta -= update # Updating the thetas
        change = update # Update the amount the momentum gets added 

    print('GD thetas:')
    print(theta, '\n')

    y_predict_GD = X @ theta
    return y_predict_GD

def SGD(X, y, Optimizer_method, Gradient_method, minibatch_size, n_epochs, init_LR, decay, momentum, seed):

    """ Stochastic Gradient Decent """
    np.random.seed(seed)
    theta = np.random.randn(np.shape(X)[1],1) # Initial thetas/betas
    change = 0.0

    n_minibatches = int(n/minibatch_size) #number of minibatches

    np.random.seed(seed)
    random_index = minibatch_size*np.random.randint(n_minibatches)
    X_batch = X[random_index:random_index + minibatch_size]
    y_batch = y[random_index:random_index + minibatch_size]

    for epoch in range(n_epochs):
        Giter = np.zeros(shape=(3,3))
        for batch in range(n_minibatches):

            """ Gradient method """
            if Gradient_method == 'auto':
                gradients =  auto_gradient(theta, X_batch, y_batch) # Autograd
            if Gradient_method == 'anal':
                gradients = gradient_CostOLS(theta, X_batch, y_batch) # Analytical 


            """ Optimizer method """
            if Optimizer_method == 'Adagrad':
                eta = init_LR
                update = Adagrad(gradients, eta, Giter)
                theta -= update

            if Optimizer_method == 'RMSprop':
                eta = init_LR
                update = RMSprop(gradients, eta, Giter)
                theta -= update

            if Optimizer_method == 'momentum':
                eta = learning_schedule(epoch, init_LR, decay) # LR
                update = eta * gradients + momentum * change # Update to the thetas            
                theta -= update
                change = update

    print('SGD thetas:')
    print(theta, '\n')

    y_predict_SGD = X @ theta
    return y_predict_SGD


if __name__=='__main__':
    """ Data """
    n = 100
    x = np.linspace(0, 1, n)
    y = 3 + 2*x + 3*x**2
    exact_theta = [3.0, 2.0, 3.0]
    print('Exact theta:\n', np.array(exact_theta).reshape(-1, 1), '\n')
    alpha = 0.5 # Noise scaling
    seed = 3155
    np.random.seed(seed)
    y = (y + alpha*np.random.normal(0, 1, x.shape)).reshape(-1, 1)

    # Design matrix
    X = np.c_[np.ones((n,1)), x, x**2]

    n_epochs = 20 #number of epochs
    init_LR = 0.01 # Initial learning rate (LR)
    decay = init_LR/n_epochs # LR decay rate (for fixed LR set decay to 0)
    momentum = 0.9 # Momentum value
    minibatch_size = 10


    """ Gradient method """
    Gradient_method = ['auto', 'anal']

    # If you want plain GD without any optimization choose 'momentum' with momentum value of 0
    Optimizer_method = ['Adagrad', 'RMSprop', 'momentum']

    Niterations = 400 # Number of GD iterations
    y_predict_GD = GD(X, y, Gradient_method[0], Niterations, init_LR, decay, momentum, seed)
    y_predict_SGD = SGD(X, y, Optimizer_method[2], Gradient_method[0], minibatch_size, n_epochs, init_LR, decay, momentum, seed)
    y_predict_OLS = OLS(X, y)

    plt.plot(x, y_predict_GD, "r-", label='GD') 
    plt.plot(x, y_predict_SGD, "b-", label='SGD') 
    plt.plot(x, y_predict_OLS, "k-", zorder=100, label='OLS') 
    plt.plot(x, y ,'g.', label='Data') # Data
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    plt.legend()
    plt.show()