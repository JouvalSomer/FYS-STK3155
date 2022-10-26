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

def designMatrix(x, polygrad):
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
    return y_predict_OLS, theta_linreg

def RIDGE(x_train, scale, lambda_min, lambda_max, nlambdas):
        
    lambdas = np.logspace(lambda_min, lambda_max, nlambdas)

    for degree in range(order):

        D_M_train = designMatrix(x_train, degree)
        D_M_test = designMatrix(x_train, degree)
        if scale:
            """Scaling the design matrices based on train matrix"""
            D_M_train, D_M_test = scaling(D_M_train, D_M_test)

        for l in range(nlambdas):
            lmb = lambdas[l]

            I = np.eye(np.shape(D_M_train)[1],np.shape(D_M_train)[1])

            Ridgebeta = np.linalg.inv(D_M_train.T @ D_M_train+lmb*I) @ D_M_train.T @ z_train

            z_tilde_test = D_M_test @ Ridgebeta
       
def GD(X, y, method, Niterations, init_LR, decay, momentum, seed):

    """ Gradient Decent """
    np.random.seed(seed)
    theta = np.random.randn(np.shape(X)[1],1) # Initial thetas/betas
    change = 0.0
    mse = np.zeros(Niterations)
    # Gradient decent:
    for i in range(Niterations):

        if method == 'auto':
            gradients =  auto_gradient(theta, X, y) # Autograd
        if method == 'anal':
            gradients = gradient_CostOLS(theta, X, y) # Analytical 

        eta = learning_schedule(i, init_LR, decay) # LR
        update = eta * gradients + momentum * change # Update to the thetas
        theta -= update # Updating the thetas
        
        y_predict_GD_test = X_test @ theta
        mse[i] = MSE(y_test, y_predict_GD_test)
        change = update # Update the amount the momentum gets added 

    print('GD thetas:')
    print(theta, '\n')

    y_predict_GD = X @ theta
    return y_predict_GD, theta, mse

def SGD(X, y, Optimizer_method, Gradient_method, minibatch_size, n_minibatches, n_epochs, init_LR, decay, momentum, seed):

    """ Stochastic Gradient Decent """
    np.random.seed(seed)
    theta = np.random.randn(np.shape(X)[1],1) # Initial thetas/betas
    change = 0.0


    np.random.seed(seed)
    random_index = minibatch_size*np.random.randint(n_minibatches)
    X_batch = X[random_index:random_index + minibatch_size]
    y_batch = y[random_index:random_index + minibatch_size]
    
    mse = np.zeros(n_epochs*n_minibatches)
    count = 0
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

            y_predict_GD_test = X_test @ theta
            mse[count] = MSE(y_test, y_predict_GD_test)
            count += 1

    print('SGD thetas:')
    print(theta, '\n')

    y_predict_SGD = X @ theta
    return y_predict_SGD, theta, mse


if __name__=='__main__':

    """ Data """
    n = 400
    x = np.linspace(0, 1, n)
    exact_theta = [3.0, 2.0, 3.0]
    print('Exact theta:\n', np.array(exact_theta).reshape(-1, 1), '\n')
    alpha = 0.5 # Noise scaling
    seed = 3155
    y = y_func(x, exact_theta)
    np.random.seed(seed)
    y = (y + alpha*np.random.normal(0, 1, x.shape)).reshape(-1, 1)



    """ Train Test Split """
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state = seed)
    polygrad = len(exact_theta)
    X_train = designMatrix(x_train, polygrad)
    X_test = designMatrix(x_test, polygrad)


    """ Hyperparameters """
    n_epochs = 40 #number of epochs
    init_LR = 0.01 # Initial learning rate (LR)
    decay = init_LR/n_epochs # LR decay rate (for fixed LR set decay to 0)
    momentum = 0.9 # Momentum value
    minibatch_size = 40
    n_minibatches = int(np.shape(X_train)[0]/minibatch_size) #number of minibatches
    Niterations = 400 # Number of GD iterations

    """ Gradient method """
    Gradient_method = ['auto', 'anal']

    """ Optimization method """
    # If you want plain GD without any optimization choose 'momentum' with momentum value of 0
    Optimizer_method = ['Adagrad', 'RMSprop', 'momentum']



    """ Results """
    y_predict_GD, theta_GD, mse_GD = GD(X_train, y_train, Gradient_method[0], Niterations, init_LR, decay, momentum, seed)
    y_predict_SGD, theta_SGD, mse_SGD = SGD(X_train, y_train, Optimizer_method[2], Gradient_method[0], minibatch_size, n_minibatches, n_epochs, init_LR, decay, momentum, seed)
    y_predict_OLS, theta_OLS = OLS(X_train, y_train)

    y_predict_GD_test = X_test @ theta_GD
    print(f'MSE_GD  = {MSE(y_test, y_predict_GD_test):.5f}')

    y_predict_SGD_test = X_test @ theta_SGD
    print(f'MSE_SGD = {MSE(y_test, y_predict_SGD_test):.5f}')

    y_predict_OLS_test = X_test @ theta_OLS
    print(f'MSE_OLS = {MSE(y_test, y_predict_OLS_test):.5f}')


    """ Regression line plot """
    plt.scatter(x_train, y_predict_GD, c='r', label='GD') 
    plt.scatter(x_train, y_predict_SGD, c='b', label='SGD') 
    plt.scatter(x_train, y_predict_OLS, c='k', zorder=100, label='OLS') 
    plt.plot(x_train, y_train, 'g.', label='Data') # Data
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    plt.legend()
    plt.show()
    

    """ MSE plot """
    plt.yscale('log')
    iter_max = min(Niterations, n_epochs*n_minibatches)
    plt.plot(np.arange(Niterations)[:iter_max], mse_GD[:iter_max], label='MSE for GD')
    plt.plot(range(n_epochs*n_minibatches)[:iter_max], mse_SGD[:iter_max], label='MSE for SGD')
    plt.xlabel(r'$Iterations$')
    plt.ylabel(r'$MSE$')
    plt.legend()
    plt.show()