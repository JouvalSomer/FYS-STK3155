"""logistic regression"""
import numpy as np

from descent_methods_logistic import*

import matplotlib.pyplot as plt
from sklearn.model_selection import  train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

#PCA:
from sklearn.decomposition import PCA

"""Load the data"""
cancer = load_breast_cancer()
X = cancer.data
Y = cancer.target.reshape(-1,1)

"""Bootstrapping over B boots to find mean Cost and Accuracy:"""
B=1000
costboot = np.zeros((B)).reshape(-1,1)
acc_boot = np.zeros((B)).reshape(-1,1)

for boot in range (B):

    """Split Randomly and get design matrix"""
    X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.20)

    """The Standardscaler"""
    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    """Add Intercept (did actually not make much difference.)"""
    X_train    = np.c_[np.ones(X_train.shape[0]), X_train]
    X_test     = np.c_[np.ones(X_test.shape[0]), X_test]


    """PCA (disabled. Did not improve model)"""
    # pca = PCA(n_components=20)
    # X_train = pca.fit_transform(X_train)
    # X_test = pca.transform(X_test)

    # print("PCA explained variance:")
    # print(pca.explained_variance_ratio_)
    # print("PCA components:")
    # print(abs( pca.components_ ))

    """ Hyperparameters """
    n_epochs = 500          # Number of epochs
    init_LR = 0.0025        # Initial learning rate (LR)
    decay = 0.0             # init_LR/n_epochs # LR decay rate (for fixed LR set decay to 0)
    momentum = 0.8          # Momentum value for GD and SGD.
    minibatch_size = 30
    n_minibatches =  (X_train.shape[0]//minibatch_size) #number of minibatches
    N_iter_GD = n_epochs*n_minibatches
    lmb =  0.000875
    seed = np.random.randint(0, 100000)

    """ Gradient method """
    Gradient_method = ['log', 'anal']
    G_M = Gradient_method[0] #Choose the Gradient method

    """ Optimization method """
    # If you want plain GD without any optimization choose 'momentum' with momentum value of 0
    Optimizer_method = ['Adagrad', 'RMSprop', 'Adam','momentum']
    O_M = Optimizer_method[2] #Choose the optimization method

    y_pred, prob_SGD, theta, cost, acc_score = SGD(X_train, X_test, y_train, y_test, O_M, G_M, minibatch_size, n_minibatches, n_epochs, init_LR, decay, momentum, seed, lmb)

    #print(cost.shape)

    """TBD: Lage TEST COST PLOT"""

    costboot[boot] = cost[-1]

    tol = 0.50
    """Classsfication:"""
    y_predicted = prob_SGD.copy()
    y_predicted[y_predicted >= tol] = 1
    y_predicted[y_predicted < tol] = 0

    """Accuracy Score Based on the test set:"""
    acc_score = accuracy(y_predicted,y_test)
    acc_boot[boot] = acc_score


"""The mean acc score and mean cost of the B bootstraps:"""
print(" ")
print("RESULTS FROM THE BOOTSTRAP: ")

acc_confint = np.percentile(acc_boot,[2.5,97.5])

print(f"The mean accuracy score is {np.mean(acc_boot):.6f}  and its standard dev is {np.std(acc_boot):.6f} based on {B} boot runs.")

print(" ")
# Print the confidence interval
print('Accuracy score within 95% interval of =', acc_confint, 'percent')
print(" ")
print(f"The mean train entropy cost and its standard dev from {B} boot runs:")
#print(costboot)
print(f"{np.mean(costboot):.8f},{np.std(costboot):.8f}")
# Print the confidence interval
print(" ")
cost_confint = np.percentile(costboot,[2.5,97.5])
print(f"95% confidence interval of the train cost for {B} boot runs is {cost_confint}")


# Plot the histogram of the boot accuracies:
plt.hist(acc_boot, bins=20, density=True)
plt.xlabel(r'Accuracy Score')
plt.ylabel('PDF')
plt.savefig("accuracy.jpg")
plt.show()


# Plot the histogram of the boot cost:
plt.hist(costboot, bins=20, density=True)
plt.xlabel(r' Cost')
plt.ylabel('PDF')
plt.savefig("cost.jpg")
plt.show()


print(" ")
print("RESULT FROM THE LAST RUN IN THE BOOTSTRAP:")
print(" ")
print(f"Correctly scored {np.sum(y_predicted==y_test)} out of {y_test.shape[0]}")

print(f"Incorrectly scored {np.sum(y_predicted!=y_test)} out of {y_test.shape[0]}")

print("Scored positive by the LR:")
print(int(np.sum(y_predicted)))

print("Scored negative by the LR:")
print(int(np.sum(1 - y_predicted)))


print("Correctly scored positive by LR out of total positive:")
print(int(np.sum(y_predicted*y_test)),np.sum(y_test))
#
print("Correctly scored negative by LR out of total negative:")
print(int(np.sum((1-y_predicted)*(1-y_test))), np.sum(1-y_test))
