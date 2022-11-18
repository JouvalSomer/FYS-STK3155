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


""" Hyperparameter Search Space """
Epochs = [500]
init_LR = 0.0025        # Initial learning rate (LR)
decay = 0.0             # init_LR/n_epochs # LR decay rate (for fixed LR set decay to 0)
momentum = 0.8          # Momentum value for GD and SGD.
MiniBatch = [30]
Lmbdas = [0.1]
testsize = 0.20

"""Boot runs"""
B=1000   #10 for quick, then 100 then 1000

"""Lists of champion hyperpara and other rankings:"""
champion = [0,0,0,0,0,0,0,0,0,0,0]
secondrunner = [0,0,0,0,0,0,0,0,0,0,0]
thirdrunner = [0,0,0,0,0,0,0,0,0,0,0]
fourthrunner = [0,0,0,0,0,0,0,0,0,0,0]
fifthrunner = [0,0,0,0,0,0,0,0,0,0,0]

for n_epochs in Epochs:
    for minibatch_size in MiniBatch:
        for lmb in Lmbdas:


            """Load the data"""
            cancer = load_breast_cancer()
            X = cancer.data
            Y = cancer.target.reshape(-1,1)

            """Corr matrix:"""
            # cancerpd = pd.DataFrame(cancer.data, columns=cancer.feature_names)
            # correlation_matrix = cancerpd.corr().round(1)

            """Bootstrapping over B boots to find mean Cost and Accuracy:"""

            costboot = np.zeros((B)).reshape(-1,1)
            acc_boot = np.zeros((B)).reshape(-1,1)

            for boot in range (B):

                """Split Randomly"""
                X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=testsize)

                """The Standardscaler"""
                scaler = StandardScaler()
                scaler.fit(X_train)

                X_train = scaler.transform(X_train)
                X_test = scaler.transform(X_test)


                """PCA (with PCA small performance loss, so deactive.)"""

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

                n_minibatches =  (X_train.shape[0]//minibatch_size) #number of minibatches
                N_iter_GD = n_epochs*n_minibatches
                seed = np.random.randint(0, 100000)

                """ Gradient method """
                Gradient_method = ['log', 'anal']
                G_M = Gradient_method[0] #Choose the Gradient method

                """ Optimization method """
                # If you want plain GD without any optimization choose 'momentum' with momentum value of 0
                Optimizer_method = ['Adagrad', 'RMSprop', 'Adam','momentum']
                O_M = Optimizer_method[2] #Choose the optimization method

                y_pred, prob_SGD, theta, cost, acc_score = SGD(X_train, X_test, y_train, y_test, O_M, G_M, minibatch_size, n_minibatches, n_epochs, init_LR, decay, momentum, seed, lmb)


                costboot[boot] = cost[-1]

                tol = 0.50
                """Classsfication:"""
                y_predicted = prob_SGD.copy()
                y_predicted[y_predicted >= tol] = 1
                y_predicted[y_predicted < tol] = 0

                """Accuracy Score Based on the test set:"""
                acc_score = accuracy(y_predicted,y_test)
                acc_boot[boot] = acc_score

            #print(acc_confint)
            challenger_score = np.mean(acc_boot)
            challenger_variance = np.std(acc_boot)
            challenger_cost = np.mean(costboot)
            challenger_cost_var = np.std(costboot)

            if challenger_score > champion[6]: #compare acc score.
                fifthrunner = fourthrunner.copy()
                fourthrunner = thirdrunner.copy()
                thirdrunner = secondrunner.copy()
                secondrunner = champion.copy() # makes the prev champ second.


                champion[2] = n_epochs
                champion[3] = minibatch_size
                champion[4] = lmb

                champion[6] = challenger_score
                champion[7] = challenger_variance
                champion[8] = challenger_cost
                champion[9] = challenger_cost_var
                champion[10] = testsize

                print("We have a new champion:")
                print (champion)
                print("We have a new second:")
                print (secondrunner)

                print(f"n.epoch: {n_epochs} minibatch_size: {minibatch_size} Mean:{np.mean(acc_boot):.6f}  std:{np.std(acc_boot):.6f} based on {B} boot runs.")
print("The Champion is:")
print (champion)

print("The Secondrunner is:")
print (secondrunner)

print("The Thirdrunner is:")
print (thirdrunner)

print("The Fourthrunner is:")
print (fourthrunner)

print("The Fifthrunner is:")
print (fifthrunner)
