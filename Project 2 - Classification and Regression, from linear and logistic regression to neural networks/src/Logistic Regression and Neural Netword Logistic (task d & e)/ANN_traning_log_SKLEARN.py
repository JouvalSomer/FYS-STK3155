"""SKLearn compare of NN Network."""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

#tried to avoid nuisance warnings:

from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning


#Breast Cancer Data set:
from sklearn.datasets import load_breast_cancer

#PCA:
from sklearn.decomposition import PCA

HidLayer = [1] #Best for NN
#HidLayer = [0] # Simulates Logistic Reg.

Nodes = [20] #Final.

""" Hyperparameter Search Space """
Epochs = [1000]
init_LR =[0.0025,0.001]        # Initial learning rate (LR)
decay = 0.0             # init_LR/n_epochs # LR decay rate (for fixed LR set decay to 0)
momentum = 0.8          # Momentum value for GD and SGD.
MiniBatch = [15,20,30,50,100]
Lmbdas = [0,0.000875]
ActFunc = ["Sigmoid"]
testsize = 0.20

"""Note: The champion network had the following hyperpara:
NN: 1 Layer, 20 nodes, 25 epochs, 30 minibatxh size, lambda 0.000875."""


"""Boot runs"""
B=100  #10 for quick, then 100 then 1000

"""Lists of champion hyperpara and other rankings:"""
champion = [0,0,0,0,0,0,0,0,0,0,0] #list for champion details cost start at 9999
secondrunner = [0,0,0,0,0,0,0,0,0,0,0] #list for champion details
thirdrunner = [0,0,0,0,0,0,0,0,0,0,0] #list for champion details
fourthrunner = [0,0,0,0,0,0,0,0,0,0,0] #list for champion details
fifthrunner = [0,0,0,0,0,0,0,0,0,0,0] #list for champion details

for hidden_layers in HidLayer:

    for Nodes_per_layer in Nodes:
        for n_epochs in Epochs:
            for minibatch_size in MiniBatch:
                for lmb in Lmbdas:
                    for ActivationFunc in ActFunc:

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

                            y_train = y_train.ravel()
                            y_test = y_test.ravel()

                            n_hidden_neurons = Nodes_per_layer
                            lmbd_vals = lmb
                            lmbd_vals = Lmbdas
                            epochs = n_epochs
                            eta = init_LR
                            eta_vals = init_LR

                        #from sklearn.neural_network import MLPClassifier
                        # store models for later use
                            DNN_scikit = np.zeros((len(eta_vals), len(lmbd_vals)), dtype=object)

                            for i, eta in enumerate(eta_vals):
                                for j, lmbd in enumerate(lmbd_vals):
                                    dnn = MLPClassifier(hidden_layer_sizes=(n_hidden_neurons), activation='logistic',
                                                        alpha=lmbd, learning_rate_init=eta, max_iter=epochs)
                                    dnn.fit(X_train, y_train)

                                    DNN_scikit[i][j] = dnn

                                    acc_score = dnn.score(X_test, y_test)

                            acc_boot[boot] = acc_score
#
                        """The mean acc score and mean cost of the B bootstraps:"""

                        challenger_score = np.mean(acc_boot)
                        challenger_variance = np.std(acc_boot)

                        if challenger_score > champion[6]: #compare acc score.
                            fifthrunner = fourthrunner.copy()
                            fourthrunner = thirdrunner.copy()
                            thirdrunner = secondrunner.copy()
                            secondrunner = champion.copy() # makes the prev champ second.
                            champion[0] = hidden_layers
                            champion[1] = Nodes_per_layer
                            champion[2] = n_epochs
                            champion[3] = minibatch_size
                            champion[4] = lmb
                            #champion[5] = str(ActivationFunc)
                            champion[6] = challenger_score
                            champion[7] = challenger_variance
                            #champion[8] = challenger_cost
                            #champion[9] = challenger_cost_var
                            champion[10] = testsize

                            print("We have a new champion:")
                            print (champion)
                            print("We have a new second:")
                            print (secondrunner)

                        print(f"Layers: {hidden_layers}, Nodes:{Nodes_per_layer}, n.epoch: {n_epochs} minibatch_size: {minibatch_size} Mean:{np.mean(acc_boot):.6f}  std:{np.std(acc_boot):.6f} based on {B} boot runs.")
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
