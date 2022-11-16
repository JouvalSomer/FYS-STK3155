"""Hyper Search for Binary Classification NN
Search with loops over parameters and for each parameter setting B bootstraps
NN are made and the mean and std dev of there are compared."""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor

from activation import*
from layer import*
from train_NN_Class import*
from func import*

#Breast Cancer Data set:
from sklearn.datasets import load_breast_cancer

#PCA:
from sklearn.decomposition import PCA

HidLayer = [1,2]
Nodes = [15,20,30]

""" Hyperparameter Search Space """
Epochs = [20,25,30]
init_LR = 0.0025        # Initial learning rate (LR)
decay = 0.0             # init_LR/n_epochs # LR decay rate (for fixed LR set decay to 0)
momentum = 0.8          # Momentum value for GD and SGD.
MiniBatch = [20,30,40]
Lmbdas = [0.000875]
ActFunc = [Sigmoid,LeakyReLU,Hyperbolic]

"""Note: The champion network had the following hyperpara:
NN: 1 Layer, 20 nodes, 25 epochs, 30 minibatxh size, lambda 0.000875."""


"""Boot runs"""
B=10 #10 for quick, then 100 then 1000

"""Lists of champion hyperpara and other rankings:"""
champion = [0,0,0,0,0,0,0,0] #list for champion details
secondrunner = [0,0,0,0,0,0,0,0] #list for champion details
thirdrunner = [0,0,0,0,0,0,0,0] #list for champion details
fourthrunner = [0,0,0,0,0,0,0,0] #list for champion details
fifthrunner = [0,0,0,0,0,0,0,0] #list for champion details

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
                            X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.20)

                            """The Standardscaler"""
                            scaler = StandardScaler()
                            scaler.fit(X_train)

                            X_train = scaler.transform(X_train)
                            X_test = scaler.transform(X_test)


                            """PCA (with PCA small performance loss, so deactive.)"""
                            # pca = PCA(n_components=20)
                            # X_train = pca.fit_transform(X_train)
                            # X_test = pca.transform(X_test)

                            # print("PCA explained variance:")
                            # print(pca.explained_variance_ratio_)
                            # print("PCA components:")
                            # print(abs( pca.components_ ))

                            """Nodes:"""
                            n_nodes_inputLayer = X_train.shape[1]
                            n_nodes_outputLayer = y_train.shape[1]



                            if hidden_layers == 0:
                                ANN_SGD = [
                                    Layer(n_nodes_inputLayer, n_nodes_outputLayer),
                                    Sigmoid()
                                ]


                            if hidden_layers == 1:
                                ANN_SGD = [
                                    Layer(n_nodes_inputLayer, Nodes_per_layer),
                                    ActivationFunc(),
                                    Layer(Nodes_per_layer, n_nodes_outputLayer),
                                    Sigmoid()
                                ]

                            if hidden_layers == 2:
                                ANN_SGD = [
                                    Layer(n_nodes_inputLayer, Nodes_per_layer),
                                    ActivationFunc(),
                                    Layer(Nodes_per_layer, Nodes_per_layer),
                                    ActivationFunc(),
                                    Layer(Nodes_per_layer, n_nodes_outputLayer),
                                    Sigmoid()
                                ]

                            if hidden_layers == 3:
                                ANN_SGD = [
                                    Layer(n_nodes_inputLayer, Nodes_per_layer),
                                    ActivationFunc(),
                                    Layer(Nodes_per_layer, Nodes_per_layer),
                                    ActivationFunc(),
                                    Layer(Nodes_per_layer, Nodes_per_layer),
                                    ActivationFunc(),
                                    Layer(Nodes_per_layer, n_nodes_outputLayer),
                                    Sigmoid()
                                ]

                            n_minibatches =  (X_train.shape[0]//minibatch_size) #number of minibatches
                            N_iter_GD = n_epochs*n_minibatches
                            seed = np.random.randint(0, 100000)

                            """ Optimization method """
                            Optimizer_method = ['Adam','momentum'] # Adagrad and RMSprop not yet implemented
                            O_M = Optimizer_method[0] #Choose the optimization method

                            mse_SGD = train_NN_SGD(ANN_SGD, X_train, y_train, n_epochs, init_LR, decay, O_M, momentum, minibatch_size, n_minibatches, seed, lmb)

                            y_pred_SGD = fwd(ANN_SGD, X_test)[0]


                            """TBD: Lage TEST COST PLOT"""


                            costboot[boot] = mse_SGD[-1] #stores last train MSE.

                            tol = 0.50
                            """Classsfication:"""
                            y_predicted = y_pred_SGD.copy()
                            y_predicted[y_predicted >= tol] = 1
                            y_predicted[y_predicted < tol] = 0

                            """Accuracy Score Based on the test set:"""
                            acc_score = accuracy(y_predicted,y_test)
                            acc_boot[boot] = acc_score

                        """The mean acc score and mean cost of the B bootstraps:"""
                        #print(" ")
                        #print("Confidence interval RESULTS FROM THE BOOTSTRAP: ")

                        #acc_confint = np.percentile(acc_boot,[2.5,97.5])

                        #print(acc_confint)
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
                            champion[5] = str(ActivationFunc)
                            champion[6] = challenger_score
                            champion[7] = challenger_variance
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
