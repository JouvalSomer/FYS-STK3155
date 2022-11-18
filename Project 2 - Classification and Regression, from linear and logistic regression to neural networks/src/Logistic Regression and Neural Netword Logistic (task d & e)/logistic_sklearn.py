"""logistic regression with Sklearn to compare."""
import numpy as np

from descent_methods_logistic import*

import matplotlib.pyplot as plt
from sklearn.model_selection import  train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

#PCA:
from sklearn.decomposition import PCA
seed = np.random.randint(0, 100000)

"""Load the data"""
cancer = load_breast_cancer()
X = cancer.data
Y = cancer.target.reshape(-1,1)

"""Bootstrapping over B boots to find mean Accuracy:"""
B=1000
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


    """Compares with Sklearn"""

    from sklearn import linear_model

    logreg = linear_model.LogisticRegression(random_state = seed, max_iter = 5000)
    logreg.fit(X_train, y_train.ravel())
    SK_accuracy = logreg.score(X_test, y_test)
    acc_boot[boot] = SK_accuracy

print(np.mean(acc_boot))

"""The mean acc score and mean cost of the B bootstraps:"""
print(" ")
print("RESULTS FROM THE SKLEARN BOOTSTRAP: ")

acc_confint = np.percentile(acc_boot,[2.5,97.5])

print(f"The mean accuracy score is {np.mean(acc_boot):.6f}  and its standard dev is {np.std(acc_boot):.6f} based on {B} boot runs.")

print(" ")
# Print the confidence interval
print('Accuracy score within 95% interval of =', acc_confint, 'percent')
print(" ")
