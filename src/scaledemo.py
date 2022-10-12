"""A test function where we compared the performance of scaled and not scaled
y_tilde_test and y_tilde. Also compared against a two manual scaler algos.
One algo that performs same as StandardScaler, and one that only subtracts
the mean."""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Common imports
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model as skl
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import  train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer

from func import*



def create_X(x, y, n ):
	if len(x.shape) > 1:
		x = np.ravel(x)
		y = np.ravel(y)

	N = len(x)
	l = int((n+1)*(n+2)/2)		# Number of elements in beta
	X = np.ones((N,l))

	for i in range(1,n+1):
		q = int((i)*(i+1)/2)
		for k in range(i+1):
			X[:,q+k] = (x**(i-k))*(y**k)

	return X

# Making meshgrid of datapoints and compute Franke's function

seed = 10
n = 30
N = 10000
np.random.seed(seed)
x = np.sort(np.random.uniform(0, 1000, N))
y = np.sort(np.random.uniform(0, 1000, N))

noice = 0.5
alpha = 1.0

z = FrankeFunction(x,y, noice, alpha, seed)

X = create_X(x, y, n=n)


X_train, X_test, y_train, y_test = train_test_split(X,z,test_size=0.2)





# """QR alt"""
# beta = QR_beta(X_train,y_train)


clf = skl.LinearRegression().fit(X_train, y_train)


print("The condition number of X_train is")
"""High condition number can result in numeric instability"""
print(np.linalg.cond(X_train))


print("The condition number of X_test is")
"""High condition number can result in numeric instability"""
print(np.linalg.cond(X_test))

"""Predictions:"""
y_tilde_not_scaled = clf.predict(X_train)
#y_tilde_not_scaled = X_train @ beta



print("predicted test without scale, and actual values:")

y_tilde_test = clf.predict(X_test)

#y_tilde_test = X_test @ beta


# The mean squared error and R2 score
print("MSE train before scaling: {:.4f}".format(mean_squared_error(y_tilde_not_scaled, y_train)))
#print("R2 score before scaling {:.2f}".format(clf.score(X_test,y_test)))
print("MSE test before scaling: {:.4f}".format(mean_squared_error(y_tilde_test, y_test)))


"""Code below for manual scaler, it behaved the same as the StandardScaler:"""

# X_train_mean = np.mean(X_train[:, 1:], axis = 0)
# X_train_std = np.std(X_train[:, 1:], axis = 0)
# X_train_std = np.std(X_train[:, 1:], axis = 0)
# X_train_std = np.std(X_train, axis = 0)


# X_train_scaled = np.ones_like(X_train)
# X_train_scaled[:,1:] = (X_train[:,1:] - X_train_mean)# / X_train_std
# #X_train_scaled = X_train_scaled


# X_test_mean = np.mean(X_test[:, 1:], axis = 0)
# X_test_std = np.std(X_test[:, 1:], axis = 0)
# X_test_scaled = np.ones_like(X_test)
# X_test_scaled[:,1:] = (X_test[:,1:] - X_train_mean)#/X_train_std
# #X_test_scaled = X_test_scaled/X_train_std
# #X_test_scaled[:,1:] = (X_test[:,1:] - X_test_mean) / X_test_std



"""Manual scaler that only subtracts the mean:"""

X_train_mean = np.mean(X_train[:, 1:], axis = 0)
X_train_std = np.std(X_train[:, 1:], axis = 0)
X_train_std = np.std(X_train[:, 1:], axis = 0)
X_train_std = np.std(X_train, axis = 0)


X_train_scaled = np.ones_like(X_train)
X_train_scaled[:,1:] = (X_train[:,1:] - X_train_mean)# / X_train_std
#X_train_scaled = X_train_scaled


X_test_mean = np.mean(X_test[:, 1:], axis = 0)
X_test_std = np.std(X_test[:, 1:], axis = 0)
X_test_scaled = np.ones_like(X_test)
X_test_scaled[:,1:] = (X_test[:,1:] - X_train_mean)#/X_train_std
# X_test_scaled = X_test_scaled/X_train_std
# X_test_scaled[:,1:] = (X_test[:,1:] - X_test_mean) / X_test_std




"""The Standardscaler"""
scaler = StandardScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)




print("The condition number of X_train_scaled is")
"""High condition number can result in numeric instability"""
print(np.linalg.cond(X_train_scaled))

print("The condition number of X_test_scaled is")
"""High condition number can result in numeric instability"""
print(np.linalg.cond(X_test_scaled))

clf_scale = skl.LinearRegression().fit(X_train_scaled, y_train)

"""QR alt"""
#beta_scale = QR_beta(X_train_scaled,y_train)

#y_tilde_scaled = X_train_scaled @ beta_scale

#y_tilde_test_scaled = X_test_scaled @ beta_scale

y_tilde_scaled = clf_scale.predict(X_train_scaled)
y_tilde_test_scaled = clf_scale.predict(X_test_scaled)

print("MSE train after  scaling: {:.4f}".format(mean_squared_error(y_tilde_scaled, y_train)))
print("MSE test after  scaling: {:.4f}".format(mean_squared_error(y_tilde_test_scaled, y_test)))



diff = y_tilde_not_scaled - y_tilde_scaled

print("the max diff between a y_tilde train scaled and y_tilde train is:")
print(np.max(np.abs(diff)))

y_tilde_test_scaled = clf_scale.predict(X_test)


diff_test = y_tilde_test_scaled- y_tilde_test

print("the max diff between a y_tilde test scaled and y_tilde test (not scaled) is:")
print(np.max(np.abs(diff_test)))


diff_test2 = y_tilde_test_scaled - y_test


print("the mean diff between a y_tilde test scaled and y_test is:")
print(np.mean(np.abs(diff_test2)))

print("the max diff between a y_tilde test scaled and y_test is:")
print(np.max(np.abs(diff_test2)))


diff_test3 = y_tilde_test- y_test

print("the mean diff from y_test for: y_tilde test and y_tilde test scaled:")
print(np.mean(np.abs(diff_test3)),np.mean(np.abs(diff_test2)))
