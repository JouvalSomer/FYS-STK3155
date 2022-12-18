import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from mlxtend.evaluate import bias_variance_decomp
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import operator

def create_design_matrix(x,n):
    """
    Description: 
        Function that returns a design matrix X.
    Input: 
        Independent variable x.
    Output: 
        Design matrix with polynomial terms of x in columns.
    """
    N = len(x)
    X = np.zeros(shape=(N,n))
    for i in range(n):
        X[:,i] = x**(i+1)
    return X

# Set seed
seed = 1
np.random.seed(seed)

# Define dataset
x = np.linspace(0,1,int(2e2))
x0 = .5
s2 = 0.1

y = np.exp(-(x-x0)**2/s2) + np.random.normal(0,.1,len(x))

# List for storing error data
mse = []
mse_train = []
mse_test2 = []
bias = []
variance = []

# Looping through fits of different polynomial degrees
polydeg = np.linspace(1, 18, (18-1+1))
for i in range(len(polydeg)):
    X = create_design_matrix(x, int(polydeg[i]))

    # Split data into train and test sets
    X_train, X_test, y_train, y_test, x_train, x_test = train_test_split(X, y, x, test_size=0.2, random_state = seed)

    # Linear regression
    reg = LinearRegression().fit(X_train, y_train)

    ypred_train = reg.predict(X_train)
    R2_train = reg.score(X_train, y_train) 
    mse_train.append(mean_squared_error(y_train, ypred_train))

    ypred_test = reg.predict(X_test)
    R2_test = reg.score(X_test, y_test) 
    mse_test2.append(mean_squared_error(y_test, ypred_test))

    # # Bias-Variance analysis
    mse_test, bias_i, vari_i = bias_variance_decomp(LinearRegression(), X_train, y_train, X_test, y_test, loss='mse', num_rounds=100)
    mse.append(mse_test); bias.append(bias_i); variance.append(vari_i)


# Plotting
L = sorted(zip(x_train, y_train), key=operator.itemgetter(0))
new_x_train, new_y_train = zip(*L)


fig = go.Figure()
fig.add_trace(go.Scatter(x=polydeg, y=mse,
                    mode='lines',
                    name='MSE',
                    line=dict(width=2)))
fig.add_trace(go.Scatter(x=polydeg, y=bias,
                    mode='lines',
                    name=f'Bias^2',
                    line=dict(width=2)))
fig.add_trace(go.Scatter(x=polydeg, y=variance,
                    mode='lines',
                    name='Variance',
                    line=dict(width=2)))

fig.update_layout(title='Bias-Variance Tradeoff',
                   xaxis_title='Polynomial Degree',
                   yaxis_title='Error')
fig.update_yaxes(type="log")
fig.show()


X = create_design_matrix(x,4)
# X_train, X_test, y_train, y_test, x_train, x_test = train_test_split(X, y, x, test_size=0.2)
reg = LinearRegression().fit(X, y)
ypred = reg.predict(X)

fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=y,
                    mode='lines',
                    name='Train Data',
                    line=dict(width=2)))
fig.add_trace(go.Scatter(x=x, y=ypred,
                    mode='markers',
                    name=f'Polynomial Fit',
                    line=dict(width=2)))
fig.update_layout(title='Linear Regression',
                   xaxis_title='x',
                   yaxis_title='y')
fig.show()


fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=y,
                    mode='lines',
                    # name='All Data',
                    line=dict(width=2)))
fig.update_layout(title='The Dataset',
                   xaxis_title='x',
                   yaxis_title='y')
fig.show()

# for (x,y) in zip(polydeg,mse):
#     print(x, y)

