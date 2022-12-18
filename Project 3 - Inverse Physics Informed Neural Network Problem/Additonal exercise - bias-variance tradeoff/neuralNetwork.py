import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from mlxtend.evaluate import bias_variance_decomp
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Set seed
seed = 1
np.random.seed(seed)

# Define dataset
x = np.linspace(0,1,int(2e2))
x0 = 0.5
s2 = 0.1

y = np.exp(-(x-x0)**2/s2) + np.random.normal(0,.1,len(x)) 

# Splitting into test and train data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state = seed)

# Reshaping
input_train = np.c_[x_train]
input_test = np.c_[x_test]

mse = []
bias = []
vari = []

# Loop through nodes
nodes = [10, 20, 30, 40, 50, 100, 200, 250, 300, 325, 350, 375]
for i in range(len(nodes)):
    regr = MLPRegressor(hidden_layer_sizes=(int(nodes[i]), int(nodes[i]), int(nodes[i])), random_state=1, learning_rate_init=0.01, max_iter=100, activation='relu', solver='adam').fit(input_train, y_train)
    y_pred = regr.predict(x.reshape(-1,1))

    x_train = x_train.reshape(-1,1)
    x_test = x_test.reshape(-1,1)

    # Bias-Variance analysis
    mse_test, bias_current, variance_current = bias_variance_decomp(regr, x_train, y_train, x_test, y_test, loss='mse', num_rounds=10)
    mse.append(mse_test); bias.append(bias_current); vari.append(variance_current)


# Plotting
fig = go.Figure()
fig.add_trace(go.Scatter(x=nodes, y=mse,
                    mode='lines',
                    name='MSE',
                    line=dict(width=2)))
fig.add_trace(go.Scatter(x=nodes, y=bias,
                    mode='lines',
                    name='Bias^2',
                    line=dict(width=2)))
fig.add_trace(go.Scatter(x=nodes, y=vari,
                    mode='lines',
                    name='Variance',
                    line=dict(width=2)))
fig.update_layout(title='Bias-Variance Tradeoff',
                   xaxis_title='Number of Nodes',
                   yaxis_title='Error')
fig.update_yaxes(type="log")
fig.show()


fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=y,
                    mode='lines',
                    name='All Data',
                    line=dict(width=2)))
fig.add_trace(go.Scatter(x=x, y=y_pred,
                    mode='lines',
                    name='Neural Net',
                    line=dict(width=2)))
fig.update_layout(title='Neural Network',
                   xaxis_title='x',
                   yaxis_title='y')
fig.show()

