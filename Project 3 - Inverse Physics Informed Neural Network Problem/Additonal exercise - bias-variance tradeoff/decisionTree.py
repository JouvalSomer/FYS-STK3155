import numpy as np
from sklearn.tree import DecisionTreeRegressor, plot_tree, export_text
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from mlxtend.evaluate import bias_variance_decomp
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import operator
from sklearn.ensemble import BaggingRegressor

# Set seed
seed = 1
np.random.seed(seed)

# Define dataset
x = np.linspace(0,1,int(2e2))
x0 = .5
s2 = 0.1

y = np.exp(-(x-x0)**2/s2) + np.random.normal(0,.1,len(x))

# Splitting into test and train data
x_train, x_test, y_train, y_test = train_test_split(x.reshape(-1,1), y, test_size=0.2, random_state = seed)

# Loop through depths
mse = []
bias = []
vari = []

depth = 8
depths = np.linspace(1,depth,depth-1+1)
for i in depths:
    tree = RandomForestRegressor(max_depth=i)
    # tree = DecisionTreeRegressor(max_depth=i)

    # Bias-Variance analysis
    mse_test, bias_current, variance_current = bias_variance_decomp(tree, x_train, y_train, x_test, y_test, loss='mse', num_rounds=100)
    mse.append(mse_test); bias.append(bias_current); vari.append(variance_current)

# Make a fit for plotting
myFit = tree.fit(x_train, y_train)
yHat = myFit.predict(x_test)

# # Plotting
# plt.figure(figsize=(9,6))
# plot_tree(myFit)
# plt.show()


fig = go.Figure()
fig.add_trace(go.Scatter(x=depths, y=mse,
                    mode='lines',
                    name='MSE',
                    line=dict(width=2)))
fig.add_trace(go.Scatter(x=depths, y=bias,
                    mode='lines',
                    name='Bias^2',
                    line=dict(width=2)))
fig.add_trace(go.Scatter(x=depths, y=vari,
                    mode='lines',
                    name='Variance',
                    line=dict(width=2)))
fig.update_layout(title='Bias-Variance Tradeoff',
                   xaxis_title='Depth of Tree',
                   yaxis_title='Error')
fig.show()


L = sorted(zip(x_test.T[0], yHat), key=operator.itemgetter(0))
new_x_test, new_yHat = zip(*L)

H = sorted(zip(x_train.flatten(), y_train), key=operator.itemgetter(0))
new_x_train, new_y_train = zip(*H)


fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=y,
                    mode='lines',
                    name='All Data'))
fig.add_trace(go.Scatter(x=x, y=myFit.predict(x.reshape(-1,1)),
                    mode='lines',
                    name='Decision Tree'))
fig.add_trace(go.Scatter(x=new_x_test, y=new_yHat,
                    mode='markers',
                    name='Prediction'))
fig.update_layout(title='Decision Tree',
                   xaxis_title='x',
                   yaxis_title='y')
fig.show()