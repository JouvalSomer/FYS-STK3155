import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def HyperbolicTangent(x):
    return np.tanh(x)

def ReLU(x):
    return x*(x>=0)
    
def LeakyReLU(x):
    return x*(x>=0) + 0.1*x*(x<0)

def elu(x):
    alpha = 1 #leak hyperparameter.
    return alpha*(np.exp(x) - 1)*(x<=0) + x*(x>0)

def Linear_Activation(x):
    return x


plt.rcParams.update({
"text.usetex": True,
"font.family": "serif",
"font.serif": ["ComputerModern"]})
plt.rcParams['figure.figsize'] = (12,6)
plt.rcParams.update({'font.size': 24})
plt.rc('axes', facecolor='whitesmoke', edgecolor='none',
    axisbelow=True, grid=True)
plt.rc('grid', color='w', linestyle='solid')
plt.rc('lines', linewidth=2)

x = np.linspace(-4, 4, 400)

fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3)
fig.suptitle('Activation functions', fontsize='xx-large')
ax1.plot(x, sigmoid(x))
ax1.set_title('Sigmoid', pad=10)
ax2.plot(x, HyperbolicTangent(x), 'tab:blue')
ax2.set_title('Hyperbolic Tangent', pad=10)
ax3.plot(x, ReLU(x), 'tab:blue')
ax3.set_title('ReLU', pad=10)
ax4.plot(x, LeakyReLU(x), 'tab:blue')
ax4.set_title('LeakyReLU', pad=10)
ax5.plot(x, elu(x), 'tab:blue')
ax5.set_title('ELU', pad=10)
ax6.plot(x, Linear_Activation(x), 'tab:blue')
ax6.set_title('Linear Activation', pad=10)
fig.set_size_inches((12, 9), forward=True)
plt.tight_layout()
plt.savefig('activation_functions.png', dpi=150)
plt.show()
