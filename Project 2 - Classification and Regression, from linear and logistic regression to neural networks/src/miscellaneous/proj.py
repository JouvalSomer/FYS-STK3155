from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm

from func import*

def data_FF(noise=True, step_size=0.05, alpha=0.05):
    x = np.arange(0, 1, step_size)
    y = np.arange(0, 1, step_size)
    X, Y = np.meshgrid(x, y)
    Z = FrankeFunction(X, Y, noise, alpha, seed=3155)
    return X, Y, Z


fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# ax = fig.gca(projection='3d')
X_, Y_, Z_ = data_FF(noise=True, step_size=0.01, alpha=0.2)
X, Y, Z = data_FF(noise=False, step_size=0.01, alpha=0.05)

Z_proj = abs(Z_ - Z)
# X, Y, Z = axes3d.get_test_data(0.05)
print(X.shape, Y.shape, Z.shape)
ax.plot_surface(X, Y, Z, alpha=0.7, cmap="winter")
# cset = ax.contourf(X, Y, Z, zdir='z', offset=-0.1, cmap=cm.coolwarm)
cset = ax.contourf(X, Y, Z_proj, zdir='z', offset=-0.1, cmap=cm.coolwarm)
# cset = ax.contourf(X, Y, Z, zdir='x', offset=-0.1, cmap=cm.coolwarm)
# cset = ax.contourf(X, Y, Z, zdir='y', offset=1.1, cmap=cm.coolwarm)

ax.set_xlabel('X')
# ax.set_xlim(-40, 40)
ax.set_ylabel('Y')
# ax.set_ylim(-40, 40)
ax.set_zlabel('Z')
# ax.set_zlim(-100, 100)

plt.show()