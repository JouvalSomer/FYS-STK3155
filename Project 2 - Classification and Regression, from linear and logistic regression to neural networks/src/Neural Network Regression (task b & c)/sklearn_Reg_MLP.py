from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split

from func import*

seed = 55
X_no_noise, Y_no_noise, Z_no_noise = data_FF(noise=False, step_size=0.02, alpha=0.05, reshape=False)

X, Y, Z = data_FF(noise=True, step_size=0.02, alpha=0.1, reshape=True)

x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(X, Y, Z, test_size=0.2, random_state = seed)

z_train = z_train.ravel()

input_train = np.c_[x_train, y_train]
input_test = np.c_[x_test, y_test]

regr = MLPRegressor(hidden_layer_sizes=(50, 50, 50),random_state=1, learning_rate_init=0.01, max_iter=50, activation='relu', solver='adam').fit(input_train, z_train)

z_pred = regr.predict(input_test)

R2 = regr.score(input_test, z_test)
print(R2)
