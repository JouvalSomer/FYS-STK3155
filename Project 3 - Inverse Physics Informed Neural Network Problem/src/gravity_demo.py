
import torch
import numpy as np
import matplotlib.pyplot as plt
import os 
from sklearn.model_selection import train_test_split

from network import Net


if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU ", torch.cuda.get_device_name())
else:
    device = torch.device("cpu")
    print("Using CPU")



max_iters = 1000
g = 9.81
v_0 = np.pi

times = torch.linspace(0, 1, 100, device=device)
positions = -1./2. * g * times ** 2 + v_0 * times


# Create some noisy measurements:
amplitude = 0.2
np.random.seed(15)
noise = amplitude * torch.from_numpy(np.random.sample(times.shape[0])).to(device) - amplitude / 2.
noisy_positions = (positions + noise).float().to(device)

num_measurements = 100
np.random.seed(15)
idx = np.random.choice(np.arange(len(times)), num_measurements, replace=False)
measurement_times = times[idx]
measurements = noisy_positions[idx]


times_train, times_test, measurement_times_train, measurement_times_test, measurements_train, \
    measurement_times_test = train_test_split(times, measurement_times, measurements, test_size=0.2)


NN = Net(num_hidden_units=16, num_hidden_layers=2, inputs=1).to(device)

# Initialize with bad guess:
g_init = g / 2
g_param = torch.tensor(g_init, device=device, dtype=torch.float)
g_param = torch.nn.Parameter(g_param)
g_param = g_param.to(device)

params = list(NN.parameters()) + [g_param]

loss_function=torch.nn.MSELoss(reduction="mean")

lbfgs_optim = torch.optim.LBFGS(params,
                                max_iter=max_iters,
                                tolerance_grad=1e-7,
                                tolerance_change=1e-10,
                                line_search_fn="strong_wolfe")


def pde_loss(nn, residual_points, g):
    
    residual_points.requires_grad = True
    
    y = nn(residual_points)
    
    dy_dt, = torch.autograd.grad(outputs=y,
                             inputs=residual_points,
                             grad_outputs=torch.ones_like(residual_points),
                             create_graph=True)

    ddy_dtt, = torch.autograd.grad(outputs=dy_dt,
                                 inputs=residual_points,
                                 grad_outputs=torch.ones_like(residual_points),
                                 create_graph=True) 
    # The ODE:
    residual = ddy_dtt + g
    
    return loss_function(residual, torch.zeros_like(residual))

def boundary_loss(nn, boundary_points, boundary_values):
    
    # Evaluate the NN at the boundary:
    predictions = nn(boundary_points)
    
    return loss_function(predictions, boundary_values)
    
losses_train = []
g_values_train = []


def closure():
    lbfgs_optim.zero_grad()

    data_loss_value = boundary_loss(NN, measurement_times_train, measurements_train)
    
    pde_loss_value = pde_loss(NN, times_train, g_param)
    
    loss = data_loss_value + pde_loss_value
    
    if loss.requires_grad:
        loss.backward()
        
    # Log both the loss and g during training:
    losses_train.append(loss.item())
    g_values_train.append(g_param.item())
    
    return loss


lbfgs_optim.step(closure)

prediction = NN(times_test)
plt.scatter(times_test.tolist(), prediction.tolist())
plt.show()

os.makedirs('results_gravity_test', exist_ok=True)

plt.plot(times.tolist(), positions.tolist(), label="True $y(t)$")
plt.plot(measurement_times.tolist(), measurements.tolist(), "x", label= str(num_measurements)+ " Measurements")
plt.xlabel("time")
plt.ylabel("Height $y$")
plt.legend()
# plt.savefig(f'results_gravity_test/data.png')
# plt.clf()
plt.show()

plt.figure()
plt.title("Loss")
plt.semilogy(losses_train)
# plt.savefig(f'results_gravity_test/loss_plot.png')
# plt.clf()
plt.show()

plt.figure()
plt.title("g during training, final g= "+format(g_param.item(), ".2f"), fontsize=14)
plt.plot(g_values_train, label="g parameter")
plt.plot(np.zeros(len(g_values_train)) + g, label="True")
plt.legend()
# plt.savefig(f'results_gravity_test/g_during_training.png')
# plt.clf()
plt.show()


plt.plot(times.tolist(), positions.tolist(), marker="o", markevery=14, label="True")
plt.plot(measurement_times.tolist(), measurements.tolist(), "x", label= str(num_measurements)+ " Measurements")
plt.plot(times.tolist(), NN(times).tolist(), "-", marker="s", markevery=10, label="NN")
plt.xlabel("time")
plt.ylabel("Height")
plt.legend()
# plt.savefig(f'results_gravity_test/network_predict.png')
# plt.clf()
plt.show()
