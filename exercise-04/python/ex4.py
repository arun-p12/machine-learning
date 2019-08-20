
import sys
sys.path.append("../../tool/")
import kit
import numpy as np
np.set_printoptions(precision=4)
import scipy.io as sio
import neural_networks as nn
sys.path.append("../../exercise-02/python/")
import logistic_regression as lr
sys.path.append("../../exercise-03/python/")
import one_vs_all as ova

## Setup the parameters you will use for this part of the exercise
input_layer_size  = 400  # 20x20 Input Images of Digits
hidden_layer_size = 25   # 25 hidden units
num_labels = 10          # 10 labels, from 1 to 10
                         # (note that we have mapped "0" to label 10)

# Part 1 :: Load and Visualize the data
# load the data from the matlab file, and extract X and y
mat = sio.loadmat('ex4data1.mat')
X = mat['X']
y = mat['y']
m = X.shape[0]

# visualize the data
# Randomly order numbers from 1 ... m (5000) ... then take the first 100
rand_indices = np.random.permutation(range(m))
selected = X[rand_indices[0:100], :]

# plot the data
#ova.display_data(selected)

# Load the weights into variables Theta1 and Theta2
mat2 = sio.loadmat('ex4weights.mat')
Theta1 = mat2['Theta1']             # hidden_layers x (input_layer_size + 1) ... +1 for bias
Theta2 = mat2['Theta2']             # num_labels x (hidden_layers + 1)  .... + 1 for bias

# unroll the parameters
t1_col = np.reshape(Theta1, ((Theta1.shape[0] * Theta1.shape[1]), 1))
t2_col = np.reshape(Theta2, ((Theta2.shape[0] * Theta2.shape[1]), 1))
nn_params = np.concatenate((t1_col, t2_col), axis=0)

# Part 2 :: Feedforward Neural Networks
# create a test data set to validate the cost function
lbd = 0
cost = nn.compute_cost_reg_nn(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lbd)
print("Non-regularized cost with lambda=0 is = {:7.6f}".format(cost))  # ans = 0.28763

# part 3 :: implement the regularization with the cos
lbd = 1
cost = nn.compute_cost_reg_nn(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lbd)
print("Regularized cost with lambda=1 is = {:7.6f}".format(cost))  # ans = 0.383770

# part 4 :: gradient for the sigmoid function
g = nn.sigmoid_gradient(np.array([-1, -0.5, 0, 0.5, 1]))
print("Sigmoid gradient evaluated at test z = {}".format(g))  # ans = 0.383770

# part 5 :: implment a two layer neural network that classifies digits.
# start by implementing a function to initialize the weights of the neural network
initial_Theta1 = nn.rand_initialize_weights(input_layer_size, hidden_layer_size)
initial_Theta2 = nn.rand_initialize_weights(hidden_layer_size, num_labels)
# Unroll parameters
initial_nn_params = nn.unroll_parameters(initial_Theta1, initial_Theta2)

# check back propagation gradients
#nn.check_gradients()            # lambda = 0
lbd = 3
nn.check_gradients(lbd)           # lambda = non-zero for regularization
# output the costFunction debugging :: ans = 0.576051
debug_cost  = nn.compute_cost_reg_nn(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lbd)
print("regularized cost with lambda=3 is = {:7.6f}".format(debug_cost))

# part 6 :: train the neural network and visualize
lbd = 0.1
grad = nn.optimizer_func_nn(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lbd)
grad = np.vstack(grad)
(Theta1, Theta2) = nn.roll_parameters(grad, hidden_layer_size, input_layer_size, num_labels)
ova.display_data(Theta1[:, 1:])

# part 7 :: predict the labels
pred = ova.predict_nn(Theta1, Theta2, X)
accuracy = np.sum(np.equal(pred, y)) / m
print("Accuracy of the model = {:7.3f}%".format(accuracy * 100))