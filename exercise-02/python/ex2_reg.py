
import sys
sys.path.append("../../tool/")
import kit
import logistic_regression as lr
import numpy as np
np.set_printoptions(precision=4)


# visualize the data
data = np.array(kit.load_file('ex2data2.txt'))
X = np.vstack(data[:, [0, 1]])           # convert to vertical array ... column vector
y = np.vstack(data[:, 2])
m = len(y)

# plot the data
lr.plot_data_ex2_reg(X, y, 'Microchip Test #1', 'Microchip Test #2')

# Add Polynomial Features

# Note that mapFeature also adds a column of ones for us,
# so the intercept term is handled
X1 = lr.map_feature(np.vstack(X[:, 0]), np.vstack(X[:, 1]))

# Initialize fitting parameters
initial_theta = np.zeros((X1.shape[1], 1))

# Set regularization parameter lambda to 1
lbd = 1

# Compute and display initial cost and gradient for regularized logistic regression
cost = lr.compute_cost_reg(initial_theta, X1, y, lbd)
print('Cost at initial theta: {}'.format(cost))  # ans = 0.693

# expected response = 0.0085 0.0188 0.0001 0.0503 0.0115
grad = lr.gradient_descent_reg(initial_theta, X1, y, lbd)
print('Gradient at initial theta: first five values\n', grad[0:5])

# Compute and display cost and gradient with all-ones theta and lambda = 10
# expected results :: cost = 3.16 and grad = 0.3460 0.1614 0.1948 0.2269 0.0922
test_theta = np.ones((X1.shape[1], 1))
#(cost, grad) = lr.cost_and_gradient_reg(test_theta, X1, y, 10)
cost = lr.compute_cost_reg(test_theta, X1, y, 10)
grad = lr.gradient_descent_reg(test_theta, X1, y, 10)
print('Cost at test theta: {}'.format(cost))  # ans = 3.16
print('Gradient at test theta: first five values\n', grad[0:5])

# visualize the impact of different values of lambda
# A low value should result in overfitting the training data ... and high value in underfitting

lbd_start = 1
lbd_end = 1
while(lbd_start <= lbd_end):
    theta = lr.optimizer_func_reg(initial_theta, X1, y, lbd_start)
    theta = theta.reshape((X1.shape[1], 1))
    #print(theta)
    lr.decision_boundary(theta, X1, y, 1)
    lbd_start *= 3.0

# Compute accuracy on our training set :: ans = 83.1
p = lr.predict(theta, X1)
accuracy = np.sum(np.equal(p, y)) / m
print("Accuracy of the model = {:7.3f}%".format(accuracy * 100))
