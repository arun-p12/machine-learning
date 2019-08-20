
import sys
sys.path.append("../../tool/")
import kit
import numpy as np
np.set_printoptions(precision=4)
import linear_regression as lr


# visualize the data  ... contains area in sqft, bedroom count, and the price
data = np.array(kit.load_file('ex1data2.txt'))
X = np.vstack(data[:, [0, 1]])           # convert to vertical array ... column vector
y = np.vstack(data[:, 2])
m = len(y)

# plot the data
#kit.plot_data(X[:, 0], y, 'Square Footage', 'Price', X[:, 1])


# normalize the data, and replot
X_norm, mu, sigma = lr.normalize_features(X)
#kit.plot_data(X_norm[:, 0], y, 'Square Footage', 'Price', X[:, 1])      # data is in X_norm ... BR color is in X

# cost function
one_v = np.reshape(np.ones(m), (m, 1))         # alternative to vstack
X1 = np.concatenate((one_v, X_norm), axis=1)   # one column prefixed for intercept (y = b + mx)
theta = np.array([[0.0], [0.0], [0.0]])

# Some gradient descent settings
iterations = 1500
alpha = 0.01

# run gradient descent  :: ans = [ [340412.563], [109370.0567], [-6500.6151] ]
theta_g, J_history = lr.gradient_descent(X1, y, theta, alpha, iterations)
print("Calculated theta (GD) = \n",theta_g)

# Estimate the price of a 1650 sq-ft, 3 br house
ho = np.array([[1650], [3]])    # specs of the house
ho = ho.T - mu                  # deviation from the mean
ho = ho / sigma                 # divided by the standard deviation
ho = np.insert(ho, 0, 1.0)      # insert a 1 at the beginning
price = np.dot(ho, theta_g)
print("Calculated price (GD) = ", price)

# run normal equation
X2 = np.concatenate((one_v, X,), axis=1)
theta_n = lr.normal_eqn(X2, y)
print("Calculated theta (NE) = \n",theta_n)

price = np.dot([1, 1650, 3], theta_n)
print("Calculated price (NE) = ", price)

# plot the convergence graph
kit.plot_data(np.arange(0, J_history.size), J_history, "Iterations", "Cost J")

# plot with different learning rates
i = 50
a = 0.01
step = 3
stop = 10

while(a <= stop):
    theta_g, J_history = lr.gradient_descent(X1, y, theta, a, i)
    kit.plot_data(np.arange(0, J_history.size), J_history, "Iterations", "Cost J")
    a *= step

