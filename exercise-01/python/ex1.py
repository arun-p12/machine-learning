
import sys
sys.path.append("../../tool/")
import kit
import numpy as np
np.set_printoptions(precision=4)


# visualize the data
data = np.array(kit.load_file('ex1data1.txt'))
X = np.vstack(data[:, 0])           # convert to vertical array ... column vector
y = np.vstack(data[:, 1])
m = len(y)

# plot the data
kit.plot_data(X, y, 'Price (x $10k)', 'Population (x 10k)')

# cost function
one_v = np.reshape(np.ones(m), (m, 1))         # alternative to vstack
X1 = np.concatenate((one_v, X), axis=1)        # two columns
theta = np.array([[0.0], [0.0]])

# compute and display initial cost :: ans = 32.07
import linear_regression as lr

J = lr.compute_cost(X1, y, theta)
print("With theta = [0 ; 0] ... Cost computed = {:7.3f}".format(J));

# further testing of the cost function :: ans = 54.24
J = lr.compute_cost(X1, y, [[-1.0], [2.0]])
print("With theta = [-1 ; 2] ... Cost computed = {:7.3f}".format(J));

# Some gradient descent settings
iterations = 1500
alpha = 0.01

# run gradient descent  :: ans = [ [-3.6303],  [1.1664] ]
theta, J_history = lr.gradient_descent(X1, y, theta, alpha, iterations)
print("Calculated theta = \n",theta)

# predict values for population sizes of 35,000 and 70,000
p1 = np.dot([[1, 3.5]], theta)
p2 = np.dot([[1, 7.0]], theta)
print("For population of 35k, profit = {}".format(p1 * 10000))
print("For population of 70k, profit = {}".format(p2 * 10000))

# overlay the hypothesis on the data
from matplotlib import pyplot
#pyplot.scatter(X, y, c='b', s=7)
#pyplot.xlabel('Price (x $10k)')
#pyplot.ylabel('Population (x 10k)')
#pyplot.plot(X, np.dot(X1, theta), 'r-')
#pyplot.show()

# https://stackoverflow.com/questions/9170838/surface-plots-in-matplotlib
# https://matplotlib.org/examples/color/colormaps_reference.html
# https://jakevdp.github.io/PythonDataScienceHandbook/04.12-three-dimensional-plotting.html
# Visualizing J(theta_0, theta_1)
# Grid over which we will calculate J
theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-1, 4, 100)

# initialize J_vals to a matrix of 0's
J_vals = np.zeros((len(theta0_vals), len(theta1_vals)))

# Fill out J_vals
for i in range(0, len(theta0_vals)):
    for j in range(0, len(theta1_vals)):
        t = [[theta0_vals[i]], [theta1_vals[j]]]
        J_vals[i][j] = lr.compute_cost(X1, y, t)

J_vals = J_vals.T
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import pandas as pd

fig = pyplot.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(theta0_vals, theta1_vals, J_vals, cmap=cm.gnuplot2)
pyplot.show()

# Contour plot
fig = pyplot.figure()
# Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
pyplot.contour(theta0_vals, theta1_vals, J_vals, np.logspace(-2, 3, 20))
pyplot.xlabel('theta_0')
pyplot.ylabel('theta_1')
pyplot.plot(theta[0], theta[1], 'rx')
pyplot.show()
