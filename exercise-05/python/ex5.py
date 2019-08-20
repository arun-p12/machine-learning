
import sys
sys.path.append("../../tool/")
import kit
import numpy as np
np.set_printoptions(precision=4)
import scipy.io as sio
sys.path.append("../../exercise-02/python/")
import logistic_regression as lr
sys.path.append("../../exercise-03/python/")
import one_vs_all as ova
import applying_ml as aml

# Part 1 :: Load and Visualize the data
# load the data from the matlab file, and extract X and y sub-divided into
# training, cross-validation, and test sets.
mat = sio.loadmat('ex5data1.mat')
X = mat['X']
y = mat['y']
Xtest = mat['Xtest']
ytest = mat['ytest']
Xval = mat['Xval']
yval = mat['yval']
m = X.shape[0]

# plot and visualize the data
#kit.plot_data(X, y, 'Change in water level', 'Water flowing out of the dam')

# Part 2 :: Implement the cost function for regularized linear regression
theta = np.array([[1], [1]])
lbd = 1
X1 = np.concatenate((np.ones((m, 1)), X), axis=1)
cost = aml.linear_compute_cost_reg(theta, X1, y, lbd)
grad = aml.linear_gradient_descent_reg(theta, X1, y, lbd)
print("Regularized cost with lambda=1 is = {:7.6f}".format(cost))  # ans = 303.993192
print("Linear gradient evaluated with theta as [[1], [1]] =\n{}".format(grad))  # ans =  [-15.303016; 598.250744]

# Part 3 :: Train the linear regression model
# note that a sraight line curve, will grossly underfit the given data
lbd = 0
initial_theta = np.vstack(np.zeros((X1.shape[1], 1)))
theta = aml.train_linear_reg(X1, y, lbd)
aml.plot_fit_over_data(X, y, aml.hypothesis(X1, theta), 'Change in water level', 'Water flowing out of the dam')

# Part 4 :: Now that we know our model doesn't do justice to the data, lets
# explore what type of an issue do we have. Overfit, Underfit, high bias / variance ...
# For this lets implement a learning curve
lbd = 0
Xval1 = np.concatenate((np.ones((Xval.shape[0], 1)), Xval), axis=1)
(error_train, error_val) = aml.learning_curve(X1, y, Xval1, yval, lbd)
aml.plot_learning_curve(range(1, m+1), error_train, error_val, 'Number of training examples', 'Error')

# Part 5 :: Feature Mapping for Polynomial Regression
# From the plot we can see that the model doesn't fit the data well....
# ans also we have high errors on training and cross-validation data
# Thus, a clear case of 'High Bias'. Lets add more polynomial features to fix this
p = 8
# Map X onto Polynomial Features and Normalize
X_poly = aml.poly_features(X, p)
(X_poly, mu, sigma) = aml.feature_normalize(X_poly)             # Normalize
X_poly = np.concatenate((np.ones((m, 1)), X_poly), axis=1)      # Add Ones

# Map X_poly_test and normalize (using mu and sigma)
X_poly_test = aml.poly_features(Xtest, p)
(X_poly_test, jk1, jk2) = aml.feature_normalize(X_poly_test, mu, sigma)
X_poly_test = np.concatenate((np.ones((X_poly_test.shape[0], 1)), X_poly_test), axis=1)

# Map X_poly_val and normalize (using mu and sigma)
X_poly_val = aml.poly_features(Xval, p)
(X_poly_val, jk1, jk2) = aml.feature_normalize(X_poly_val, mu, sigma)
X_poly_val = np.concatenate((np.ones((X_poly_val.shape[0], 1)), X_poly_val), axis=1)

# Part 6 :: Learning Curve for Polynomial Regression
lbd = 0
theta = aml.train_linear_reg(X_poly, y, lbd)
aml.poly_fit_and_plot(X, y, theta, mu, sigma, p)
(error_train, error_val) = aml.learning_curve(X_poly, y, X_poly_val, yval, lbd)
aml.plot_learning_curve(range(1, m+1), error_train, error_val, 'Number of training examples', 'Error')

# Part 7 ::  Validation for Selecting Lambda
(lbd_vec, error_train, error_val) = aml.validation_curve(X_poly, y, X_poly_val, yval)
aml.plot_learning_curve(lbd_vec, error_train, error_val, 'lambda', 'Error')

# Part 8 :: Optional :: Unseen data
#  Test the learning algorithm with the computed theta and lambda
#  values against the test data, that hasn't been seen by the model yet
pos = np.where(error_val == np.min(error_val))      # point of lowest error
lbd = lbd_vec[pos]                                  # the value of lambda at that point
theta = aml.train_linear_reg(X_poly, y, lbd)
error_test = aml.linear_compute_cost_reg(theta, X_poly_test, ytest, 0)
print('Err at lbd={} on Xtest = [{}]\n'.format(lbd, error_test))   # ans =  3.8599

# Part 9 :: Optional :: Small training size
#  Test the learning algorithm against differing subsets of data
#  Run them multiple times, and average out the results ....
lbd = 0.01
m = X_poly.shape[0]
error_test = np.zeros((m, 1))
for i in range(1, m+1):     # since the dataset is limited, run thru each count
    cnt = 4 * int(m/i)      # number of attempts to be averaged out
    cost = 0
    for j in range(1, cnt+1):
       rand_indices = np.random.permutation(range(m))   # randomize data
       pos = rand_indices[0:i]
       theta = aml.train_linear_reg(X_poly[pos, :], y[pos, :], lbd)
       cost = cost + aml.linear_compute_cost_reg(theta, X_poly_val, yval, 0)
    error_test[i-1, 0] = cost / cnt

# need to move this to a callable convenience function
import matplotlib.pyplot
matplotlib.pyplot.plot(range(m), error_test, label="Prediction from model")

matplotlib.pyplot.xlabel('Number of training examples')
matplotlib.pyplot.ylabel('Error')
matplotlib.pyplot.title('Polynomial Regression Fit (lambda = 0.01)')
matplotlib.pyplot.show()