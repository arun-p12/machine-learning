import sys
sys.path.append("../../tool/")
import kit
import numpy as np
np.set_printoptions(precision=4)
import scipy.io as sio
import one_vs_all as ova
sys.path.append("../../exercise-02/python/")
import logistic_regression as lr

## Setup the parameters you will use for this part of the exercise
input_layer_size  = 400  # 20x20 Input Images of Digits
num_labels = 10          # 10 labels, from 1 to 10
                         # (note that we have mapped "0" to label 10)

# Part 1 :: Load and Visualize the data
# load the data from the matlab file, and extract X and y
mat = sio.loadmat('ex3data1.mat')
X = mat['X']
y = mat['y']
m = X.shape[0]

# visualize the data
# Randomly order numbers from 1 ... m (5000) ... then take the first 100
rand_indices = np.random.permutation(range(m))
selected = X[rand_indices[0:100], :]

# plot the data
ova.display_data(selected)

# Part 2a :: Logistic Regression for Neural Networks
# create a test data set to validate the cost function
theta_t = np.array([[-2], [-1], [1], [2]])

# create a 5x4 matrix ..... 5x1 concatenated with 5x3
# In octave reshape(1:15, 5,3)]; results in a 5x3 matrix with 1-5 in the first column 6-10 in the second etc.
# While np.arange gets the numbers 1 thru 15, reshape with 5x3 results in 1-5 in the first row.
# To get the format similar to octave, arrange the numbers as a 3x5 matrix, and transform.
X_t = np.concatenate((np.ones((5, 1)), (np.reshape(np.arange(1, 16), (3, 5)).T / 10.0)), axis=1)
y_t = np.array([[1], [0], [1], [0], [1]])
lambda_t = 3

# compute the cost and gradient, and validate its accuracy
J = lr.compute_cost_reg(theta_t, X_t, y_t, lambda_t)
grad = lr.gradient_descent_reg(theta_t, X_t, y_t, lambda_t)

print('Cost at test theta: {}'.format(J))  # ans = 2.534819
print('Gradient at test theta: \n', grad)       # ans = [0.146561], [-0.548558], [0.724722], [1.398003]]

# Part 2b :: One vs All training
# for each of the digits (classified output), find the optimized gradient.
# In the original data, digit 0 is represented as y=10, because octave indexing starts at 1
# But, since python has zero-indexing, we'll have to account for that in our predictions.
lambda_t = 0.1
all_theta = ova.one_vs_all(X, y, num_labels, lambda_t)

# Part 3: Predict for One-vs-All
# predict each of the digits based on the probability measure using the sigmoid measure
# since this is a classification problem, pick the one with the highest value
# regardless of whether the computed probability is greater than or less than 0.5
pred = ova.predict_one_vs_all(all_theta, X)
# compare our predictions with the real data, as a measure of the accuracy of our model
accuracy = np.sum(np.equal(pred, y)) / m
print("Accuracy of the model = {:7.3f}%".format(accuracy * 100))

# Part 4: Neural networks ... forward propagation
# The optimized Theta values from input to hidden layer (Theta1)
# and from the hidden layer to the output layer (Theta2) is already provided
hidden_layer_size = 25   # 25 hidden units
mat2 = sio.loadmat('ex3weights.mat')
Theta1 = mat2['Theta1']
Theta2 = mat2['Theta2']

# use the data to predict the output for each of the inputs
# and perform a random set of tests
#pred = ova.predict_nn(Theta1, Theta2, X)
#accuracy = np.sum(np.equal(pred, y)) / m
#print("Accuracy of the model = {:7.3f}%".format(accuracy * 100))
# display each digit, picked up at random, and compare against our prediction
for i in np.arange(m):
    my_X = X[rand_indices[i]]
    my_X = my_X.reshape((1, my_X.shape[0]))     # my_X is a 1xn matrix
    ova.display_data(my_X)
    pred_digit = ova.predict_nn(Theta1, Theta2, my_X)
    print("Neural Network Prediction: ", pred_digit)
    s = input('Paused - press enter to continue; q to exit:')
    if(s == 'q'): break