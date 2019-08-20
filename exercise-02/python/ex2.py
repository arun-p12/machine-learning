
import sys
sys.path.append("../../tool/")
import kit
import numpy as np
np.set_printoptions(precision=4)


# load and visualize the data
data = np.array(kit.load_file('ex2data1.txt'))
X = np.vstack(data[:, [0, 1]])           # convert to vertical array ... column vector
y = np.vstack(data[:, 2])
m = len(y)

# plot the data
kit.plot_data_classify(X, y, 'Exam #1 Score', 'Exam #2 Score')

# cost function
one_v = np.reshape(np.ones(m), (m, 1))         # alternative to vstack
X1 = np.concatenate((one_v, X), axis=1)        # two columns
initial_theta = np.zeros((X.shape[1]+1, 1))    # Column vector with 1 + # of features (X.shape[1])

# calculate cost function :: ans = 0.693
import logistic_regression as lr
J = lr.compute_cost(initial_theta, X1, y)
print("With theta = [0; 0; 0] ... Cost computed = {:7.3f}".format(J));

# run gradient descent :: ans = [ [-0.1], [-12.0092],  [-11.2628] ]
grad = lr.gradient_descent(initial_theta, X1, y)
print("Calculated GD = \n", grad)

# Compute and display cost and gradient with non-zero theta
test_theta = np.array([[-24], [0.2], [0.2]])
J = lr.compute_cost(test_theta, X1, y);
grad = lr.gradient_descent(test_theta, X1, y)

print('Cost at test theta: {:7.3f}'.format(J))  # ans = 0.218
print('Gradient at test theta: \n', grad)       # ans = [[0.043], [2.566], [2.647]]

# overlay the decision boundary on the data
# but, first compute the optimized theta for global min :: ans = [[-25.161], [0.206], [0.201]]
theta = lr.optimizer_func(initial_theta, X1, y)
print('Computed theta: ', theta)
theta = np.vstack(theta)

# now compute the decision boundary
lr.decision_boundary(theta, X1, y)

# test the model by running a prediction  :: ans = 0.775 +/- 0.002
# for a student with score 45 on exam 1 and score 85 on exam 2
X_test = np.array([1, 45, 85])
prob = lr.sigmoid(np.dot(X_test, theta))
print("Probability of student with scores {} getting admitted = {}".format(X_test[[1,2]], prob));

# calculate the overall accuracy of our model :: ans = 89.0
p = lr.predict(theta, X1)
accuracy = np.sum(np.equal(p, y)) / m
print("Accuracy of the model = {:7.3f}%".format(accuracy * 100));
