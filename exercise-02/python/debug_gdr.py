
import sys
sys.path.append("../tool/")
import kit
import logistic_regression as lr
import numpy as np
np.set_printoptions(precision=4)


data = np.array(kit.load_file('ex2data2.txt'))
X = np.vstack(data[:, [0, 1]])
y = np.vstack(data[:, 2])
m = len(y)

X1 = lr.map_feature(np.vstack(X[:, 0]), np.vstack(X[:, 1]))

my_theta = np.zeros((X1.shape[1], 1))
lbd_start = 7.5
lbd_end = lbd_start

lbd = lbd_start
while(lbd <= lbd_end):
    print("Computing at lambda = ", lbd)
    my_theta = my_theta + lbd

    cost = lr.compute_cost_reg(my_theta, X1, y, lbd)

    grad = lr.gradient_descent_reg(my_theta, X1, y, lbd)

    print('Cost at my theta: {}'.format(cost))
    print('Gradient at my theta: first five values :: ', lbd, "\n", grad[0:5])

    #theta = lr.optimizer_func2(my_theta, X1, y, lbd)
    #theta = theta.reshape((X1.shape[1], 1))
    #lr.decision_boundary(theta, X1, y, 1)

    #p = lr.predict(theta, X1)
    #accuracy = np.sum(np.equal(p, y)) / m
    #print("Accuracy of the model = {:7.3f}%".format(accuracy * 100))

    lbd += 0.5
