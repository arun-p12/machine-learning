
import numpy as np

def sigmoid(z):
    g = 1 / (1 + np.exp(-z))
    return(g)

def compute_cost(theta, X, y):
    p, q = X.shape
    theta = theta.reshape((q, 1))

    # compute the hypothesis
    hx = sigmoid(np.dot(X, theta))                              # m x n   * n x 1   =   m x 1

    # compute the individual cost terms ... (inside sigma)
    temp = (-y * np.log(hx)) - ((1 - y) * np.log(1 - hx))       # m x 1
 
    # do the summation, and average it out to get the final cost
    J = np.sum(temp) / p                                        # single value

    return(J)

def gradient_descent(theta, X, y):
    p, q = X.shape
    theta = theta.reshape((q, 1))

    hx = sigmoid(np.dot(X, theta))
    grad = np.dot(X.T, (hx - y)) / p    # n x m   *   m x 1   =  n x 1
    return(grad)

def compute_cost_reg(t, X, y, lbd):
    p, q = X.shape
    #t = t.reshape((q, 1))

    # compute the cost from standard LR and apply regularization
    cost = compute_cost(t, X, y)

    # account for regularization
    temp = t * t       # element-wise squaring of each term
    temp = temp.reshape(q, 1)
    temp[0] = 0        # since theta_0 isn't regularized

    reg_cost = (lbd / (2 * p)) * sum(temp)
    cost = cost + reg_cost
    return(cost)

def gradient_descent_reg(t, X, y, lbd):
    p, q = X.shape
    t = t.reshape((q, 1))

    # compute the gradient from standard LR and apply regularization
    grad = gradient_descent(t, X, y)
    grad = grad.reshape((q, 1))

    # account for regularization
    temp = t[:]
    temp[0,:] = 0


    reg_grad = ((lbd / p) * temp)
    grad = grad + reg_grad

    return(grad.flatten())

#########################################

def optimizer_func(theta, X, y):
    import scipy.optimize as op

    #return(op.fmin_bfgs(compute_cost2, theta, fprime=compute_grad2, args=(X, y)))
    #result = op.minimize(fun=compute_cost, x0=theta, args=(X, y), method='TNC', jac=gradient_descent)
    #result = op.minimize(fun=compute_cost2, x0=theta, args=(X, y), method='TNC', jac=compute_grad2)
    #result = op.fmin_tnc(func=compute_cost, x0=theta, fprime=gradient_descent, args=(X, y))

    # working
    #result = op.minimize(fun=CostFunc, x0=theta, args=(X, y), method='TNC', jac=Gradient)

    result = op.minimize(fun=compute_cost, x0=theta, args=(X, y), method='TNC', jac=gradient_descent)
    return(result.x)

def optimizer_func_reg(theta, X, y, lbd):
    # http://www.johnwittenauer.net/machine-learning-exercises-in-python-part-3/
    import scipy.optimize as op

    # working
    #result = op.minimize(fun=costReg, x0=theta, method='TNC', jac=gradientReg, args=(X, y, lbd))
    #return(result.x)

    # working #2
    result = op.optimize.fmin_bfgs(f=compute_cost_reg, x0=theta, args=(X, y, lbd), maxiter=400,
                                   fprime=gradient_descent_reg)
    return(result)

def optimizer_func_test(theta, X, y, lbd):
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize
    import scipy.optimize as op

    # Successful :: Powell, CG
    # Failed to Execute :: COBYLA, trust-constr, dogleg, trust-ncg, trust-exact, trust-krylov

    methods = ['Nelder-Mead', 'Powell', 'CG', 'Newton-CG', 'L-BFGS-B', 'TNC', 'COBYLA', 'SLSQP',
               'trust-constr', 'dogleg', 'trust-ncg', 'trust-exact', 'trust-krylov']
    opts = {'disp': True, 'maxiter': 1000}

    m = methods[2]
    print("Performing .... ", m)
    result = op.minimize(fun=compute_cost_reg, x0=theta, method=m, jac=gradient_descent_reg, args=(X, y, lbd), options=opts)
    return(result.x)
#########################################

def predict(theta, X):
    p = sigmoid(np.dot(X, theta))
    pos = np.where(p >= 0.5)
    pred = np.vstack(np.zeros(X.shape[0]))
    pred[pos] = 1
    return(pred)

def decision_points(theta, X, y):
    plot_x = [min(X[:, 2]) - 2, np.max(X[:, 2]) + 2]
    plot_y = (-1 / theta[2]) * ((theta[1] * plot_x) + theta[0])
    return(plot_x, plot_y)

'''
map the two input features to quadratic features used in the regularization exercise.

Returns a new feature array with more features, comprising of 
X1, X2, X1^2, X2^2, X1*X2, X1*X2^2, etc..
'''
def map_feature(X1, X2):
    degree = 6
    out = np.vstack(np.ones((X1.shape[0], 1)))
    for i in range(1, degree+1):
        for j in range(0, i+1):
            quad = np.vstack((X1 ** (i - j)) * (X2 ** j))
            out = np.concatenate((out, quad), axis=1)
    return(out)

def contour_plot(theta):
    import matplotlib.pyplot

    u = np.vstack(np.linspace(-0.75, 1.0, 50))
    v = np.vstack(np.linspace(-0.75, 1.0, 50))
    z = np.zeros((len(u), len(v)))

    for i in range(0, len(u)):
        for j in range(0, len(v)):
            map_feat = map_feature(u[i], v[j])
            z[i][j] = np.dot(map_feat, theta)

    # transform z before calling contour
    z = z.T

    # Plot z = 0, for cnt+1 contours
    cnt = 0
    matplotlib.pyplot.contour(u.flatten(), v.flatten(), z, cnt, linewidths=2)

def decision_boundary(theta, X, y, contour=0):
    import matplotlib.pyplot

    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    matplotlib.pyplot.scatter(X[pos, 1], X[pos, 2], c='g', s=7, label='Admitted')
    matplotlib.pyplot.scatter(X[neg, 1], X[neg, 2], c='r', s=7, marker='d', label="Not Admitted")
    matplotlib.pyplot.xlabel("Exam #1 score")

    if(contour):
        contour_plot(theta)
    else:
        plot_x, plot_y = decision_points(theta, X, y)
        matplotlib.pyplot.plot(plot_x, plot_y, label="Decison Boundary")

    matplotlib.pyplot.legend(loc='upper right', shadow=True)
    matplotlib.pyplot.legend(loc=9, bbox_to_anchor=(0.5, -0.1 * -11), ncol=3)
    matplotlib.pyplot.show()

def plot_data_ex2_reg(X, y, x_label="X axis", y_label="Y axis"):
    import matplotlib.pyplot

    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    matplotlib.pyplot.scatter(X[pos, 0], X[pos, 1], c='g', s=7, label='y = 1')
    matplotlib.pyplot.scatter(X[neg, 0], X[neg, 1], c='r', s=7, marker='d', label="y = 0")
    matplotlib.pyplot.xlabel(x_label)
    matplotlib.pyplot.ylabel(y_label)
    matplotlib.pyplot.legend(loc='upper right', shadow=True)
    matplotlib.pyplot.legend(loc=9, bbox_to_anchor=(0.5, -0.1*-11), ncol=2)
    matplotlib.pyplot.show()



'''
def sigmoid2(z):
    # Compute the sigmoid function
    den = 1.0 + np.e ** (-1.0 * z)
    d = 1.0 / den
    return d
'''

'''
def cost_and_gradient_reg(theta, X, y, lbd):
    m = len(y)
    J = 0.0

    # compute the cost and gradient from standard LR
    cost = compute_cost(theta, X, y)
    grad = gradient_descent(theta, X, y)

    # account for regularization
    temp = theta * theta
    temp[0] = 0                # since theta_0 isn't regularized

    cost = cost + (lbd / (2 * m)) * sum(temp)
    grad = grad + ((lbd / m) * temp)

    return(cost, grad)
'''

'''
def compute_cost2(theta, X, y):  # computes cost given predicted and actual values
    m = X.shape[0]  # number of training examples
    theta = np.reshape(theta, (len(theta), 1))

    # y = reshape(y,(len(y),1))

    J = (1. / m) * (-np.transpose(y).dot(np.log(sigmoid2(X.dot(theta)))) -
                    np.transpose(1 - y).dot(np.log(1 - sigmoid2(X.dot(theta)))))

    grad = np.transpose((1. / m) * np.transpose(sigmoid2(X.dot(theta)) - y).dot(X))
    # optimize.fmin expects a single value, so cannot return grad
    return J[0][0]  # ,grad
'''

'''
def compute_grad2(theta, X, y):
    #print theta.shape
    m = len(y)
    theta.shape = (1, 3)
    grad = np.zeros(3)
    h = sigmoid(X.dot(theta.T))
    delta = h - y
    l = grad.size

    for i in range(l):
        sumdelta = delta.T.dot(X[:, i])
        grad[i] = (1.0 / m) * sumdelta * - 1
    theta.shape = (3,)
    return grad
'''

'''
def CostFunc(theta, X, y):
    m,n = X.shape
    theta = theta.reshape((n,1))
    y = y.reshape((m,1))
    term1 = np.log(sigmoid(X.dot(theta)))
    term2 = np.log(1 - sigmoid(X.dot(theta)))
    term1 = term1.reshape((m,1))
    term2 = term2.reshape((m,1))
    term = y * term1 + (1 - y) * term2
    J = -((np.sum(term))/m)
    return J

def Gradient(theta, X, y):
    m , n = X.shape
    theta = theta.reshape((n,1))
    y = y.reshape((m,1))
    sigmoid_x_theta = sigmoid(X.dot(theta))
    grad = ((X.T).dot(sigmoid_x_theta-y))/m
    return grad.flatten()


def costReg(my_theta, X1, y, learningRate):
    #my_theta = np.matrix(my_theta)
    #X1 = np.matrix(X1)
    #y = np.matrix(y)

    hx = sigmoid(X1 * my_theta.T)

    first = np.multiply(-y, np.log(hx))
    second = np.multiply((1 - y), np.log(1 - hx))
    reg = (learningRate / 2 * len(X1)) * np.sum(np.power(my_theta[:,1:my_theta.shape[1]], 2))
    return np.sum(first - second) / (len(X1)) + reg

def gradientReg(my_theta, X1, y, learningRate):
    my_theta = np.matrix(my_theta)
    X1 = np.matrix(X)
    y = np.matrix(y)

    parameters = int(my_theta.ravel().shape[1])
    grad = np.zeros(parameters)
    error = sigmoid(X1 * my_theta.T) - y

    for i in range(parameters):
        term = np.multiply(error, X1[:,i])
        if (i == 0):
            grad[i] = np.sum(term) / len(X1)
        else:
            grad[i] = (np.sum(term) / len(X1)) + ((learningRate / len(X1)) * my_theta[:,i])
    return grad
'''