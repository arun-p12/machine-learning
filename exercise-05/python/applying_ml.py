
import numpy as np
import matplotlib.pyplot

# breaking down the steps in the computation of linear regression
# start with computing the hypothesis / prediction
def hypothesis(X, theta):
    return(np.dot(X, theta))

# compute the deviation from the actual.. Sum of Squared Errors
def sum_of_squared_errors(hx, y):
    diff = hx - y
    return(np.sum(diff ** 2))

# compute the cost function
def linear_compute_cost_reg(theta, X, y, lbd):
    m = X.shape[0]

    theta = np.vstack(theta)
    hx = hypothesis(X, theta)
    sse = sum_of_squared_errors(hx, y)
    cost = sse / (2 * m)                # unregularized cost

    theta[0, :] = 0
    reg = (lbd / (2 * m)) * np.sum(theta ** 2)
    cost = cost + reg
    return(cost)

# compute the gradient
# the returned data must be flattened in the form (m, ) rather than
# (m, n) for optimizer algorithms such as 'CG' and 'TNC to work
def linear_gradient_descent_reg(theta, X, y, lbd):
    m = X.shape[0]

    theta = np.vstack(theta)
    hx = hypothesis(X, theta)
    diff = hx - y

    grad = np.dot(X.T, diff) / m     # unregulaized
    grad = np.vstack(grad)

    theta[0, :] = 0
    reg = (lbd / m) * theta
    reg = np.vstack(reg)
    grad = grad + reg
    return(grad.flatten())

# the optimizer function... eqvt of fmincg in octave
def train_linear_reg(X, y, lbd):
    import scipy.optimize as op

    initial_theta = np.zeros((X.shape[1], 1))
    # Successful :: Powell, CG
    methods = ['Nelder-Mead', 'Powell', 'CG', 'Newton-CG', 'L-BFGS-B', 'TNC', 'COBYLA', 'SLSQP']
    opts = {'disp': True, 'maxiter': 1000}

    m = methods[2]
    print("Performing .... ", m)
    result = op.minimize(fun=linear_compute_cost_reg, x0=initial_theta, method=m,
                         jac=linear_gradient_descent_reg, args=(X, y, lbd), options=opts)
    return(result.x)

# copy of code from exercise #1 ... plotting two Y curves for the same X
# one of them is the scatter plot of the data... and the other
# the curve representing our hypothesis
def plot_fit_over_data(x_list, y_list, hx_list, x_label="X axis", y_label="Y axis"):
    params = {'s':7, 'marker':'o', 'label':'Figure 1'}

    for i in range(0, len(x_list)):
        x = x_list[i]
        y = y_list[i]
        matplotlib.pyplot.scatter(x, y, c='b', s=7)

    matplotlib.pyplot.xlabel(x_label)
    matplotlib.pyplot.ylabel(y_label)

    matplotlib.pyplot.plot(x_list, hx_list, label="Prediction from model")
    matplotlib.pyplot.show()

# Compute the error in the training and cross-validation data
# generate them for multiple data size, so that we can see the trend
# The curve helps identify if the problem is of high bias or variance
def learning_curve(X1, y, Xval1, yval, lbd):
    m = X1.shape[0]
    #initial_theta = np.vstack(np.zeros((X1.shape[1], 1)))
    error_train = np.zeros((m, 1))
    error_val = np.zeros((m, 1))
    for i in range(1, m+1):
        Xt = X1[0:i, :]
        yt = y[0:i, :]
        theta = train_linear_reg(Xt, yt, 1)
        theta = np.vstack(theta)
        error_train[i-1, 0] = linear_compute_cost_reg(theta, Xt, yt, 0)
        error_val[i-1, 0] = linear_compute_cost_reg(theta, Xval1, yval, 0)
    return(error_train, error_val)

# plot the above generated data
def plot_learning_curve(x_list, error_train, error_val, xlabel, ylabel):
    matplotlib.pyplot.plot(x_list, error_train, label='Train')
    matplotlib.pyplot.plot(x_list, error_val, label='Cross-Validation')
    matplotlib.pyplot.xlabel(xlabel)
    matplotlib.pyplot.ylabel(ylabel)
    matplotlib.pyplot.legend(loc=9, bbox_to_anchor=(0.5, -0.1*-11), ncol=2)
    matplotlib.pyplot.show()

# For more complex curves than straight lines using the linear regression model
# generate higher order polynomials. X^2, X^3, ... X^n, etc.
def poly_features(X, p):
    X_poly = np.zeros((X.shape[0], p))
    for i in range(p):
        poly = X ** (i+1)
        X_poly[:, i] = poly[:, 0]
    return(X_poly)

# Normalize the features so that higher order polynomials don't skew the curve
def feature_normalize(X, mu=np.array([]), sigma=np.array([])):
    if(mu.shape[0] == 0): mu = np.mean(X, axis=0)
    # https://stackoverflow.com/questions/27600207/why-does-numpy-std-give-a-different-result-to-matlab-std
    # ddof=1 for octave / matlab like standard deviation
    if(sigma.shape[0] == 0): sigma = np.std(X, axis=0, ddof=1)
    X = X - mu
    X = X / sigma
    return (X, mu, sigma)

# with the additional features (higher order polynomials), regenerate the
# hypothesis, and plot it.
def poly_fit_and_plot(X, y, theta, mu, sigma, p):
    min_x = np.min(X)
    max_x = np.max(X)
    # extend range of data to get an idea of the how the model behave outside the
    # range of the given data points
    Xext = np.arange(min_x - 15, max_x + 25, 0.05)  # in increments of 0.05
    Xext_poly = np.vstack(Xext)
    Xext_poly = poly_features(Xext_poly, p)
    (Xext_poly, jk1, jk2) = feature_normalize(Xext_poly, mu, sigma)  # Normalize
    Xext_poly = np.concatenate((np.ones((Xext_poly.shape[0], 1)), Xext_poly), axis=1)

    # the original data as a scatter plot
    for i in range(0, len(X)):
        xp = X[i]
        yp = y[i]
        matplotlib.pyplot.scatter(xp, yp, c='r', s=7)

    # for the points extended on either size of the min and max of X,
    # compute the higher order polynomial, and from it compute the
    # hypothesis/prediction ... plot the fit.
    hx = hypothesis(Xext_poly, theta)
    matplotlib.pyplot.plot(Xext, hx, label="Prediction from model")

    matplotlib.pyplot.xlabel('Change in water level')
    matplotlib.pyplot.ylabel('Water flowing out of the dam')
    matplotlib.pyplot.title('Polynomial Regression Fit (lambda = 0)')
    matplotlib.pyplot.show()

    # return the extended polynomial data
    #return(Xext_poly)

def validation_curve(X_poly, y, X_poly_val, yval):
    # Selected values of lambda
    lbd_vec = np.array([0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10])
    lbd_vec = np.vstack(lbd_vec)
    error_train = np.zeros((lbd_vec.shape[0], 1))
    error_val = np.zeros((lbd_vec.shape[0], 1))

    for i in range(lbd_vec.shape[0]):
        lbd = lbd_vec[i]
        # use lambda to compute theta
        theta = train_linear_reg(X_poly, y, lbd)
        # rest lambda back to 0, for the non-regularized cost
        error_train[i] = linear_compute_cost_reg(theta, X_poly, y, 0)
        error_val[i] = linear_compute_cost_reg(theta, X_poly_val, yval, 0)
    return(lbd_vec, error_train, error_val)