
import numpy as np
import sys
sys.path.append("../../exercise-02/python/")
import logistic_regression as lr

def sigmoid_gradient(z):
    # g'(z) = d/dz of g(z) = g(z) . (1 - g(z))
    g = lr.sigmoid(z)
    g = g * (1 - g)
    return(np.matrix(g))

# convert theta matrix to a looooong vector, as expected by our cost
# and gradient functions
def unroll_parameters(t1, t2):
    unr_t1 = np.reshape(t1, ((t1.shape[0] * t1.shape[1]), 1))
    unr_t2 = np.reshape(t2, ((t2.shape[0] * t2.shape[1]), 1))
    unrolled = np.concatenate((unr_t1, unr_t2), axis=0)
    return (unrolled)

# convert the long vector back to the theta matrix
# presumes there is only one hidden layer (Theta1 and Theta2)
def roll_parameters(nn_params, hidden_layer_size, input_layer_size, num_labels):
    # Reshape nn_params back into the parameters Theta1 and Theta2,
    # these are the weight matrices for our 2 layer neural network
    t1_bndry = hidden_layer_size * (input_layer_size + 1)
    (t1_p, t1_q) = (hidden_layer_size, (input_layer_size + 1))
    Theta1 = np.reshape(nn_params[0:t1_bndry], (t1_p, t1_q))
    # t2 starts from 1 more than where t1 ends...
    # use python's half-openness to our advantage
    t2_bndry = t1_bndry
    (t2_p, t2_q) = num_labels, (hidden_layer_size + 1)
    Theta2 = np.reshape(nn_params[t1_bndry:], (t2_p, t2_q))
    return(Theta1, Theta2)

# null out the zeroth column
def vectorize_theta(theta):
    from copy import deepcopy
    theta_reg = deepcopy(theta)
    theta_reg[:, 0] = 0  # null out the zeroth column
    return(theta_reg)

def compute_cost_reg_nn(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lbd):
    m = X.shape[0]
    (Theta1, Theta2) = roll_parameters(nn_params, hidden_layer_size, input_layer_size, num_labels)

    # compute the hypothesis
    X1 = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
    ax = lr.sigmoid(np.dot(X1, Theta1.T))

    # prefix a0, and then repeat
    # the two steps if there are additional hidden layers
    ax = np.concatenate((np.ones((ax.shape[0], 1)), ax), axis=1)
    ax = lr.sigmoid(np.dot(ax, Theta2.T))

    # now we're at the final / output layer
    # convert y to a collection of vectors... in the oneVsAll format
    # each vector has a 1 for its classification, and a 0 for all else
    y_all = np.zeros((m, num_labels))
    for i in  np.arange(1, num_labels+1):
        pos = np.where(y == i)[0]
        if(i == 10): y_all[pos, 9] = 1
        else: y_all[pos, i-1] = 1

    # calculate the cost function, by retaining the classified value, and
    # discarding all the rest.... element-wise multiplication by 1s and 0s will ensure that
    J = sum(sum((-y_all * np.log(ax)) - ((1 - y_all) * np.log(1 - ax)))) / m

    # compute the regularization
    Theta1_Reg = vectorize_theta(Theta1)
    Theta2_Reg = vectorize_theta(Theta2)
    reg = (lbd / (2 * m)) * (sum(sum(Theta1_Reg * Theta1_Reg)) + sum(sum(Theta2_Reg * Theta2_Reg)))

    # update the cost function
    J = J + reg
    return(J)

def gradient_descent_nn(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lbd):
    # an now, onto back propagation
    # but, first lets compute gradient using forward propagation

    m = X.shape[0]
    X1 = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
    (Theta1, Theta2) = roll_parameters(nn_params, hidden_layer_size, input_layer_size, num_labels)

    Theta1_grad = np.zeros(Theta1.shape)
    Theta2_grad = np.zeros(Theta2.shape)

    for t in range(m):
        # step #1 :: perform forward propagation, stepping thru each layer
        a1 = X1[t, :]       # X1 already has X0 (the ones column) added

        z2 = np.dot(a1, Theta1.T)
        a2 = lr.sigmoid(z2)

        a2 = np.insert(a2, 0, 1)     # insert a 1 at the beginning of the np.array
        z3 = np.dot(a2, Theta2.T)
        a3 = lr.sigmoid(z3)

        # step #2 :: calculate the resultant error
        ytemp = y[t, :]                     # save the labels
        y3 = np.zeros((1, num_labels))      # create a column of labels for oneVsAll

        if(ytemp[0] < 10):
            y3[0, ytemp[0]-1] = 1           # update the appropriate column based on the label
        else: y3[0, 9] = 1                  # account for '0' is represented as 10

        delta_3 = a3 - y3                    # compute the error at the final layer
        delta_3 = delta_3.T                 # transform into a vector

        # step #3 :: propagate back, and calculate the gradient error
        # δ(2) =  Θ(2) T δ(3). ∗ g′(z(2))
        # don't have to worry about delta_1 since that's the input layer... no error
        # transform the output of the sigmoidGradient into a column vector
        # Add the bias node to z2, while calculating the gradient.
        delta_2 = np.dot(Theta2.T, delta_3)
        delta_2 = delta_2 * sigmoid_gradient(np.insert(z2, 0, 1))
        # strip out the delta for the 0th unit... we had added that unit in
        delta_2 = delta_2[1:, 0]

        # step #4 :: accumulate the gradient
        # ∆(l) = ∆(l) + δ(l+1)(a(l))T
        # add in the bias for a2 ... but, not for a1 since it is the input layer
        #Theta2_grad = Theta2_grad + np.dot(delta_3, np.matrix(a2))
        #Theta1_grad = Theta1_grad + np.dot(delta_2, np.matrix(a1))

        Theta2_grad = Theta2_grad + np.dot(delta_3, np.reshape(a2, (1, a2.shape[0])))
        Theta1_grad = Theta1_grad + np.dot(delta_2, np.reshape(a1, (1, a1.shape[0])))

    Theta1_grad = Theta1_grad / m
    Theta2_grad = Theta2_grad / m

    # regularized gradient
    # add the term (lambda / m) .* Theta(layer) to each term
    # no need to regularize for the first layer
    Theta1_Reg = vectorize_theta(Theta1)
    Theta2_Reg = vectorize_theta(Theta2)
    Theta1_grad = Theta1_grad + ((lbd / m) * Theta1_Reg)
    Theta2_grad = Theta2_grad + ((lbd / m) * Theta2_Reg)

    # unroll the gradients
    grad = unroll_parameters(Theta1_grad, Theta2_grad)
    return(grad)

def rand_initialize_weights(L_in, L_out):
    # Randomly initialize the weights to small values
    epsilon_init = 0.12
    data = np.random.random(L_out * (L_in + 1))
    W = np.reshape(data, (L_out, 1 + L_in)) * 2 * epsilon_init - epsilon_init
    return(W)

def compute_numerical_gradient(theta, input_layer_size, hidden_layer_size, num_labels, X, y, lbd):
    numgrad = np.zeros(theta.shape)
    perturb = np.zeros(theta.shape)
    e = 1e-4
    for p in range(theta.size):
        # Set perturbation vector
        perturb[p] = e
        loss1 = compute_cost_reg_nn((theta - perturb), input_layer_size, hidden_layer_size, num_labels, X, y, lbd)
        loss2 = compute_cost_reg_nn((theta + perturb), input_layer_size, hidden_layer_size, num_labels, X, y, lbd)
        # Compute Numerical Gradient
        numgrad[p] = (loss2 - loss1) / (2 * e)
        perturb[p] = 0
    return(numgrad)

# Initialize the weights of a layer with fan_in incoming connections
# and fan_out outgoing connections using a fixed set of values
def debug_initialize_weights(fan_out, fan_in):
    # W should be set to a matrix of size(1 + fan_in, fan_out) as
    # the first row of W handles the "bias" terms
    W = np.zeros((fan_out, 1 + fan_in))
    # Initialize W using "sin", this ensures that W is always of the same values
    #W = np.reshape(np.sin[0:W.size], W.shape) / 10
    return(W)

def check_gradients(lbd=0):
    input_layer_size = 3
    hidden_layer_size = 5
    num_labels = 3
    m = 5

    # We generate some 'random' test data
    Theta1 = debug_initialize_weights(hidden_layer_size, input_layer_size)
    Theta2 = debug_initialize_weights(num_labels, hidden_layer_size)
    # Reusing debugInitializeWeights to generate X
    X = debug_initialize_weights(m, input_layer_size - 1)
    y = np.vstack(1 + np.mod(range(1, m+1), num_labels))

    # unroll the parameters
    nn_params = unroll_parameters(Theta1, Theta2)

    grad = gradient_descent_nn(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lbd)
    numgrad = compute_numerical_gradient(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lbd)

    # Visually examine the two gradient computations.
    # The two columns should be very similar.
    #print("Gradient examination\n", np.concatenate((numgrad, grad, (numgrad - grad)), axis=1))

def optimizer_func_nn(theta, input_layer_size, hidden_layer_size, num_labels, X, y, lbd):
    # https://github.com/arturomp/coursera-machine-learning-in-python/blob/master/mlclass-ex4-004/mlclass-ex4/ex4.py
    import scipy.optimize as op

    opts = {'disp': True, 'maxiter': 50}
    mthd = 'TNC'

    result = op.minimize(fun=compute_cost_reg_nn, x0=theta, method=mthd, options=opts, jac=gradient_descent_nn,
                         args=(input_layer_size, hidden_layer_size, num_labels, X, y, lbd))
    return(result.x)

    # working #2
    #result = op.optimize.fmin_bfgs(f=compute_cost_reg_nn, x0=theta,
    #                               args=(input_layer_size, hidden_layer_size, num_labels,X, y, lbd),
    #                               maxiter = 400, fprime = gradient_descent_nn)
    #return (result)

'''
def predict(Theta1, Theta2, X):
    m = X.shape[0]
    num_labels = Theta2.shape[0]

    X1 = np.concatenate((np.ones((m, 1)), X), axis=1)
    h1 = lr.sigmoid(np.dot(X1, Theta1.T))
    h1 = np.concatenate((np.ones((m, 1)), h1), axis=1)
    h2 = lr.sigmoid(np.dot(h1, Theta2.T))
    p = h2.max(axis=1)      # find the max per row
    return(p)
'''