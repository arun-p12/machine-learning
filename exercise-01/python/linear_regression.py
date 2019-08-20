
import numpy as np

def compute_cost(X, y, theta):
    m = len(y)
    J = 0.0

    hx = np.dot(X, theta)       # m x n   * n x 1   =   m x 1
    temp = (hx - y)             # m x 1
    temp *= temp                # square each element of the array
    J = np.sum(temp) / (2 * m)  # single value
    return(J)

def gradient_descent(X, y, theta, alpha, iter):
    m = len(y)
    J_history = np.reshape(np.zeros(iter), (iter, 1))

    for i in range(0, iter):
        hx = np.dot(X, theta)           # m x n  *  n x 1  =  m x 1
        temp = (hx - y)
        temp2 = np.dot(X.T, temp)       # n x m  * m x 1   = n x 1
        temp2 = (alpha/m) * temp2
        theta = theta - temp2

        # Save the cost J in every iteration
        J_history[i] = compute_cost(X, y, theta)
    return(theta, J_history)

def normal_eqn(X, y):                   # m x n        and m x 1
    theta = np.dot(X.T, X)              # theta = n x n
    theta = np.linalg.inv(theta)        # theta = n x n ..... and the inverse
    theta = np.dot(theta, X.T)          # theta = n x m
    theta = np.dot(theta, y)            # theta = n x 1
    return(theta)

def normalize_features(X):
    X_norm = X
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    X_norm = X_norm - mu
    X_norm = X_norm / sigma
    return(X_norm, mu, sigma)
