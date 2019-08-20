
# https://matplotlib.org/users/image_tutorial.html
# https://matplotlib.org/users/colormaps.html

# exmple usage: plot_colormaps(get_image_as_array('img_hunt.png'))

import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.append("../../exercise-02/python")
import logistic_regression as lr

# load an image and convert to a numpy array.
# two different options, using two separate modules listed.
def get_image_as_array(f, lib=''):
    if(lib == 'PIL'):
        from PIL import Image
        img = Image.open(f).convert("L")
        img = np.asarray(img)
    else:
        import matplotlib.image as mpimg
        img = mpimg.imread(f)
    return(img)

# show how the colormap can be set (two different ways)
# also highlights how to define the title, and the colorbar
def plot_colormaps(img):
    cmaps = ['plasma', 'Oranges', 'BuPu', 'YlGnBu', 'gray', 'spring', 'hot', 'Spectral', 'rainbow']
    for c in cmaps:
        plt.imshow(img, cmap=c)     # alternative :: plt.imshow(img).set_cmap(c)
        plt.title(c)
        plt.colorbar()
        plt.show()

# display the set of images in a grid.
def display_data(images, eg_wd=''):
    import math
    # Set width of examples automatically if not passed in
    if(eg_wd == ''): eg_wd = int(math.sqrt(images.shape[1]))

    (m, n) = images.shape       # number of images and pixels per image
    eg_ht = int(n / eg_wd)      # calculate height based on width

    # Compute number of items to display
    disp_rows = int(np.floor(math.sqrt(m)))     # round down
    disp_cols = int(np.ceil(m / disp_rows))     # round up
    pad = 1                                     # padding betwee images

    # setup a blank display
    disp_array = - np.ones((pad + disp_rows * (eg_ht + pad),
                           pad + disp_cols * (eg_wd + pad)))

    current = 0
    for j in range(0, disp_rows):
        for i in range(0, disp_cols):
            if current < m:
                # get the patch by calculating the pixel points for each image
                row = [pad + (j * (eg_ht + pad)) + x for x in range(0, eg_ht)]
                col = [pad + (i * (eg_wd + pad)) + x for x in range(0, eg_wd)]

                # scale down the intensity of each image .... make them look similar
                # each row is an image, and columns represents its 400 pixles
                # get the max value in each row, and normalize pixel values b/w 0 and 1
                # while doing so, also convert the pixels to a eg_ht x eg_wd grid (20x20)
                max_val = max(abs(images[current, :]))
                new_shape = np.reshape(images[current, :], (eg_ht, eg_wd)) / max_val

                disp_array[row[0]:row[-1]+1, col[0]:col[-1]+1] =  new_shape
                current += 1

    plt.imshow(disp_array.T, cmap='gray')
    plt.show()

# account for 1-indexed octave, which put digit zero in the 10th column
# reset y to have a 1 for the digit of interest, and a 0 for all other digits
def new_y(y, i):
    # initialize everything to zero .. the vsAll set
    # Identify the positions in the original y vector that match digit-of-interest
    # update those positions with 1
    new_y = np.zeros(y.shape)
    pos = np.where(y == i)[0]       # [0] --> rows ; [1] --> columns
    new_y[pos, :] = 1
    return(new_y)

# for each of the classfied outputs (digits 0 thru 9), find the optimal gradient
# iterate thru each digit, with the updated y (label), and store results in all_theta
# the results are stored per row for each digit, where each column of a row represents
# a 'feature' ... in the handwriting example there are 400+1 features!
def one_vs_all(X, y, num_labels, lbd):
    (m, n) = X.shape
    X = np.concatenate((np.ones((m, 1)), X), axis=1)    # add in the 1-vector
    initial_theta = np.zeros((n+1, 1))                  # initialize fitting parameters

    all_theta = np.zeros((num_labels, n+1))             # the final output of the classifier

    for i in np.arange(1, num_labels+1):                # 0 is represented as 10
        y_ova = new_y(y, i)
        result = lr.optimizer_func_test(initial_theta, X, y_ova, lbd)
        result = result.reshape((1, n+1))
        if(i < 10): all_theta[i] = result
        else: all_theta[0] = result
    return(all_theta)

# using the optimized gradient, compute the probability of each input being one of the digits
# The one with the highest probability wins, and represents our predicted result.
def predict_one_vs_all(theta, X):
    (p, q) = X.shape
    num_labels = theta.shape[0]

    X = np.concatenate((np.ones((p, 1)), X), axis=1)    # add in the 1-vector
    prob = lr.sigmoid(np.dot(X, theta.T))
    pred = np.zeros((p, 1))
    for i in range(p):
        col = prob[i, :].argmax()
        if(col): pred[i] = col
        else: pred[i] = 10
    return(pred)

# prediction using the neural network model
def predict_nn(Theta1, Theta2, X):
    p = X.shape[0]
    num_labels = Theta2.shape[0]

    X = np.concatenate((np.ones((p, 1)), X), axis=1)
    prob = lr.sigmoid(np.dot(X, Theta1.T))
    t = prob.shape[0]
    prob = np.concatenate((np.ones((t, 1)), prob), axis=1)
    prob = lr.sigmoid(np.dot(prob, Theta2.T))

    pred = np.zeros((p, 1))         # our prediction vector

    # step thru each digit in the training set, and get the column with highest probability
    # the column number corresponds to the digit ... with 0 as column 10
    # note that 1 is in the 0th column, 2 in the 1st, and 0 and in the 9th [[ 0 ... 9]
    for i in range(p):
        col = prob[i, :].argmax()
        if (col == 9): pred[i] = 0
        else: pred[i] = col+1
    return(pred)

'''
def cost_function_reg(theta, X, y, lbd):
    import sys
    sys.path.append("../ex2/")
    import logistic_regression as lr

    J = lr.compute_cost_reg(theta, X, y, lbd)        # code from ex2
    return(J)

def gradient_descent_reg(theta, X, y, lbd):
    import sys
    sys.path.append("../ex2/")
    import logistic_regression as lr

    grad = lr.gradient_descent_reg(theta, X, y, lbd)        # code from ex2
    return(grad)
'''

'''
def fix_y(y, i=10, j=0):
    # presuming y is a column vector, get the row #s where y = 10
    # replace it with 0s, to account for the handwritten digit 0
    # being represented as a 10 to account for octave one-indexing
    #new_y = y[:]
    from copy import deepcopy
    new_y = deepcopy(y)
    pos = np.where(new_y == i)[0]
    new_y[pos, :] = j
    return(new_y)

def one_vs_all_bad(X, y, num_labels, lbd):
    (m, n) = X.shape

    X = np.concatenate((np.ones((m, 1)), X), axis=1)    # add in the 1-vector
    initial_theta = np.zeros((n+1, 1))                  # initialize fitting parameters
    all_theta = np.zeros((num_labels, n+1))             # the final output of the classifier

    for i in np.arange(0, num_labels):                     # 0 is represented as 0
        result = lr.optimizer_func_test(initial_theta, X, y, lbd)
        result = result.reshape((1, n+1))
        all_theta[i] = result
    return(all_theta)

def predict_one_vs_all_bad(theta, X):
    (p, q) = X.shape
    num_labels = theta.shape[0]

    X = np.concatenate((np.ones((p, 1)), X), axis=1)    # add in the 1-vector
    prob = lr.sigmoid(np.dot(X, theta.T))
    pred = np.zeros((p, 1))
    for i in range(p):
        col = prob[i, :].argmax()
        #if(col): pred[i] = col
        #else: pred[i] = 10
        pred[i] = col
    return(pred)

'''