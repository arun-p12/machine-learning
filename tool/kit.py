
# http://cs231n.github.io/python-numpy-tutorial/
# https://machinelearningmastery.com/index-slice-reshape-numpy-arrays-machine-learning-python/

import numpy as np
import matplotlib.pyplot

def load_file(f, delim=","):
    import csv
    reader = csv.reader(open(f, "r"), delimiter=delim)
    x = list(reader)
    result = np.array(x).astype("float")
    return(result)

def load_file2(f, delim=",", header=True, case=False):
    r = np.genfromtxt(f, delimiter=',', names=header, case_sensitive=case)
    return(r)

def load_file3(f, delim=",", head=None):
    import pandas as pd
    df = pd.read_csv(f, sep=delim, header=head)
    return(df)

def plot_data(x_list, y_list, x_label="X axis", y_label="Y axis", ttable=[]):
    params = {'s':7, 'marker':'o', 'label':'Figure 1'}

    # differentiate between various data types...
    color = {0:'b', 1:'g', 2:'r', 3:'c', 4:'m', 5:'y', 6:'k'}
    if(len(ttable) == 0): ttable = [0] * len(x_list)

    for i in range(0, len(x_list)):
        x = x_list[i]
        y = y_list[i]
        matplotlib.pyplot.scatter(x, y, c=color[ttable[i]], s=7)

    matplotlib.pyplot.xlabel(x_label)
    matplotlib.pyplot.ylabel(y_label)
    matplotlib.pyplot.show()

def plot_data_classify(X, y, x_label="X axis", y_label="Y axis"):
    # http://jb-blog.readthedocs.io/en/latest/posts/0012-matplotlib-legend-outdide-plot.html
    # https://python-graph-gallery.com/scatter-plot/
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    matplotlib.pyplot.scatter(X[pos, 0], X[pos, 1], c='g', s=7, label='Admitted')
    matplotlib.pyplot.scatter(X[neg, 0], X[neg, 1], c='r', s=7, marker='d', label="Not Admitted")
    matplotlib.pyplot.xlabel(x_label)
    matplotlib.pyplot.ylabel(y_label)
    matplotlib.pyplot.legend(loc='upper right', shadow=True)
    matplotlib.pyplot.legend(loc=9, bbox_to_anchor=(0.5, -0.1*-11), ncol=2)
    matplotlib.pyplot.show()

def plot_scatter_plot(data, opts):
    data = [{'x': [1, 2, 3, 4, 5], 'y': [1, 2, 3, 4, 5]}, {'x': [2, 4, 6, 8, 10, 12], 'y': [5, 7, 9, 11, 13, 15]}]
    for i in range(len(data)):
        matplotlib.pyplot.scatter(data[i]['x'], data[i]['y'])
    matplotlib.pyplot.show()

def plot_line_plot(data, opts):
    data = [{'x': [1, 2, 3, 4, 5], 'y': [1, 2, 3, 4, 5]},
            {'x': [2, 4, 6, 8, 10, 12], 'y': [5, 7, 9, 11, 13, 15]},
            {'x': [-1, -2, -3, -4, -5], 'y': [3, 6, 9, 12, 15]}
            ]
    for i in range(len(data)):
        matplotlib.pyplot.plot(data[i]['x'], data[i]['y'])
    matplotlib.pyplot.show()
