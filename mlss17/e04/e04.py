#!/usr/bin/env python
# encoding: utf-8
"""
This is a mini demo of how to use numpy arrays and plot data.
NOTE: the operators + - * / are element wise operation. If you want
matrix multiplication use ‘‘dot‘‘ or ‘‘mdot‘‘!
"""
import numpy as np
from numpy import dot
from numpy.linalg import inv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D

# 3D plotting
###############################################################################
# Helper functions
import math

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def mdot(*args):
    """Multi argument dot function. http://wiki.scipy.org/Cookbook/MultiDot"""
    return reduce(np.dot, args)

def prepend_one(X):
    """prepend a one vector to X."""
    return np.column_stack([np.ones(X.shape[0]), X])

def grid2d(start, end, num=50):
    """Create an 2D array where each row is a 2D coordinate.
    np.meshgrid is pretty annoying!
    """
    dom = np.linspace(start, end, num)
    X0, X1 = np.meshgrid(dom, dom)
    return np.column_stack([X0.flatten(), X1.flatten()])

    ###############################################################################



def import_data(filename):
    # load the data
    data = np.loadtxt(filename)
    # print "data.shape:", data.shape
    # np.savetxt("tmp.txt", data)  # save data if you want to

    # split into features and labels
    X, y = data[:, :2], data[:, 2]
    # print "X.shape:", X.shape
    # print "y.shape:", y.shape

    # 3D plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')  # the projection arg is important!
    ax.scatter(X[:, 0], X[:, 1], y, color="red")
    ax.set_title("raw data")
    plt.draw()
    plt.show()
    return X,y

def quad_feature(X):
    phi = list()
    for row in X:
        t = np.transpose([np.tile(row, len(row)), np.repeat(row, len(row))])
        t2 = np.prod(t, 1)
        t2 = np.unique(t2)
        temp = list(row.tolist())
        feature = list()
        for x in row:
            for t in temp:
                feature.append(x * t)
            temp.remove(x)
        phi.append(np.array(feature))

    phi = np.array(phi)


    return phi

def predict(beta,X,y,phi,feature=None):
    # prep for prediction
    X_grid = prepend_one(grid2d(-3, 3, num=30))
    # print "X_grid.shape:", X_grid.shape

    temp = np.copy(y)
    if feature:
        y_grid = mdot(quad_feature(X_grid), beta)
        y_grid = [sigmoid(y) for y in y_grid]
    else:
        y_grid = mdot(X_grid, beta)
        y_grid = [sigmoid(y) for y in y_grid]
    # print "Y_grid.shape", y_grid.shape

    # vis the result
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')  # the projection part is important
    ax.scatter(X_grid[:, 1], X_grid[:, 2], y_grid)  # don’t use the 1 infront
    ax.scatter(X[:, 1], X[:, 2], dot(phi,beta), color="red")  # also show the real data
    ax.set_title("predicted data")
    plt.show()


def one_iterate(y,lambda_,beta,phi):

    length =  beta.shape
    #print length
    discriminative_f = dot(phi,beta)
    p = np.array([sigmoid(f) for f in discriminative_f])

    gradient = dot(phi.T, p-y)+2*lambda_*dot(np.eye(len(beta)), beta)
    W = np.diag([pi*(1-pi) for pi in p])

    hessian = mdot(phi.T, W, phi)+2*lambda_*np.eye(len(beta))
    beta -= dot(inv(hessian),gradient)
    print beta


if __name__ == "__main__":
    quad = 1
    X, y = import_data("data2Class.txt")
    X = prepend_one(X)
    phi = np.copy(X)
    if quad:
        phi = quad_feature(X)
    length, width = phi.shape
    beta = np.zeros(width)


    for i in range(20):
        one_iterate(y,.5,beta,phi)
    # print beta
    if quad:
        predict(beta,X,y,phi,1)
    else:
        predict(beta, X, y,phi)
