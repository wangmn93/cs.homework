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

def example(X,y):


    # Fit model/compute optimal parameters beta
    beta_ = mdot(inv(dot(X.T, X)), X.T, y)
    print "Optimal beta:", beta_
    return  beta_

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
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')  # the projection arg is important!
    # ax.scatter(X[:, 0], X[:, 1], y, color="red")
    # ax.set_title("raw data")
    # plt.draw()
    return X,y

def quad_feature(row):
    temp = list(row.tolist())
    feature = list()
    for x in row:
        for t in temp:
            feature.append(x*t)
        temp.remove(x)
    return np.array(feature)



def fit_data(X,y, lambda_, feature):
    if feature == "quad":
        # phi = np.multiply(X,X)
        phi =list()
        for x in X:
            t =np.transpose([np.tile(x, len(x)), np.repeat(x, len(x))])
            t2 = np.prod(t,1)
            t2 = np.unique(t2)
            phi.append(quad_feature(x))
            # pd.Series(list(it.combinations(np.unique(b), 2)))
        phi = np.array(phi)
    else:
        phi = X

    beta_ = mdot(inv(dot(phi.T, phi)+lambda_*np.identity(phi.shape[1])), phi.T, y)
    print "Optimal beta:", beta_.tolist()
    # square_error(phi,beta_,y)
    return beta_

def predict(beta_,X,y,feature):
    # prep for prediction
    X_grid = prepend_one(grid2d(-3, 3, num=30))
    # print "X_grid.shape:", X_grid.shape

    # Predict with trained model
    if feature == 'quad':
        y_grid = mdot(np.multiply(X_grid,X_grid), beta_)
    else:
        y_grid = mdot(X_grid, beta_)
    # print "Y_grid.shape", y_grid.shape

    # vis the result
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')  # the projection part is important
    ax.scatter(X_grid[:, 1], X_grid[:, 2], y_grid)  # don’t use the 1 infront
    ax.scatter(X[:, 1], X[:, 2], y, color="red")  # also show the real data
    ax.set_title("predicted data")
    plt.show()

def square_error(phi,beta_, y):
    # square error
    sq_error = 0
    for i in range(0, y.shape[0], 1):
        predict = dot( beta_,phi[i, :])
        sq_error += pow(y[i] - predict, 2)
    print "Square error", sq_error
    return sq_error

def cross_validation(X,y,lambda_,feature,k):
    if k>1:
        block_size = X.shape[0]/int(k)

        blocks_X = list()
        blocks_y = list()

        remain_X = X
        remain_y = y
        # partition data into blocks_X
        for i in range(0,k,1):
            begin = 0
            end = block_size

            if i < k-1:
                blocks_X.append(remain_X[begin:end,:])
                blocks_y.append(remain_y[begin:end])

                remain_X = remain_X[end:, :]
                remain_y = remain_y[end:]
            else:
                blocks_X.append(remain_X)
                blocks_y.append(remain_y)
        # cross validation
        square_error_list = list()
        beta_list = list()
        for i in range(0, k, 1):
            print "\n%d-th cross validation------------------"%(i+1)
            validation_X = np.array(list(blocks_X).pop(i))
            validation_y = np.array(list(blocks_y).pop(i))

            temp_X = list(blocks_X)
            temp_y = list(blocks_y)
            del temp_X[i]
            del temp_y[i]

            train_X = []
            train_y = []
            for block in temp_X:
                train_X += block.tolist()
            for block in temp_y:
                train_y += block.tolist()
            train_X = np.array(train_X)
            train_y = np.array(train_y)

            beta_ = fit_data(train_X, train_y, lambda_, feature)

            #validation
            if feature =='quad':
                phi = list()
                for x in validation_X:
                    phi.append(quad_feature(x))
                phi = np.array(phi)
            else:
                phi = validation_X
            square_error_list.append(square_error(phi,beta_,validation_y))
            beta_list.append(beta_)
            # predict(beta_, train_X, train_y, feature)


        #
        mean = np.mean(square_error_list)
        var = np.var(square_error_list)
        print "Mean squared error: ",mean
        print "Variance: ",var
        return (beta_list,mean,var)

    elif k==1:
        beta_ = fit_data(X, y, lambda_, feature)
        if feature == 'quad':
            phi = list()
            for x in X:
                phi.append(quad_feature(x))
            phi = np.array(phi)
        else:
            phi = X
        mean =square_error(phi, beta_, y)
        # predict(beta_, X, y, feature)
        print "Mean squared error: ", mean
        print "Variance: ", 0
        return ([beta_],mean,0)

def mean_std_plot(lambda_,mean,std):
    x = np.array(lambda_)
    y = np.array(mean)
    e = np.array(std)

    plt.errorbar(x, y, e, linestyle='None', marker='o')

    plt.show()

def bar_chart():
    N = 5
    men_means = (20, 35, 30, 35, 27)
    men_std = (2, 3, 4, 1, 2)

    ind = np.arange(N)  # the x locations for the groups
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(ind, men_means, width, color='r')

    women_means = (25, 32, 34, 20, 25)
    women_std = (3, 5, 2, 3, 3)
    rects2 = ax.bar(ind + width, women_means, width, color='y')

    # add some text for labels, title and axes ticks
    ax.set_ylabel('Scores')
    ax.set_title('Scores by group and gender')
    # ax.set_xticks(ind + width / 2)
    # ax.set_xticklabels(('G1', 'G2', 'G3', 'G4', 'G5'))

    ax.legend((rects1[0], rects2[0]), ('Men', 'Women'))

    # def autolabel(rects):
    #     """
    #     Attach a text label above each bar displaying its height
    #     """
    #     for rect in rects:
    #         height = rect.get_height()
    #         ax.text(rect.get_x() + rect.get_width() / 2., 1.05 * height,
    #                 '%d' % int(height),
    #                 ha='center', va='bottom')
    #
    # autolabel(rects1)
    # autolabel(rects2)

    plt.show()

if __name__ == "__main__":
    # show, use plt.show() for blocking
    # X, y = import_data("dataLinReg2D.txt")
    X, y = import_data("dataQuadReg2D.txt")
    # X, y = import_data("dataQuadReg2D_noisy.txt")

    # prep for linear reg.
    X = prepend_one(X)
    # print "X.shape:", X.shape
    lambda_list = [i * i for i in range(1, 30, 2)]
    mean_list = list()
    var_list = list()
    for lambda_ in lambda_list:
        result_list = cross_validation(X,y,lambda_,"quad",10)
        print "lambda ",lambda_
        print "\n"
        mean_list.append(result_list[1])
        var_list.append(result_list[2]**(1/2.0) )

    # bar_chart()
    mean_std_plot(lambda_list,mean_list,var_list)
    # beta_ = example(X,y)
    # beta_ = fit_data(X,y,100,"quad")
    # np.array(beta_list[0].tolist())
    # predict(beta_list[0],X,y,'')

