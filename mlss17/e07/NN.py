import numpy as np
from numpy import dot,multiply
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from math import exp,fabs

def sigmoid(x):
    return 1/(1+exp(-x))

element_wise_sigmoid = np.vectorize(sigmoid)

def import_data(filename):
    # load the data
    data = np.loadtxt(filename)
    X= data[:, :-1]
    # 2d plotting
    # fig = plt.figure()
    # ax = fig.add_subplot(111)  # the projection arg is important!
    # ax.scatter(X[:, 0], X[:, 1], color="red")
    # ax.set_title("raw data")
    # plt.show()
    y = data[:,-1]
    return X,y

def plot3Ddata(X,y):
    #3D plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')  # the projection arg is important!
    ax.scatter(X[:, 1], X[:, 2], y, color="red")
    ax.set_title("raw data")
    plt.show()


def initializeWeight(layers):
    weightMats = []
    for i in range(len(layers)-1):
        l = i#current layer
        l_n = i + 1#next layer
        weight = np.random.uniform(-1,1,(layers[l_n],layers[l]))
        weightMats.append(weight)
    return weightMats

#input x is column vector
def forward_one_step(x,weightMat,final):
    if final:
        return dot(weightMat,x)
    else:
        return element_wise_sigmoid(dot(weightMat,x))

#forward propagation
def forward(input,weightMats):
    x = np.array(input).T#convert data into column vector
    x_record = [x]
    for weightMat in weightMats[:-1]:
        x = forward_one_step(x,weightMat,False)
        x_record.append(x)
    #output
    z = forward_one_step(x,weightMats[-1],True)
    # x_record.append(z)
    return z,x_record

def computeDelta(delta,weightMat,x):
    temp = dot(delta, weightMat)
    temp2 = multiply(x, 1 - x).T
    return multiply(temp, temp2)

def backward_one_step(x,delta,weightMat):
    temp = dot(delta,weightMat)
    temp2 = multiply(x,1-x).T
    return multiply(temp,temp2)


#backwawrd propagation
def backward(loss,weightMats,x_record):
    delta = loss

    # weightMat_gradient = []
    #first back propagation
    weightMat_gradient = [dot(delta.T,x_record[-1].T)]
    r_w = weightMats[::-1]
    r_x = x_record[::-1]
    for x,n_x,weightMat in zip(r_x[:-1],r_x[1:],r_w[:-1]):
        # x = r_x[i]
        # weightMat = r_w[i]
        # n_x = r_x[i+1]
        delta = backward_one_step(x,delta,weightMat)
        # print "delta",delta.shape
        # print "n_x",n_x.shape
        weightMat_gradient.append(dot(delta.T,n_x.T))#
    return weightMat_gradient[::-1]

def computeLoss(target,output):
    i = 1 if 1-target*output>0 else 0
    return -target*i




if __name__=="__main__":
    layers = [3,100,1]
    weightMats = initializeWeight(layers)
    sumedWeightMats = list(weightMats)
    print "weight matrix shape"
    for weightMat in weightMats:
        print weightMat.shape
    X,Y = import_data("data2Class_adjusted.txt")
    # print X.shape
    # test = np.zeros(5)

    # print element_wise_sigmoid(test)
    # plot3Ddata(X,y)
    test_x = np.reshape(X[0,:],(1,3))
    test_y = Y[0]
    output,x_record = forward(test_x,weightMats)
    # print output
    loss = computeLoss(test_y,output)
    print "x record shape"
    for x in x_record:
        print x.shape
    # print backward_one_step(x_record[1],2,weightMats[1]).shape
    print "gradient shape"
    weightMats_gradient = backward(np.array([[loss]]),weightMats,x_record)
    for gradient in weightMats_gradient:
        print gradient.shape

    #estimate gradient
    # sumedLoss = 0
    for it in range(500):
        sumedLoss = 0
        for x,y in zip(X,Y):
            x = np.reshape(x,(1,3))
            output,x_record = forward(x,weightMats)
            loss = computeLoss(y,output)

            sumedLoss += max([0,1-output*y])
            weightMats_gradient = backward(np.array([[loss]]), weightMats, x_record)
            for i in range(len(sumedWeightMats)):
                sumedWeightMats[i] -= .05*weightMats_gradient[i]
        print sumedLoss
        weightMats = list(sumedWeightMats)

    # print computeLoss(1,-1)