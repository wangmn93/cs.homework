import numpy as np
from numpy import dot,multiply
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from math import exp,fabs

def sigmoid(x):
    return 1/(1+exp(-x))

element_wise_sigmoid = np.vectorize(sigmoid)

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

def plot3Ddata(X,y,title,block):
    #3D plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')  # the projection arg is important!
    ax.scatter(X[:, 1], X[:, 2], y, color="red")
    ax.set_title(title)
    plt.show(block=block)

def createSurface(range,n,function,parameter):
    X_grid = prepend_one(grid2d(range[0], range[1], num=n))
    y = []
    for x in X_grid:
        output,weightMats = function(x,parameter)
        y.append(output)
    return X_grid,np.array(y)

def initializeWeight(layers,zero):
    weightMats = []
    for i in range(len(layers)-1):
        l = i#current layer
        l_n = i + 1#next layer
        if zero:
            weight = np.zeros((layers[l_n],layers[l]))
        else:
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
    return z,x_record

def backward_one_step(x,delta,weightMat):
    temp = dot(delta,weightMat)
    temp2 = multiply(x,1-x).T
    return multiply(temp,temp2)

#backwawrd propagation
def backward(loss,weightMats,x_record):
    #first back propagation
    delta = loss
    weightMat_gradient = [dot(delta.T,x_record[-1].T)]
    r_w = weightMats[::-1]
    r_x = x_record[::-1]
    for x,n_x,weightMat in zip(r_x[:-1],r_x[1:],r_w[:-1]):
        delta = backward_one_step(x,delta,weightMat)
        weightMat_gradient.append(dot(delta.T,n_x.T))#
    return weightMat_gradient[::-1]

def computeLoss(target,output):
    i = 1 if 1-target*output>0 else 0
    return -target*i

if __name__=="__main__":
    layers = [3,100,1]
    weightMats = initializeWeight(layers,False)
    sumedWeightMats = list(weightMats)
    X,Y = import_data("data2Class_adjusted.txt")
    plot3Ddata(X,Y,"raw data",False)
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
        print it,sumedLoss[0]
        weightMats = list(sumedWeightMats)
    X_surface,predict_y = createSurface((-3,3),50,forward,weightMats)
    # print X_surface.shape
    # print predict_y.shape
    plot3Ddata(X_surface,element_wise_sigmoid(predict_y),"predict surface",True)