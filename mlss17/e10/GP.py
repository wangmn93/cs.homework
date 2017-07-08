import numpy as np
from numpy.linalg import norm,inv
from numpy import dot
from math import exp,sqrt
from matplotlib import pyplot as plt

#kernel
def gamma_exp(x,x_p,l =.2, gamma = 2):
    return exp(-(norm((x-x_p)/l)**gamma))

def mean(x,kernel,X,y,lambda_):
    func = np.vectorize(kernel)
    kappa = func([x],X)
    K = [func(e,X) for e in X]
    K = np.array(K)
    return dot(dot(kappa,inv(K+np.eye(len(X))*lambda_)),y)


def var(x,kernel,X,sigma,lambda_):
    func = np.vectorize(kernel)
    kappa = func([x], X)
    K = [func(e, X) for e in X]
    K = np.array(K)
    return sigma**2/lambda_*(kernel(x,x)-dot(dot(kappa,inv(K+np.eye(len(X))*lambda_)),kappa.T))

if __name__ == "__main__":
    include_sigma = 0
    sigma = .1
    data = np.array([[-.5,.3],[.5,.1]])
    y = data[:,1]
    X = data[:,0]
    xs = [i*0.01 for i in range(-100,101,2)]
    ys = [mean(x,gamma_exp,X,y,sigma**2) for x in xs]
    vars = [var(x,gamma_exp,X,sigma,sigma**2) for x in xs]
    upper_bound = [y+sqrt(var+include_sigma*sigma**2) for y,var in zip(ys,vars)]
    lower_bound = [y-sqrt(var+include_sigma*sigma**2) for y,var in zip(ys,vars)]
    plt.figure()
    plt.plot(xs,ys)
    plt.plot(xs, upper_bound,c='r')
    plt.plot(xs, lower_bound,c='r')
    plt.show()