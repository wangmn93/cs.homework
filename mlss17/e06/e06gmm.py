import numpy as np
from matplotlib import pyplot as plt
from math import e, sqrt, pi,exp
from numpy.linalg import inv,det
import random

def mdot(*args):
    """Multi argument dot function. http://wiki.scipy.org/Cookbook/MultiDot"""
    return reduce(np.dot, args)

def import_data(filename):
    # load the data
    data = np.loadtxt(filename)
    X= data[:, :2]
    # 2d plotting
    # fig = plt.figure()
    # ax = fig.add_subplot(111)  # the projection arg is important!
    # ax.scatter(X[:, 0], X[:, 1], color="red")
    # ax.set_title("raw data")
    # plt.show()
    return X

def gaussian_evaluate(mu,sigma,x):
    return (2*pi)**(-.5*len(mu))*det(sigma)**(-.5)*exp(-.5*mdot((x-mu).T,inv(sigma),x-mu))

#E step
def compute_gammas(X,mus,pis,sigmas):
    gammas = [np.zeros(len(mus))] * len(X)
    gammas = np.array(gammas)
    for x, gamma in zip(X, gammas):
        sum = 0
        i = 0
        for mu, sigma, pi in zip(mus, sigmas, pis):
            g = pi * gaussian_evaluate(mu, sigma, x)
            gamma[i] = g
            sum += g
            i += 1
        gamma /= sum
    return gammas
#M step
def update(X,gammas,K,k):
    N_ks = [np.sum(gammas[:, i], axis=0) for i in range(K)]
    new_pis = [N_k / len(X) for N_k in N_ks]
    new_mus = [1 / N_ks[i] * (np.dot(gammas[:, i].T, X)) for i in range(K)]
    new_sigmas = []
    for i in range(K):
        gamma_k = gammas[:, i]
        comp_X = X - np.tile(new_mus[i], (len(X), 1))
        # np.multiply(gamma_k,comp_X)
        new_sigma = np.zeros((k, k))
        for gamma_i, cx in zip(gamma_k, comp_X):
            outer = gamma_i * np.outer(cx, cx.T)
            new_sigma += outer
        new_sigmas.append(1 / N_ks[i] * new_sigma)
    return new_pis, new_mus, new_sigmas

def EM_oneit(X,mus,pis,sigmas,init_gammas=None):
    if init_gammas != None:
        gammas = np.copy(init_gammas)
    else:
        gammas = np.copy(compute_gammas(X, mus, pis, sigmas))
    new_pis, new_mus, new_sigmas = update(X, gammas, K, k)
    return new_pis, new_mus, new_sigmas

# def gmm_oneit(X,mus,pis,sigmas):
#     gammas = [np.zeros(len(mus))]*len(X)
#     gammas = np.array(gammas)
#     for x,gamma in zip(X,gammas):
#         sum = 0
#         i = 0
#         for mu,sigma,pi in zip(mus,sigmas,pis):
#             g = pi*gaussian_evaluate(mu,sigma,x)
#             gamma[i] = g
#             sum += g
#             i+=1
#         gamma /= sum
#     N_ks = [np.sum(gammas[:,i],axis=0) for i in range(len(mus))]
#     new_pis = [ N_k/len(X) for N_k in N_ks]
#     new_mus = [1/N_ks[i]*(np.dot(gammas[:,i].T,X)) for i in range(len(mus))]
#     new_sigmas = []
#     for i in range(len(mus)):
#         gamma_k = gammas[:,i]
#         comp_X = X-np.tile(new_mus[i], (len(X), 1))
#         # np.multiply(gamma_k,comp_X)
#         new_sigma = np.zeros((2,2))
#         for gamma_i,cx in zip(gamma_k,comp_X):
#             outer = gamma_i*np.outer(cx,cx.T)
#             new_sigma+=outer
#         new_sigmas.append(1/N_ks[i]*new_sigma)
#     return new_pis,new_mus,new_sigmas

def plot(X,mus,sigmas,K):
    # 2d plotting
    fig = plt.figure()
    ax = fig.add_subplot(111)  # the projection arg is important!
    for x in X:
        prob = [gaussian_evaluate(mu, sigma, x) for mu, sigma in zip(mus, sigmas)]
        colors = ["red", "green", "blue"]
        ax.scatter(x[0], x[1], color=colors[prob.index(max(prob))])
    ax.set_title("raw data")
    plt.show()

if __name__=="__main__":
    X = import_data("mixture.txt")
    # initialization
    K=3
    k=2
    pis = [1./3 for i in range(K)]
    mus = [X[random.randint(0,len(X)-1)],X[random.randint(0,len(X)-1)],X[random.randint(0,len(X)-1)]]
    sigmas = [np.eye(2),np.eye(2),np.eye(2)]
    init_gammas = []

    for x in X:
        gamma = [0]*K
        gamma[random.randint(0,K-1)] = 1
        init_gammas.append(gamma)
    init_gammas = np.array(init_gammas)

    for i in range(10):
        # gammas = np.copy(compute_gammas(X,mus,pis,sigmas))
        # pis, mus, sigmas = update(X,gammas,K,k)
        if i==0:
            pis, mus, sigmas = EM_oneit(X,mus,pis,sigmas,init_gammas)
        else:
            pis, mus, sigmas = EM_oneit(X, mus, pis, sigmas)

    plot(X,mus,sigmas,K)

