import numpy as np
from os import listdir
from os.path import isfile, join
from matplotlib import pyplot as plt
import scipy.sparse.linalg as sla
import random

#read img
def readImg(folder):
    mypath = folder
    onlyfiles = [f for f in listdir(mypath) if f!="Readme.txt"]
    X =[]
    # print len(onlyfiles)
    # print len(listdir(mypath))
    for f in onlyfiles:
        img = plt.imread("%s/%s"%(folder,f))
        X.append(img[:,:,0].flatten())#only take first channel
    X = np.array(X) # data
    return X

def knn_oneIter(centers,K,X,k):
    clusters = [[] for i in range(K)]
    for x in X:
        dist = [distance(x,center) for center in centers]
        clusterIndex = dist.index(min(dist))
        clusters[clusterIndex].append(x)
    #compute new centers
    newCenters = []
    error_list = []
    for cluster in clusters:
        #print "numOfCluster",len(cluster)
        if cluster!=[]:
            temp = np.array(cluster)
            newCenter = np.mean(temp,0)

            # plt.imshow(newCenter.reshape((243, 160, 4)))
            # plt.show()
            newCenters.append(newCenter)
            error = 0
            for data in cluster:
                error += distance(newCenter,data)
            error_list.append(error)
        else:
            newCenters.append(np.random.rand(k))
            error_list.append(float("inf"))
    #compute error

    return newCenters,error_list,clusters

def distance(v1,v2):
    dist = np.dot(v1-v2,v1-v2)
    return dist

def knn(init,X,K,k,it):
    old_centers = list(init)
    clusters= []
    for i in range(it):
        centers,error_list,clusters = knn_oneIter(old_centers,K,X,k)
        old_centers = list(centers)
        print "error ",error_list
        # for cluster in clusters:
        #     plt.imshow(np.reshape(cluster[0],(243, 160, 4)))
        #     plt.show()
    return old_centers,clusters

def randomInit(K,l,hi):
    init = []
    for i in range(K):
        init.append(np.random.randint(hi,size=l))
    return init

def PCA(X,p):
    mu = np.mean(X, 0)  # mean image
    center = np.tile(mu, (len(X), 1))  # build matrix
    X_c = X - center  # centering images
    u, s, vt = sla.svds(X_c, k=p)
    # print vt.shape
    Z = np.dot(X_c, vt.T)
    #X_prime = center + np.dot(Z, vt)
    return center[0],Z,vt

if __name__=="__main__":
    X = readImg("yalefaces_cropBackground")
    k = 243 * 160
    K = 4

    #PCA
    center, Z, vt = PCA(X, 20)
    X = Z
    k = 20

    init = randomInit(K,k,255)
    centers,clusters = knn(init,X,K,k,10)

    for mu,cluster in zip(centers,clusters) :
        mu = center + np.dot(mu, vt)
        plt.figure()
        plt.imshow(mu.reshape((243, 160)),cmap="gray")
        plt.figure()
        cluster[0] = center + np.dot(cluster[0], vt)
        plt.imshow(cluster[0].reshape((243, 160)),cmap="gray")
        plt.show()
