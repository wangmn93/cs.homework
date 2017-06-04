import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse.linalg as sla
from os import listdir
from os.path import isfile, join

if __name__ == "__main__":
    #read img
    mypath = "yalefaces"
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f)) and f!="Readme.txt"]
    X =[]
    for f in onlyfiles:
        img = plt.imread("yalefaces/%s"%f)
        # print img.shape
        X.append(img.flatten())

    X = np.array(X) # data
    mu = np.mean(X,0)# mean image
    center = np.tile(mu,(165,1))# build matrix
    X_c = X - center # centering images
    # print center.shape
    # SVD
    error_list = []
    for i in range(1,13,1):

        p = i*5
        u, s, vt = sla.svds(X_c,k=p)
        # print vt.shape
        Z = np.dot(X_c,vt.T)
        X_prime = center + np.dot(Z,vt)
        # print X_prime.shape
        error = 0
        for origin, after in zip(X, X_prime):
            # plt.imshow(np.reshape(after,(243, 320)),cmap="gray")
            # plt.show()
            error+= np.dot(origin-after,origin-after)
        print "Reconstruction error ",error
        error_list.append(error)
    plt.figure()
    plt.plot([i*5 for i in range(1,13,1)],error_list,'r',label = "Reconstruction error ")
    plt.legend(loc='upper right')
    # red_patch = mpatches.Patch(color='red', label='Reconstruction error')
    # plt.legend(handles=[red_patch])
    plt.show()