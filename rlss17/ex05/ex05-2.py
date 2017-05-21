import numpy as np
import math
from numpy.linalg import inv
import gym

import pylab
actions = [0,1,2]



def mdot(*args):
    """Multi argument dot function. http://wiki.scipy.org/Cookbook/MultiDot"""
    return reduce(np.dot, args)

#input type np.array
def RBF(center, covariance, x):
    d_2 = mdot((x-center).T,inv(np.diag(covariance)),x-center)
    #print d_2
    return math.exp(-d_2/2.)


#input np.array
def getFeature(x,n =4, m=8, range1 =(-1.2,.5),range2=(-0.07,0.07),covariance = np.array([.04, .0004])):
    # generate center according to n and m

    p_centers,v_centers = generateCenter(range1,n,range2,m)
    #p_centers = [-1.3+p*0.4 for p in range(1,5,1)]

    #v_centers = [-0.09+v*0.02 for v in range(1,9,1)]
    #print p_centers
    #print v_centers
    # covariance = np.array([1,1])
    feature = [1]#append one
    for i in p_centers:
        for j in v_centers:
            rbf = RBF(np.array([i,j]),covariance,x)
            feature.append(rbf)
    return np.array(feature)

#input range tuple of 2
# n integer
def generateCenter(range1,n, range2, m):
    range1_span = (range1[1]-range1[0])/(n-1)
    range2_span = (range2[1]-range2[0])/(m-1)
    center1 = [range1[0]+i*range1_span for i in range(n)]
    center2 = [range2[0]+i*range1_span for i in range(m)]
    #print center1
    #print center2
    return center1,center2


def one_iteration(current,next,reward,beta,alpha=.1):
    #todo
    diff = reward + np.dot(beta,next)-np.dot(beta,current)
    gradient = getFeature(x)#???
    beta -= alpha*gradient*diff#???

def chooseAction(feature,beta_list):
    values = [np.dot(beta_list[action],feature) for action in actions]
    return values.index(max(values))

if __name__ == "__main__":
    env = gym.make('MountainCar-v0')
    beta_list = [np.zeros(33),np.zeros(33),np.zeros(33)]#weight vector with zeros
    #covaranice = np.array([.04, .0004])
    for i_episode in range(5):
        observation = env.reset()  # initialize S
        x = np.array([observation[0], observation[1]])
        feature = getFeature(x)
        for t in range(10):
            env.render(close=True)  # turn off animation
            action = chooseAction(feature)
            observation, reward, done, info = env.step(action)  # perform action
            #print RBF(np.array([1,2]),np.array([1,1]),np.array([1,1]))
            #print getFeature(np.array([-.5,0.05]))
            if observation[0] >= .5:  # reward
                reward = 0
            else:
                reward = -1
            next_x = np.array([observation[0],observation[1]])#continuous state space
            next_feature = getFeature(next_x)
            one_iteration(feature,next_feature,reward,beta_list[action])#update weight vector
    #generateCenter((-1.2,.5),4,(-0.07,.07),8)
    #print getFeature(np.array([-1.,0.])).shape