import numpy as np
import math
from numpy.linalg import inv
import gym
import random

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
    center2 = [range2[0]+i*range2_span for i in range(m)]
    #print center1
    #print center2
    return center1,center2

def chooseAction(feature,beta_list):
    action = random.choice(actions)#random tile breaker
    maxValue = np.dot(feature,beta_list[action])
    for a in actions:
        if np.dot(feature,beta_list[a])>maxValue:
            maxValue = np.dot(feature,beta_list[a])
            action = a
    return action

if __name__ == "__main__":
    env = gym.make('MountainCar-v0')
    beta_list = [np.zeros(33),np.zeros(33),np.zeros(33)]#weight vector with zeros
    e_list = [np.zeros(33),np.zeros(33),np.zeros(33)]#eligible trace
    gamma = .99
    lambda_ = .7
    alpha = .001
    # for i in range(1):
    #     observation = env.reset()
    #     print observation
    #     x = np.array([observation[0], observation[1]])
    #     feature = getFeature(x)
    #     print feature

    old_beta_list = np.copy(beta_list)

    covariance = np.array([.04, .0004])
    for i_episode in range(200):
        print i_episode,
        observation = env.reset()  # initialize S
        x = np.array([observation[0], observation[1]])
        feature = getFeature(x)
        for t in range(5000):
            #print t
            env.render(close=True)  # turn off animation

            action = chooseAction(feature,beta_list)#greedy
            #print feature
            #print action
            observation, reward, done, info = env.step(action)  # perform action

            #print RBF(np.array([1,2]),np.array([1,1]),np.array([1,1]))
            #print getFeature(np.array([-.5,0.05]))
            if observation[0] >= .5:  # reward
                reward = 0
            else:
                reward = -1
            next_x = np.array([observation[0],observation[1]])#continuous state space
            next_feature = getFeature(next_x)
            e_list[action] += feature
            if observation[0]>=.5:
                #different iteration
                #one_iteration(feature, next_feature, reward, beta_list[action], e_list[action])
                diff = reward-np.dot(feature,beta_list[action])
                beta_list[action] += alpha*e_list[action]*diff
                print "episode finished ",t,
                break
            maxAction = chooseAction(next_feature,beta_list)
           # one_iteration(feature,next_feature,reward,beta_list[maxAction],e_list[action])#update weight vector
            #gradient decent
            diff = reward+gamma*np.dot(next_feature,beta_list[maxAction])-np.dot(feature,beta_list[action])
            beta_list[action] += alpha*e_list[action]*diff
            #print diff
            e_list[action]*=gamma*lambda_
            feature = next_feature
       # print beta_list
    #generateCenter((-1.2,.5),4,(-0.07,.07),8)
    #print getFeature(np.array([-1.,0.])).shape
        difference = 0
        for k in range(3):
            difference += np.dot(old_beta_list[k] - beta_list[k], old_beta_list[k] - beta_list[k])
        print "diff of beta ",math.sqrt(difference)
        old_beta_list = np.copy(beta_list)