import numpy as np
import math
from numpy.linalg import inv
import gym
import random
import math
from matplotlib import  pyplot as plt

actions = [0,1,2]

env = gym.make('MountainCar-v0')

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

def getFeature2(x,action):
    featureList = []
    for i in range(3):

        if i == action:
            featureList.append(getFeature(x))
        else:
            featureList.append(np.zeros(33))

    return np.concatenate((featureList[0],featureList[1],featureList[2]))

#input range tuple of 2
# n integer
def generateCenter(range1,n, range2, m):
    range1_span = (range1[1]-range1[0])/(n-1)
    range2_span = (range2[1]-range2[0])/(m-1)
    center1 = [range1[0]+i*range1_span for i in range(n)]
    center2 = [range2[0]+i*range2_span for i in range(m)]
    return center1,center2


def chooseAction(feature,beta_list):
    action = random.choice(actions)#random tile breaker
    maxValue = np.dot(feature,beta_list[action])
    for a in actions:
        if np.dot(feature,beta_list[a])>maxValue:
            maxValue = np.dot(feature,beta_list[a])
            action = a
    return action

def getAction(actions,weight,state,feature_function):
    values = [np.dot(weight,feature_function(state,action))for action in actions]
    return values.index(max(values))

def transition(cState,action):
    #cState = env.observation_space.sample()
    #print cState,
    actions2 = [-1, 0, 1]
    a = actions2[action]
    #print action,

    p_dot = cState[1] + 0.001 * a - 0.0025 * math.cos(3 * cState[0])
    p_dot = max(min(p_dot, 0.07), -0.07)  # limit speed
    nState = [cState[0] + p_dot, p_dot]
    if cState[0] + p_dot <= -1.2:
        nState[1] = 0
    nState[0] = max(min(nState[0], 0.6), -1.2)
    # env.observation_space.contains()
    #print nState,
    if env.observation_space.contains(np.array(nState)):
        return np.array(nState)
    else:
        print "invalid state",nState

def sample(weight,feature_function,transition,reward_function,getAction,n):
    exp_list = []
    for i in range(n):
        state = env.observation_space.sample()
        action = env.action_space.sample()
        feature = feature_function(state,action)
        nState = transition(state,action)
        reward = reward_function(nState)
        nAction = getAction([0,1,2],weight,nState,feature_function)
        nFeature = feature_function(nState,nAction)
        exp_list.append((feature,reward,nFeature))
        print  state,action,reward,nState,nAction
    return exp_list


def sample2(env):
    state = env.observation_space.sample()
    action = env.action_space.sample()
    return state,action

def sample3(feature_function,weight,getAction):
    exp_list = []
    observation = env.reset()
    for t in range(10000):
        env.render(close=True)
        one_exp = []

        action = np.random.choice([0,1,2])
        one_exp.append(feature_function(observation,action))
        observation, reward, done, info = env.step(action)
        one_exp.append(action)
        one_exp.append(reward)

        nAction = getAction([0,1,2],weight,observation,feature_function)
        one_exp.append(feature_function(observation,nAction))
        one_exp.append(nAction)

        exp_list.append(one_exp)
        if done:
            env.reset()
    return exp_list

def getReward(next_state):
    if next_state[0] >= .5:  # reward
        reward = 0
    else:
        reward = -1
    return reward



def LSTDQ(getAction,old_weight,feature_function,transition_function,reward_function,k,samples,gamma=.99):
    A_tilde = np.zeros((k, k))
    b_tilde = np.zeros(k)
    for i in range(samples):
        state,action = sample2()
        feature = feature_function(state, action)
        next_state = transition_function(state, action)
        next_action = getAction([0,1,2],old_weight,next_state,feature_function)
        # reward = getReward(next_state)
        reward = reward_function(next_state)
        # expect_feature = .9 * getFeature(next1, pi[next1]) + .1 * getFeature(next2, pi[next2])
        # expect_reward = .9 * getReward(next1) + .1 * getReward(next2)
        diff = feature - gamma * feature_function(next_state, next_action)
        # next_state = transition(state,action)
        # reward = getReward(next_state)
        A_tilde += np.outer(feature, diff.T)
        b_tilde += feature * reward
    weight = np.dot(inv(A_tilde), b_tilde)
    return weight

def LSTD(exp_list,k,gamma=.99):
    A_tilde = np.zeros((k, k))
    b_tilde = np.zeros(k)
    for one_exp in exp_list:
        feature = one_exp[0]
        reward = one_exp[1]
        nFeature = one_exp[2]
        #update
        diff = feature - gamma * nFeature
        A_tilde += np.outer(feature, diff.T)
        b_tilde += feature * reward
    weight = np.dot(inv(A_tilde), b_tilde)
    return weight




if __name__ == "__main__":


    weight = np.zeros(99)
    old_weight = np.copy(weight)
    diff = 10000

    # for i in range(100):
    #     weight = LSTDQ(getAction,weight,getFeature2,transition,getReward,99,500)
    #     diff = np.dot(old_weight-weight,old_weight-weight)
    #     print diff
    #     sample(weight, getFeature2, transition, getReward, getAction, 10)
        #print weight
    exp = sample3(getFeature2)

    for i in range(20):
        exp_list = random.sample(exp,500)
        exp_list = sample(weight, getFeature2, transition, getReward, getAction, 500)#sample
        weight = LSTD(exp_list,99)
        diff = np.dot(old_weight - weight, old_weight - weight)
        sum = np.dot(weight,weight)
        print diff,sum

    #
    #
    #     print weight

    #print sample()
    # for i_episode in range(10):
    #     print i_episode
    #     observation = env.reset()  # initialize S
    #         # x = np.array([observation[0], observation[1]])
    #         # feature = getFeature(x)
    #     for t in range(5000):
    #             # print t
    #         env.render(close=False)  # turn off animation
    #
    #         action = getAction([0, 1, 2], weight, observation, getFeature2)  # greedy
    #             # print feature
    #             # print action
    #         observation, reward, done, info = env.step(action)  # perform action
    #         if observation[0] >= 0.5:
    #             print "finished"

                # print RBF(np.array([1,2]),np.array([1,1]),np.array([1,1]))
                # print getFeature(np.array([-.5,0.05]))

    # cState = env.observation_space.sample()
    # #print cState,
    # action = actions2[env.action_space.sample()]
    # print action,
    # print getFeature2(cState)

    # old_beta_list = np.copy(beta_list)
    # step_list = []
    # covariance = np.array([.04, .0004])




    # for i_episode in range(0):
    #     print i_episode,
    #     observation = env.reset()  # initialize S
    #     x = np.array([observation[0], observation[1]])
    #     feature = getFeature(x)
    #     for t in range(5000):
    #         #print t
    #         env.render(close=True)  # turn off animation
    #
    #         action = chooseAction(feature,beta_list)#greedy
    #         #print feature
    #         #print action
    #         observation, reward, done, info = env.step(action)  # perform action
    #
    #         #print RBF(np.array([1,2]),np.array([1,1]),np.array([1,1]))
    #         #print getFeature(np.array([-.5,0.05]))
    #         if observation[0] >= .5:  # reward
    #             reward = 0
    #         else:
    #             reward = -1
    #         next_x = np.array([observation[0],observation[1]])#continuous state space
    #         next_feature = getFeature(next_x)
    #         e_list[action] += feature
    #         if observation[0]>=.5:
    #             #different iteration
    #             #one_iteration(feature, next_feature, reward, beta_list[action], e_list[action])
    #             diff = reward-np.dot(feature,beta_list[action])
    #             beta_list[action] += alpha*e_list[action]*diff
    #             print "episode finished ",t,
    #             step_list.append(t)
    #             break
    #         maxAction = chooseAction(next_feature,beta_list)
    #        # one_iteration(feature,next_feature,reward,beta_list[maxAction],e_list[action])#update weight vector
    #         #gradient decent
    #         diff = reward+gamma*np.dot(next_feature,beta_list[maxAction])-np.dot(feature,beta_list[action])
    #         beta_list[action] += alpha*e_list[action]*diff
    #         #print diff
    #         e_list[action]*=gamma*lambda_
    #         feature = next_feature
    #    # print beta_list
    # #generateCenter((-1.2,.5),4,(-0.07,.07),8)
    # #print getFeature(np.array([-1.,0.])).shape
    #     difference = 0
    #     for k in range(3):
    #         difference += np.dot(old_beta_list[k] - beta_list[k], old_beta_list[k] - beta_list[k])
    #     print "diff of beta ",math.sqrt(difference)
    #     old_beta_list = np.copy(beta_list)
    #
    # # plt.plot(step_list)
    # # plt.show()