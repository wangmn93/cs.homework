import numpy as np
from numpy.linalg import inv
import random
import math
import time
from matplotlib import pyplot as plt
#actions
L = -1
R = 1
actions = [L,R]
states = [0,1,2,3]
states2 = range(20)
rewards = [0,1,1,0]
#transition model
transitionMatrix = np.zeros((4,2,4))
def generateTransitionMatrix():
    for state in states:
        for action in actions:
            next1 = min(max(0,state+action),3)
            next2 = min(max(0,state-action),3)
            if action == L:
                transitionMatrix[state][0][next1] = .9
                transitionMatrix[state][0][next2] = .1
            elif action == R:
                transitionMatrix[state][1][next1] = .9
                transitionMatrix[state][1][next2] = .1

# def checkValidTransition(state,action,next_state):
#     if next_state in [min(max(0,state+action),3),min(max(0,state-action),3)]:
#         return True
#     else:
#         return False
def transition2(state, action):
    if random.uniform(0, 1)<0.9:
        return min(max(0,state+action),19)
    else:
        return min(max(0,state-action),19)

def transition3(state, action):
    if random.uniform(0, 1)<0.9:
        return min(max(0,state+action),49)
    else:
        return min(max(0,state-action),49)

def transition(state, action):
    if random.uniform(0, 1)<0.9:
        return min(max(0,state+action),3)
    else:
        return min(max(0,state-action),3)

def getReward(next_state):
    if next_state in [1,2]:
        return 1
    else:
        return 0

def getReward2(next_state):
    if next_state in [0,19]:
        return 1
    else:
        return 0

def getReward3(next_state):
    if next_state in [9,40]:
        return 1
    else:
        return 0

def getFeature(state, action):
    feature = np.zeros(6)
    if action == L:
        feature[0] = 1
        feature[1] = state
        feature[2] = state**2
    elif action == R:
        feature[3] = 1
        feature[4] = state
        feature[5] = state ** 2
    return feature

def getFeature2(state, action):
    feature = np.zeros(10)
    if action == L:
        feature[0] = 1
        feature[1] = state
        feature[2] = state ** 2
        feature[3] = state ** 3
        feature[4] = state ** 4
    elif action == R:
        feature[5] = 1
        feature[6] = state
        feature[7] = state ** 2
        feature[8] = state ** 3
        feature[9] = state ** 4
    return feature

# def LSTDQ2(pi,gamma=.9,k=10):
#     A_tilde = np.zeros((k, k))
#     b_tilde = np.zeros(k)
#     for i in range(5000):  # random sampling 50 times
#         state = np.random.choice(states2)
#         action = np.random.choice(actions)
#         feature = getFeature2(state, action)
#         next_state = transition2(state, action)
#         # reward = getReward(next_state)
#         reward = getReward2(state)
#         # expect_feature = .9 * getFeature(next1, pi[next1]) + .1 * getFeature(next2, pi[next2])
#         # expect_reward = .9 * getReward(next1) + .1 * getReward(next2)
#         diff = feature - gamma * getFeature2(next_state, pi[next_state])
#         # next_state = transition(state,action)
#         # reward = getReward(next_state)
#         A_tilde += np.outer(feature, diff.T)
#         b_tilde += feature * reward
#     weight = np.dot(inv(A_tilde), b_tilde)
#     return weight

def LSTDQ3(pi,states,actions,feature_function,transition_function,reward_function,k,samples,gamma=.9):
    A_tilde = np.zeros((k, k))
    b_tilde = np.zeros(k)
    for i in range(samples):
        state = np.random.choice(states)
        action = np.random.choice(actions)
        feature = feature_function(state, action)
        next_state = transition_function(state, action)
        # reward = getReward(next_state)
        reward = reward_function(state)
        # expect_feature = .9 * getFeature(next1, pi[next1]) + .1 * getFeature(next2, pi[next2])
        # expect_reward = .9 * getReward(next1) + .1 * getReward(next2)
        diff = feature - gamma * feature_function(next_state, pi[next_state])
        # next_state = transition(state,action)
        # reward = getReward(next_state)
        A_tilde += np.outer(feature, diff.T)
        b_tilde += feature * reward
    weight = np.dot(inv(A_tilde), b_tilde)
    return weight

# def LSTDQ(pi,gamma=.9,k=6):
#     A_tilde = np.zeros((k, k))
#     b_tilde = np.zeros(k)
#     for i in range(50):#random sampling 50 times
#         state = np.random.choice(states)
#         action = np.random.choice(actions)
#         feature = getFeature(state,action)
#         next_state = transition(state,action)
#         reward = getReward(next_state)
#         reward = getReward(state)
#         # expect_feature = .9 * getFeature(next1, pi[next1]) + .1 * getFeature(next2, pi[next2])
#         # expect_reward = .9 * getReward(next1) + .1 * getReward(next2)
#         diff = feature - gamma * getFeature(next_state,pi[next_state])
#         # next_state = transition(state,action)
#         # reward = getReward(next_state)
#         A_tilde += np.outer(feature, diff.T)
#         b_tilde += feature * reward
#     weight = np.dot(inv(A_tilde), b_tilde)
#     return weight

def LSTDQ_MODEL(pi,gamma=.9,k=6):
    A_tilde = np.zeros((k,k))
    b_tilde = np.zeros(k)
    for state in states:
        for action in actions:
            feature = getFeature(state,action)
            next1 = min(max(0, state + action), 3)#deisred
            next2 = min(max(0, state - action), 3)
            # if action == L:
            #     a1 = 0
            #     a2 = 1
            # elif action == R:
            #     a1 = 1
            #     a2 = 0
            # a1 = getFeature(next1,pi[next1])
            # b1 = .9*getFeature(next1,pi[next1])
            # a = getFeature(next2,pi[next2])
            # b=.1*getFeature(next2,pi[next2])
            expect_feature = .9*getFeature(next1,pi[next1])+.1*getFeature(next2,pi[next2])
            #expect_reward = .9*getReward(next1)+.1*getReward(next2)
            expect_reward = .9 * getReward(state) + .1 * getReward(state)
            diff = feature - gamma*expect_feature
            # next_state = transition(state,action)
            # reward = getReward(next_state)
            A_tilde += np.outer(feature, diff.T)
            b_tilde += feature*expect_reward
    weight = np.dot(inv(A_tilde),b_tilde)
    return weight

# def extractPolicy(weight):
#     policy = []
#     for state in states:
#         feature_l = getFeature(state,L)
#         feature_r = getFeature(state,R)
#         if np.dot(weight,feature_l)>np.dot(weight,feature_r):
#             policy.append(L)
#         else:
#             policy.append(R)
#     return policy

# def extractPolicy2(weight):
#     policy = []
#     for state in states2:
#         feature_l = getFeature2(state,L)
#         feature_r = getFeature2(state,R)
#         if np.dot(weight,feature_l)>np.dot(weight,feature_r):
#             policy.append(L)
#         else:
#             policy.append(R)
#     return policy

def extractPolicy3(weight,states,actions,feature_function):
    policy = []
    for state in states:
        feature_l = feature_function(state,L)
        feature_r = feature_function(state,R)
        if np.dot(weight,feature_l)>np.dot(weight,feature_r):
            policy.append(L)
        else:
            policy.append(R)
    return policy

def LSPI(pi0,states,actions,feature_function,transition_function,reward_function,policy_extract,k,samples,iterations,gamma=.9):
    policy = pi0
    for i in range(iterations):
        weight = LSTDQ3(policy, states, actions, feature_function, transition_function, reward_function,k, samples,gamma)
        policy = policy_extract(weight,states,actions,feature_function)

    value_r = []
    value_l = []
    for state in states:
        value_l.append(np.dot(weight, feature_function(state, L)))
        value_r.append(np.dot(weight, feature_function(state, R)))

    plt.figure()
    plt.bar(range(len(states)),policy, align='center', alpha=0.5)
    # plt.plot(policy,label="Policy")
    # plt.legend(loc='upper right')
    #plt.show()

    plt.figure()
    plt.plot(value_l, "o-r", label="L")
    plt.plot(value_r, "*-b", label="R")
    plt.legend(loc='lower right')
    plt.show()

if __name__ == "__main__":
    # gamma = .9
    #generateTransitionMatrix()
    pi0 = [R,R,R,R]#initial policy
    pi1= [R for i in range(20)]
    pi2 = [R for i in range(50)]
    # test transition
    # print 0," ",transition(0,R)
    # print 3, " ", transition(3, L)
    # test getFeature
    # print getFeature(0,L)
    # print getFeature(3,R)
    # while(True):
    #     pi = pi_prime
    # generateTransitionMatrix()
    # print transitionMatrix
    # old_weight = np.zeros(6)
    # weight = LSTDQ_MODEL(pi_prime)
    # #diff = math.sqrt(np.dot(weight,old_weight))
    # value_r = []
    # value_l = []
    # plt.figure()
    # plt.ion()
    # for i in range(10):
    #     weight = LSTDQ2(pi_prime)
    #     print weight
    #     pi_prime = extractPolicy2(weight)
    #     print pi_prime
    #     for state in states2:
    #         value_l.append(np.dot(weight,getFeature2(state,L)))
    #         value_r.append(np.dot(weight,getFeature2(state, R)))
    #     plt.clf()
    #
    #     plt.plot(value_l,"o-r",label="L")
    #     plt.plot(value_r,"*-b",label="R")
    #     plt.legend(loc='lower right')
    #     plt.show(block=False)
    #     # time.sleep(2)
    # print
    LSPI(pi0, range(4), actions, getFeature, transition, getReward, extractPolicy3, 6, 50, 4)
    LSPI(pi1,range(20),actions,getFeature2,transition2,getReward2,extractPolicy3,10,5000,8)
    LSPI(pi2, range(50), actions, getFeature2, transition3, getReward3, extractPolicy3, 10, 10000, 20)