import gym
import numpy as np
import math
from numpy.linalg import inv
import random

def mdot(*args):
    """Multi argument dot function. http://wiki.scipy.org/Cookbook/MultiDot"""
    return reduce(np.dot, args)

def RBF(center, covariance, x):
    d_2 = mdot((x-center).T,inv(np.diag(covariance)),x-center)
    #print d_2
    return math.exp(-d_2/2.)

def generateCenter(range1,n, range2, m):
    range1_span = (range1[1]-range1[0])/(n-1)
    range2_span = (range2[1]-range2[0])/(m-1)
    center1 = [range1[0]+i*range1_span for i in range(n)]
    center2 = [range2[0]+i*range2_span for i in range(m)]
    return center1,center2

#input np.array
def getFeature(x,action,n =4, m=4, range1 =(-1.2,.5),range2=(-0.07,0.07),covariance = np.array([.04, .0004])):
    # generate center according to n and m
    p_centers,v_centers = generateCenter(range1,n,range2,m)
    feature = [1]#append one
    for i in p_centers:
        for j in v_centers:
            rbf = RBF(np.array([i,j]),covariance,x)
            feature.append(rbf)
    final_feature = []
    for i in [0,1,2]:
        if i == action:
            final_feature+=feature
        else:
            final_feature+=[0]*17
    return np.array(final_feature)

def transition(state,action):
    position = state[0]
    velocity = state[1]
    velocity += (action - 1) * 0.001 + math.cos(3 * position) * (-0.0025)
    velocity = np.clip(velocity, -0.07, 0.07)
    position += velocity
    position = np.clip(position, -1.2, 0.6)
    if (position == -1.2 and velocity < 0): velocity = 0
    if position>=.5:
        reward = 0
    else:
        reward = -1
    return np.array([position,velocity]),reward


def getAction(weight,actions,obv):
    values = []
    for i in actions:
        feature = getFeature(obv,i)
        values.append(np.dot(weight,feature))
    return values.index(max(values))

def sampleFromPolicy(weight,getAction,env,k,eps):
    exp_list = []
    success = 0
    observation = env.reset()
    for i in range(k):
        env.render(close=True)
        if random.random()>eps:
            action = getAction(weight,[0,1,2],observation)
        else:
            action = env.action_space.sample()
        # print action
        next_obv, reward, done, info = env.step(action)
        exp_list.append((observation, action, reward, next_obv))
        # print (observation,action,reward,next_obv)

        observation = next_obv
        if next_obv[0] >= 0.5:
            success += 1
            observation = env.reset()
    return exp_list,success

def uniformSample(weight,getAction,env,k,eps):
    # random sample from state space and
    exp_list = []
    success = 0
    obv = env.observation_space.sample()
    for i in range(k):
        if random.random()>eps:
            action = getAction(weight,[0,1,2],obv)
        else:
            action = env.action_space.sample()
        next_obv, reward = transition(obv, action)
        exp_list.append((obv, action, reward, next_obv))
        obv = next_obv
        if next_obv[0] >= .5:
            success += 1
            obv = env.observation_space.sample()

    return exp_list, success

def LSTDQ(exp_list,getAction,old_weight,k,gamma):
    A_tilde = np.eye(k) * 0.1  # initialize A
    b_tilde = np.zeros(k)  # initialize b
    for exp in exp_list:
        obv = exp[0]
        action = exp[1]
        reward = exp[2]
        next_obv = exp[3]

        feature = getFeature(obv,action)
        next_action = getAction(old_weight,[0,1,2],next_obv)
        next_feature = getFeature(next_obv,next_action)
        A_tilde += np.outer(feature,feature-gamma*next_feature)
        b_tilde += feature*reward
    weight = np.dot(inv(A_tilde),b_tilde)
    return weight

def LSPI_ONE(exp_list,getAction,old_weight,k,gamma,env):
    weight = LSTDQ(exp_list, getAction, old_weight, 51, .99)
    difference = np.dot(weight - old_weight, weight - old_weight)
    #testWeight(weight, env, True, 500)
    return weight,difference

def LSPI(exp_list,getAction,old_weight,k,gamma,env):
    difference = 100
    o_w = np.copy(old_weight)
    it=0
    while difference > .01:
        o_w, difference = LSPI_ONE(exp_list, getAction, o_w, 51, .99, env)
        print difference
        it+=1
        if it>30:
            break
    #testWeight(old_weight, env, False, 2000)
    return o_w

def testWeight(old_weight,env,close,k):
    observation = env.reset()
    for i in range(k):
        env.render(close=close)
        action = getAction(old_weight, [0, 1, 2], observation)
        #print action
        next_obv, reward, done, info = env.step(action)
        observation = next_obv
        if next_obv[0] >= 0.5:
            print "Finished"
            # observation = env.reset()
            break

if __name__ == "__main__":
    env = gym.make('MountainCar-v0')
    old_weight = np.zeros(51)
    #print success
    #print getFeature(np.array([-.5,0.01]),2)
    exp_list2 = []




        #exp_list, success = sampleFromPolicy(old_weight,getAction, env, 400,1.)

        #exp_list, success = uniformSample(old_weight, getAction,env, 500, 1.)

        #print "S ",success
    exp_list2 = []
    #exp_list, success = uniformSample(old_weight, getAction, env, 5000, 1.)

    total_exp_lsit = []
    for i in range(10):
        print i,
        exp_list, success = uniformSample(old_weight, getAction, env, 50, 1.)
        total_exp_lsit += exp_list
        diff = 10
        it = 0
        while diff>1:
            #print i,
            if it>20:
                break
            it+=1
            old_weight,diff = LSPI_ONE(total_exp_lsit,getAction,old_weight,51,.90,env)
            print diff
    testWeight(old_weight,env,False,500)
    # for i in range(100):
    #     print  getAction(old_weight,[0,1,2],env.action_space.sample())
    print old_weight
    # print len(list(set(weight_list)))
    #testWeight(old_weight,env,False,500)



