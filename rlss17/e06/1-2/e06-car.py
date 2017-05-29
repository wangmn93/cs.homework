import numpy as np
import math
from numpy.linalg import inv
import gym
import random
import math
from matplotlib import  pyplot as plt

def mdot(*args):
    """Multi argument dot function. http://wiki.scipy.org/Cookbook/MultiDot"""
    return reduce(np.dot, args)

#input type np.array
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

############Feature###############
#for one action, generate 32 rbf and 1 bias
#other action get 33 zeros
#the final feature is a vector of 99
#e.g. for action 1
# feature np.array([0 0 ... 0 0 <32 rbf +1 bias> 0 0 ... 0 0])
def getFeature3(x,action):
    featureList = []
    for i in range(3):
        if i == action:
            featureList.append(np.array([1,x[0],x[1],x[0]*x[1],x[0]**2,x[1]**2]))
        else:
            featureList.append(np.zeros(6))
    return np.concatenate((featureList[0], featureList[1], featureList[2]))

###########choose action based on weight vector###############
#return the action which get the max Q value
def getAction(actions,weight,state,feature_function):
    values = [np.dot(weight,feature_function(state,action))for action in actions]
    return values.index(max(values))


#############copy from gym env################3
def transition2(state,action):
    position = state[0]
    velocity = state[1]
    velocity += (action - 1) * 0.001 + math.cos(3 * position) * (-0.0025)
    velocity = np.clip(velocity, -0.07, 0.07)
    position += velocity
    position = np.clip(position, -1.2, 0.6)
    if (position == -1.2 and velocity < 0): velocity = 0
    return np.array([position,velocity])


####################reward function#############3
def getReward(next_state):
    if next_state[0] >= .5:  # reward
        reward = 0
    else:
        reward = -1
    return reward

if __name__ == "__main__":
    env = gym.make('MountainCar-v0')
    actions = [0,1,2]
    observation = env.reset()

    #######################random sample 10000 steps#######################
    exp = []
    episodes = 0
    for i in range(10000):
        state = env.observation_space.sample()
        action = env.action_space.sample()
        nState = transition2(state, action)
        reward = getReward(nState)
        if reward == 0:
            episodes+=1#record steps which get 0 reward
        exp.append((state, action, reward, nState))
    print episodes
    print len(exp)

    ##############################LSPI#####################################
    weight = np.zeros(99)
    ref = np.ones(99) #reference vector
    old_weight = np.copy(weight)
    weight_list =[] #store weights from each iteration
    for i in range(50):
        exp_list = random.sample(exp,1000)#take a batch of size 1000
        A_tilde = np.eye(99)*0.1 #initialize A
        b_tilde = np.zeros(99) #initialize b
        for e in exp_list:
            nAction = getAction([0, 1, 2], old_weight, e[3], getFeature2)#select action for next state according to current weight
            feature = getFeature2(e[0],e[1])
            reward = e[2]
            nFeature = getFeature2(e[3],nAction)
            diff = feature-.99*nFeature
            A_tilde += np.outer(feature,diff.T)
            b_tilde += feature*reward
        weight = np.dot(inv(A_tilde),b_tilde)
        weight_list.append(weight)
        diff2 = np.dot(old_weight - weight, old_weight - weight)#compute the diff between two weights
        old_weight = np.copy(weight)
        length = math.sqrt(np.dot(weight,weight))#compute the length of weight
        similarity = np.dot(ref,weight)/math.sqrt(np.dot(ref,ref))/math.sqrt(np.dot(weight,weight)) #compute the cos similarity of weight with ref vector
        print diff2,length,similarity

   ######################TEST performance of each weight#######################
    step_list = []
    i = 0
    for weight in weight_list:
        print i,
        i+=1
        observation = env.reset()  # initialize S
        step = 0
        for t in range(1000):
                    # print t
            step += 1
            env.render(close=True)  # turn off animation

            action = getAction([0, 1, 2], weight, observation, getFeature2)  # greedy
                    # print feature
                    # print action
            observation, reward, done, info = env.step(action)  # perform action
            if observation[0] >= 0.5:
                step_list.append(step)
                step = 0
                print "finished"
                break
        print ""
    plt.plot(step_list)
    plt.show()#plot steps of successful episode

