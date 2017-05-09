# encoding=utf8
import random
import math
import matplotlib.pyplot as plt
import numpy as np
# map
map = [1,0,0,0,0,0,1]
# actions
RIGHT = 1
LEFT = -1
ACTIONS = [RIGHT,LEFT]

def validActions(current):
    actions = list(ACTIONS)
    for action in actions:
        if current+action > 6 or current+action<0:
            actions.remove(action)
    return actions

def transition(current,action):
    next = current+action
    if isDone(next) and next == 6:
        reward = 1
    else:
        reward = 0
    return next,reward

def takeAction(current):
    actions = validActions(current)
    return  random.choice(actions)

def actionToSymbol(action):
    if action == LEFT:
        return u"◄"
    elif action == RIGHT:
        return u'►'

def isDone(next):
    if map[next] == 1:
        return True
    else:
        return False

def RMS(true, result):
    temp =[]
    for i in range(len(true)):
        temp.append((true[i]-result[i])**2)
    return math.sqrt(np.mean(temp))

if __name__=="__main__":
    lambda_ = .1
    gama_ = 1
    current = 3
    episodes = 75
    # for i in range(1,5):
    #     print actionToSymbol(takeAction(current))

    # initialize value
    # for lambda_ in [.01, .02, .03,.04]:
    # for lambda_ in [.05,.1,.15]:
    for lambda_ in [.1]:
        rms = []
        rms2 = []
        for episodes in range(0,75,24):

            values = [0.,0.5,0.5,0.5,0.5,0.5,0]

            # for i in range(0,100,1):
            #     if isDone(current):
            #         break
            #     takeAction(current)
            bag = {}
            for i in range(0,7):
                bag[i] = []
            for i in range(0,episodes,1):
                current = 3
                rewards = []
                path = []
                while True:
                    path.append(current)
                    # print str(current),
                    if isDone(current):
                        break
                    action = takeAction(current)
                    old = current
                    # print actionToSymbol(action),
                    current,reward = transition(current,action)
                    rewards.append(reward)
                    values[old] = values[old] + lambda_*(reward+gama_*values[current]-values[old])
                    # print values
                    # print '[%d]'%reward,
                # print ""
                # print rewards
                # print path
                for state in path:
                    R=0
                    for i in range(len(rewards[path.index(state):])):
                        sub_reward = rewards[path.index(state):]
                        R += sub_reward[i]*gama_**i

                    # print state,R
                    bag[state].append(R)
                # print bag
            true_value = [0, 1. / 6, 2. / 6, 3. / 6, 4. / 6, 5. / 6, 0]
            print episodes,lambda_,RMS(true_value[1:6],values[1:6])
            rms.append(RMS(true_value[1:6],values[1:6]))
            mc = []
            for i in range(0, 7):
                mc.append(np.mean(bag[i]))
            rms2.append(RMS(true_value[1:6],mc[1:6]))
            plt.plot(values[1:6])
        # plt.plot(rms)
        # plt.plot(rms2)
    true_value = [0, 1. / 6, 2. / 6, 3. / 6, 4. / 6, 5. / 6, 0]
    plt.plot(true_value[1:6], 'r')
    plt.show()
            # print values[1:6]
        # plt.plot(values[1:6], 'b-')
        # plt.plot(mc[1:6], 'g-')
        # plt.plot(true_value[1:6], 'r')
        # plt.show()
        #
        # for
        # print values

