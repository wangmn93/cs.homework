# coding=utf-8
import numpy as np
import random
import time

# grid world
map = np.full((7, 10), 0.)
# set start, goal state and wind
map[3][7] = 1.
start = np.array([3, 0])
goal = np.array([3, 7])
wind = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
# actions
LEFT = np.array([1, 0])
RIGHT = np.array([-1, 0])
UP = np.array([0, 1])
DOWN = np.array([0, -1])
ACTIONS = [LEFT, RIGHT, UP, DOWN]

state_action_pairs = list()
q_value_list = list()

def q_value(state_action):
    index = 0
    # aa = state_action_pairs.index(state_action)
    for state_action_pair in state_action_pairs:
        # print np.equal(state_action[0],state_action_pair[0])
        # print state_action[0][0] == state_action_pair[0][0]

        if state_action[0][0] == state_action_pair[0][0] and\
        state_action[0][1] == state_action_pair[0][1] and\
        state_action[1][0] == state_action_pair[1][0] and\
        state_action[1][1] == state_action_pair[1][1]:

            return index
        else:
            index += 1

    print index
    return index

def validActions(current):
    validActions = list()
    for action in ACTIONS:
        temp = current + action
        # check boundary
        if temp[0] >= 0 and temp[0] <= 9 and temp[1] >= 0 and temp[1] <= 6:
            validActions.append(np.copy(action))
    # if validActions == []:
    #     a =0
    return validActions

def takeAction(epsilon,current):
    actions  = validActions(current)
    # if actions == []:
    #     actions2 = validActions(current)
    if random.uniform(0, 1)<1-epsilon:
        max_value= -float("inf")
        final_action = None
        for action in actions:
            index = q_value((current, action))
            value = q_value_list[index]
            if value >= max_value:
                final_action = np.copy(action)
        # if final_action == None:
        #     b =0
        return final_action
    else:
        return random.choice(actions)

def actionToSymbol(action):
    if np.array_equal(action,LEFT):
        return u"▻"
    if np.array_equal(action,RIGHT):
        return u"◅"
    if np.array_equal(action,UP):
        return u"▲"
    if np.array_equal(action,DOWN):
        return u"▼"


def isDone(next):
    a = map[next[1]]
    b = a[next[0]]
    if b == 1:
        return True,1
    else:
        return False,-1


if __name__ == "__main__":
    alpha = .2
    gama = 1
    current = np.array([6, 3])
    # q_value = dict()

    # print map
    # print wind
    for i in range(0,10):
        for j in range(0,7):
            # print i,j,validActions(np.array([i,j]))
            current = np.array([i, j])
            for action in validActions(current):
                # print i,j,isDone(np.array([i, j]))
                state_action_pairs.append((current,action))
                q_value_list.append(0)
                # q_value[(current,action)] =0

    # current = np.array([0, 3]) # start position
    # # print current
    # # for i in range(0,10,1):
    # #     print actionToSymbol(takeAction(.5, current))
    # action = takeAction(.3, current)  # take action
    # print action
    path = list()
    for i in range(0,1000,1):
        current = np.array([0, 3])  # start position
        # print current
        # for i in range(0,10,1):
        #     print actionToSymbol(takeAction(.5, current))
        action = takeAction(.2, current)  # take action
        path = list()
        while True:
            # LEFT = np.array([1, 0])
            # RIGHT = np.array([-1, 0])
            # UP = np.array([0, 1])
            # DOWN = np.array([0, -1])
            # ACTIONS = [LEFT, RIGHT, UP, DOWN]

            windForce = wind[int(current[0])] #compute wind effect
            windAction = [np.copy(UP) for i in range(0,windForce)]
            temp = ""
            actualAction = np.copy(action) # add wind effect to actual action
            for u in windAction:
                temp += actionToSymbol(u)+" "
                actualAction += u
            path.append((current,actionToSymbol(action)+temp))
            # print current
            # print actionToSymbol(action),temp #display symbolic action
            next_s = current + actualAction # transition
            if next_s[1]>6 :next_s[1] =6
            if next_s[0] > 9: next_s[0] = 9
            next_action = takeAction(.2,next_s)
            # print next_s,next_action
            done, reward = isDone(next_s)
            # update q value
            index = q_value((current,action))
            # if next_action == None:
            #     a =0
            next_index = q_value((next_s,next_action))
            # print next_s,next_action
            q_value_list[index] = q_value_list[index] + \
                                        alpha*(reward+gama*q_value_list[next_index]-q_value_list[index])


            current = next_s
            action = next_action
            # time.sleep(1)
            if done:
                print "done one episode"
                # print current
                print len(path)
                break

    # for a in path:
    #     print a[0],a[1]


