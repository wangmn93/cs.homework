# coding=utf-8
from scipy import *
import random
from pybrain.rl.environments.mazes import Maze, MDPMazeTask
import numpy as np
import pylab

def roomShift(state):
    if state[0] < 6:
        if state[1] < 6:
            shift =(0,0)
        else:
            shift = (0,-6)
    else:
        if state[1] < 6:
            shift =(-6,0)
        else:
            shift = (-6,-6)
    return shift

def eps_greedy(q_value,current,eps):
    if np.random.uniform()<eps:
        return random.choice(range(len(q_value[current[0],current[1],:])))
    else:
        return np.argmax(q_value[current[0],current[1],:],0)


def learning(environment,q_value,goal,max,options=[],policies=[],terminations=[]):
    it = 0
    alpha = .3
    gamma = .99
    # gamma2 = .99
    environment.reset()  # initial s
    current = environment.perseus
    steps = 0
    while it<max:
        steps += 1
        # current = environment.perseus
        action = eps_greedy(q_value,current,.1)
        if action not in options:
            environment.performAction(action)
            next = environment.perseus
            if next == goal:
                reward = 1
            else:
                reward = 0
            #update
            diff = reward + gamma * q_value[next[0], next[1], eps_greedy(q_value, next, .0)] - q_value[
                current[0], current[1], action]
            q_value[current[0], current[1], action] = q_value[current[0], current[1], action] + alpha * diff

        elif current in [(3,6),(6,3),(9,6),(6,9)]:
            while action in options:
                action = eps_greedy(q_value,current,.1)
            environment.performAction(action)
            next = environment.perseus
            if next == goal:
                reward = 1
            else:
                reward = 0
            #update
            diff = reward + gamma * q_value[next[0], next[1], eps_greedy(q_value, next, .0)] - q_value[
                current[0], current[1], action]
            q_value[current[0], current[1], action] = q_value[current[0], current[1], action] + alpha * diff

        else:
            k = 0
            policy = policies[options.index(action)]
            termination = terminations[options.index(action)]
            reward = 0
            exp_list = []
            reward_list = []
            current2 = (current[0], current[1])
            # print "option ",action,current
            while 1:
                k += 1

                shift = roomShift(current2)
                # print current,shift,current[0]+shift[0]-1,current[1]+shift[1]-1
                action2 = policy[current2[0]+shift[0]-1][current2[1]+shift[1]-1]
                environment.performAction(action2)
                next = environment.perseus
                if next == goal:
                    reward+= (gamma**k)*1
                    reward_list.append(1)
                else:
                    reward += (gamma**k)*0
                    reward_list.append(0)
                exp_list.append((current2,action2,next,k-1))
                if next == goal or (current2[0]+shift[0]-1,current2[1]+shift[1]-1) == termination:
                    break
                else:
                    current2 = next
            #naive update
            if 0:
                diff = reward + gamma**k * q_value[next[0], next[1], eps_greedy(q_value, next, .0)] - q_value[
                    current[0], current[1], action]
                q_value[current[0], current[1], action] = q_value[current[0], current[1], action] + alpha * diff

            # update type 1
            if 1:
                for exp in exp_list:
                    c = exp[0]
                    a = exp[1]
                    n = exp[2]
                    l = exp[3]
                    r = 0
                    for i in range(k-l):
                        r += gamma**i*reward_list[i]
                    diff = r + gamma ** (k-l) * q_value[next[0], next[1], eps_greedy(q_value, next, .0)] - q_value[
                            c[0], c[1], action]
                    q_value[c[0], c[1], action] = q_value[c[0], c[1], action] + alpha * diff
            # update type 2
            if 0:
                for exp in exp_list:
                    c = exp[0]
                    a = exp[1]
                    n = exp[2]
                    l = exp[3]
                    r = reward_list[l]
                    diff = r + gamma  * q_value[n[0], n[1], eps_greedy(q_value, n, .0)] - q_value[
                            c[0], c[1], a]
                    q_value[c[0], c[1], a] = q_value[c[0], c[1], a] + alpha * diff



        if next == goal or steps > 200:
            it += 1
            if next == goal:
                print "reach goal"
            else:
                print "reach 200"
            environment.reset()
            current = environment.perseus
        else:
            current = next

def generateInitPos():
    initPos = []
    for i in range(1, 6):
        for j in range(1, 6):
            initPos.append((i, j))

    for i in range(1, 6):
        for j in range(7, 12):
            initPos.append((i, j))

    for i in range(7, 12):
        for j in range(1, 6):
            initPos.append((i, j))

    for i in range(7, 12):
        for j in range(7, 12):
            initPos.append((i, j))

    initPos.append((6, 3))
    initPos.append((3, 6))
    initPos.append((6, 9))
    initPos.append((9, 6))
    return initPos

def plotPolicy(grid,q_value,goal,option=[]):
    for i in range(13):
        for j in range(13):
            action = eps_greedy(q_value,(i,j),.0)
            d = ['|▼', '|▶', '|▲', '|◀']+option

            if (i,j)==goal:
                print '|G',
            elif grid[i,j]==0:
                print d[action],
            else:
                print '|W',
        print "|"
    print ""


if __name__ == "__main__":
    #option policy
    up_door = [[1, 1, 2, 3, 3],
                  [2, 2, 2, 2, 2],
                  [2, 1, 2, 3, 3],
                  [2, 2, 2, 2, 2],
                  [2, 1, 2, 3, 3]]
    left_door = [[0, 3, 3, 3, 3],
                 [0, 0, 0, 3, 0],
                 [3, 3, 3, 3, 3],
                 [2, 3, 2, 3, 3],
                 [2, 2, 2, 2, 2]]
    right_door = [[0, 1, 0, 0, 0],
                  [0, 1, 0, 0, 0],
                  [1, 1, 1, 1, 1],
                  [2, 1, 2, 1, 2],
                  [1, 1, 1, 2, 2]]
    down_door = [[1, 0, 0, 0, 3],
                 [1, 0, 0, 0, 0],
                 [0, 1, 0, 3, 0],
                 [0, 0, 0, 0, 0],
                 [1, 1, 0, 3, 3]]

    #grid world
    grid = array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                  [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
                  [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
                  [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                  [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
                  [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
                  [1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1],
                  [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
                  [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
                  [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                  [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
                  [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
                  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
    actions = [0, 1, 2, 3]
    options = [4, 5, 6, 7]
    terminations = [(0,2),(2,0),(2,4),(4,2)]
    target = (8, 3)
    environment = Maze(grid, goal=(8, 3),initPos=generateInitPos()) #initial state and final state
    q_value = np.ones((13,13,8))
    environment.reset()
    # learning(environment, q_value, target, 10000)
    learning(environment,q_value,target,1000,options=options,policies=[up_door,left_door,right_door,down_door],terminations=terminations)
    plotPolicy(grid,q_value,target,option=['|U','|L','|R','|D'])
    pylab.pcolor(q_value.reshape((13*13,8)).max(1).reshape((13,13)))
    pylab.show()

