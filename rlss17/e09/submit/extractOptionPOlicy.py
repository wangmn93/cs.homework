# coding=utf-8
from scipy import *
import random
from pybrain.rl.environments.mazes import Maze, MDPMazeTask
import numpy as np
from matplotlib import pyplot as plt
import pylab
import matplotlib.patches as mpatches

upper_door = [[1,1 ,2, 3, 3],
 [2 ,2 ,2 ,2 ,2],
 [2 ,1 ,2 ,3 ,3],
 [2 ,2 ,2 ,2 ,2],
 [2 ,1 ,2 ,3 ,3]]
left_door = [[0 ,3 ,3 ,3 ,3],
 [0 ,0 ,0 ,3 ,0],
 [3 ,3 ,3 ,3 ,3],
 [2 ,3 ,2 ,3, 3],
 [2 ,2 ,2 ,2 ,2]]
right_door =[[0 ,1 ,0 ,0 ,0],
 [0 ,1 ,0 ,0 ,0],
 [1 ,1 ,1 ,1 ,1],
 [2 ,1 ,2 ,1 ,2],
 [1 ,1 ,1 ,2 ,2]]
down_door = [[1 ,0 ,0 ,0 ,3],
 [1 ,0 ,0 ,0 ,0],
 [0 ,1 ,0 ,3 ,0],
 [0 ,0 ,0 ,0 ,0],
 [1 ,1 ,0 ,3 ,3]]
# grid world
structure = array([[1, 1, 1, 1, 1, 1,1],
                   [1, 0, 0, 0, 0, 0,1],
                   [1, 0, 0, 0, 0, 0,1],
                   [1, 0, 0, 0, 0, 0,1],
                   [1, 0, 0, 0, 0, 0,1],
                   [1, 0, 0, 0, 0, 0,1],
                   [1, 1, 1, 0, 1, 1,1]])
grid = np.copy(structure)
initPos = []
for i in range(1,6):
    for j in range(1,6):
        initPos.append((i,j))
environment = Maze(structure, (7, 7),initPos=initPos) #initial state and final state
task = MDPMazeTask(environment)
q_value = {}
actions = [0,1,2,3]
eps = .1
alpha  = .2
gamma = .9


def performAction(action):
    # current = environment.perseus
    environment.performAction(action)
    return environment.perseus

def eps_greedy(current,eps=.1):
    # eps greedy
    if random.uniform(0, 1) < eps:
        action = random.choice(actions)
    else:
        maxValue = q_value[(current, 0)]
        action = 0
        for i in actions:
            if q_value[(current, i)] > maxValue:
                maxValue = q_value[(current, i)]
                action = i
    return  action

def learning(goal,agent='SARSA',maxEpisode=800):
    # # initialize q_value
    # for i in range(1, 8, 1):
    #     for j in range(1, 11, 1):
    #         for k in actions:
    #             q_value[((i, j), k)] = 0.

    steps = 0
    steps2 = 0
    episode = 0
    steps_list =[]
    steps_list2 = []
    current = environment.perseus #initial s
    action = eps_greedy(current) #initial a

    while episode!=maxEpisode+1:
        steps += 1
        steps2 += 1
        # current = environment.perseus #s
        # action = eps_greedy(current) #a
        if agent != 'SARSA':
            # off-policy
            action = eps_greedy(current)


        next = performAction(action)
        action2 = eps_greedy(next)  # a

        if next !=goal:
            reward =-1
        else:
            reward=1

        # Q
        if agent != 'SARSA':
            action2 = eps_greedy(next, eps=0)

        q_value[(current,action)] = (1-alpha)*q_value[(current,action)] + alpha*(reward+gamma*q_value[(next,action2)])


        if next == goal:
            # reset environment
            environment.reset()
            episode += 1
            steps_list.append(steps)
            steps_list2.append(steps2)
            steps2 = 0
            current = environment.perseus  # initial s
            action = eps_greedy(current)  # initial a
        else:
            current = next # S=S'
            action =action2 #on-policy

    return steps_list,steps_list2

def visualizePolicy(goal,rotate=0):
    policy = []
    # visualize policy
    for i in range(1, 6, 1):
        p = []
        for j in range(1, 6, 1):
            a = eps_greedy((i, j), eps=0)
            p.append(a)
            d = ['|▼', '|▶', '|▲', '|◀']

            if (i, j) == goal:
                print '|G',
            else:
                print d[a],
        print "|"
        policy.append(p)
    print ''
    return np.array(policy)

# def rotatePolicy(policy):
#     newPolicy = policy.T
#     for i in range(0,5):
#         for j in range(0,5):
#             a = newPolicy[i,j]
#             newPolicy[i,j] = (a-1)%4
#     return newPolicy
#
def plotPolicy(policy):
    for i in range(0,5):
        for j in range(0,5):
            a = policy[i,j]
            d = ['|▼', '|▶', '|▲', '|◀']
            print d[a],
        print "|"
    print ""
# for i in actions:
#     print q_value[((5,6),i)]
if __name__ == "__main__":


    # # initialize q_value
    # for i in range(0, 7, 1):
    #     for j in range(0, 7, 1):
    #         for k in actions:
    #             q_value[((i, j), k)] = 0.
    #
    # steps_list_sarsa, steps_list2_sarsa = learning(agent='Q',goal = (6,3),maxEpisode = 800)
    # visualizePolicy((6,3))
    # plt.plot(steps_list_sarsa,'r',label='Q')
    # print mean(steps_list2_sarsa[len(steps_list2_sarsa)/2:])
    goals=[(0,3),(3,0),(3,6),(6,3)]
    # initialize q_value
    for i in range(0, 7, 1):
        for j in range(0, 7, 1):
            for k in actions:
                q_value[((i, j), k)] = 0.
    plotPolicy(np.array(upper_door))

    plotPolicy(np.array(left_door))
    plotPolicy(np.array(right_door))
    plotPolicy(np.array(down_door))
    # steps_list_q, steps_list2_q = learning(agent='Q',goal = goals[3],maxEpisode = 800)
    # policy =  visualizePolicy((3, 6))
    # print policy
    # newPolicy = rotatePolicy(policy)
    # print newPolicy
    # plotPolicy(newPolicy)
    # print ""
    # visualizePolicy((3,6))
    # plt.plot(steps_list_q,'b', label='Q')
    # print mean(steps_list2_q[len(steps_list2_q)/2:])

    # pylab.legend(loc='upper left')
    # plt.show()

    # goals = [(3,8),(5,8),(4,7),(4,9),(5,9),(5,7),(3,9)]
    # for goal in goals:
    #     print goal
    #     steps_list_q, steps_list2_q = learning('Q', maxEpisode=800,goal=goal)
    # visualizePolicy((3,8))





