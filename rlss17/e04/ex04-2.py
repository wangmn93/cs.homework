# coding=utf-8
from scipy import *
import random
from pybrain.rl.environments.mazes import Maze, MDPMazeTask
import numpy as np
from matplotlib import pyplot as plt
import pylab
# grid world
structure = array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                   [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                   [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                   [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                   [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
height = 4
width = 12

cliff = [(4,i) for i in range(2,12)]
goal = (4,12)

environment = Maze(structure, (5, 13),initPos=[(4,1)]) #initial state and unreachable final state
task = MDPMazeTask(environment)
q_value = {}
actions = [0,1,2,3]
alpha  = .1
gamma = 1

def moving_average(a, n=10) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

# def movingaverage (values, window=20):
#     weights = np.repeat(1.0, window)/window
#     sma = np.convolve(values, weights, 'valid')
#     return sma

def getReward(next):
    if next in cliff:
        return -100
    else:
        return -1

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

def MC_Reward(full_trace,agent='SARSA'):

    R = 0
    k = 0
    R_list = []
    discounted_reward = 0
    for step in full_trace[:-1]:
        R = 0
        discounted_reward += gamma ** k * step[2]
        R+=discounted_reward
        if agent == 'SARSA':
            R += gamma ** k * q_value[(full_trace[k+1][0], full_trace[k+1][1])]
        elif agent == 'Q':
            # off-policy
            action = eps_greedy(full_trace[k+1][0], eps=0)
            R += gamma ** k * q_value[(full_trace[k+1][0], action)]

        R_list.append(R)
        k+=1



    return R_list

def lambda_TD(trace,lambda_=.1,agent = 'SARSA'):
    i = 0
    R_list_full = MC_Reward(trace,agent=agent)
    # print R_list_full
    for step in trace:

        R_lambda = 0
        R = 0
        j=0
        shifted = trace[i:]
        temp = 0
        for r in range(len(trace[:i])):
            temp += (gamma**r)*trace[r][2]
        if i != 0 :
            R_list = [ (x-temp)/ (gamma**(i)) for x in R_list_full[i:]]
        else:
            R_list = R_list_full

        for t in shifted:
            weight = (1-lambda_)*(lambda_**(j))
            next_step = shifted[j+1:j+2]
            if len(next_step)==1:
                R = R_list[j]
            R_lambda += weight*R
            # print weight,R,next_step,
            j+=1
        # print R_lambda
        #update

        # print 'update',step,
        current = step[0]
        # if agent == 'SARSA':
        action = step[1]
        q_value[(current, action)] = (1 - alpha) * q_value[(current, action)] + alpha * (R_lambda)
        # print q_value[(current,action)]
        i+=1




def learning(agent='SARSA',lambda_TD_on=True,maxEpisode = 400,lambda_=.1):
    #initialize q_value
    for i in range(1,height+1,1):
        for j in range(1,width+1,1):
            for k in actions:
                q_value[((i,j),k)] = 0.

    steps = 0
    steps2 = 0
    episode = 0
    steps_list =[]
    steps_list2 = []
    reward2 = 0
    reward_list=[]
    trace_list=[]
    trace = []

    current = environment.initPos[0]  # initial S
    action = eps_greedy(current)#initial A

    while episode != 400:
        #count steps
        steps +=1
        steps2 += 1
        #transition
        current = environment.perseus  # S
        if agent != 'SARSA':
            # off-policy
            action = eps_greedy(current)  #A
        environment.performAction(action)# perform action
        next = environment.perseus #S'
        reward = getReward(next)#R
        reward2 += reward
        action2 = eps_greedy(next) #A'
        # print current, action,getReward(next), next
        trace.append((current,action,reward))
        #SARSA
        if agent=='SARSA':
            if not lambda_TD_on:
                q_value[(current, action)] = (1 - alpha) * q_value[(current, action)] + alpha * (
                    reward + gamma * q_value[(next, action2)])
            # on-policy
            action = action2

        #Q-Learning
        else:
            if not lambda_TD_on:
                action2 = eps_greedy(next,eps=0)#max q without eps greedy
                q_value[(current, action)] = (1 - alpha) * q_value[(current, action)] + alpha * (
                    reward + gamma * q_value[(next, action2)])

        #reset
        if next in cliff+[goal]:
            trace.append((next,0,0))#terminal state
            trace_list.append(trace)
            if lambda_TD_on:
                # print trace
                lambda_TD(trace,lambda_ = lambda_)
                a=0
            # print len(trace),episode #print num of episode and length of trace
            trace=[]
            episode+=1

            steps_list.append(steps)
            steps_list2.append(steps2)
            reward_list.append(reward2)
            steps2 = 0
            reward2 = 0
            environment.reset()  # reset to initial state
            current = environment.initPos[0]  # initial S
            action = eps_greedy(current)  # initial A
    return steps_list2,reward_list


def visualizePolicy():
    for i in range(1,height+1,1):
        for j in range(1,width+1,1):
            a = eps_greedy((i, j), eps=0)
            d = ['|▼', '|▶', '|▲', '|◀']

            if (i, j) == goal:
                print '|G',
            elif (i,j) in cliff:
                print '|C',
            else:
                print d[a],
        print "|"

if __name__ == "__main__":
    # q_value[((3,3),1)]=1
    # q_value[((3, 3), 0)] = 5
    # trace = [((3,3),1,1),((3,3),1,2),((3,3),1,3),((3,3),0,0)]
    # print MC_Reward(trace)
    # lambda_TD(trace)
    # print MC_Reward(trace[:1])
    # steps_sarsa, reward_sarsa = learning('SARSA',lambda_TD_on=True,maxEpisode = 400,lambda_=.2)
    # visualizePolicy()
    # plt.plot(moving_average(reward_sarsa), 'r', label='SARSA')
    for lambda_ in [.1,.2,.3,.4,.5,.6,.7,.8,.9]:
        steps_q, reward_q = learning('Q',lambda_TD_on=True,maxEpisode = 400,lambda_ = lambda_)
        #visualizePolicy()
        #plt.plot(moving_average(reward_q), 'b', label='Q')



        #pylab.legend(loc='upper left')
        # print 'SARSA mean reward after 300 iterations ',mean(reward_sarsa[300:])
        # print 'SARSA mean steps after 300 iterations ',mean(steps_sarsa[300:])
        print 'lambda ',lambda_
        print 'Q mean reward after 300 iterations ',mean(reward_q[300:])
        print 'Q mean steps after 300 iterations ',mean(steps_q[300:])
        # plt.bar(np.arange(len(reward_list)),reward_list)
        #plt.show()

