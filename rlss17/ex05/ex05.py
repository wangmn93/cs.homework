import gym
import random
import numpy as np
from gym import wrappers
from matplotlib import  pyplot as plt
import pylab
env = gym.make('MountainCar-v0')

alpha = .1
lambda_ = .7
gamma = 1
eps =0.
actions = [0,1,2]
q = np.zeros((21,21,3))#tabular
q_value = {}
e = {}
for i in range(0, 21, 1):
    for j in range(0, 21, 1):
        for k in actions:
            q_value[((i, j), k)] = 0.
            e[((i, j), k)] = 0.

def eps_greedy2(current,eps=.0):
    # eps greedy
    if random.uniform(0, 1) < eps:
        action = random.choice(actions)
    else:
        i_action = random.choice(actions)#random initial action choice
        maxValue = q[current[0]][current[1]][i_action]
        action = i_action
        for i in actions:
            if q[current[0]][current[1]][i_action] > maxValue:
                maxValue = q[current[0]][current[1]][i]
                action = i
    return  action


def eps_greedy(current,eps=.0):
    # eps greedy
    if random.uniform(0, 1) < eps:
        action = random.choice(actions)
    else:
        i_action = random.choice(actions)#random initial action choice
        maxValue = q_value[(current, i_action)]
        action = i_action
        for i in actions:
            if q_value[(current, i)] > maxValue:
                maxValue = q_value[(current, i)]
                action = i
    return  action

def discretizeStates(observation):
    return ((int)(observation[0]/0.085)+14,(int)(observation[1]/0.007)+10)

steps = []
for i_episode in range(5):
    observation = env.reset() #initialize S
    cState = discretizeStates(observation)
    #action = eps_greedy(cState,eps=.0)#initialize A
    action = eps_greedy(cState, eps =.0)
    #trace = []na
    #print i_episode
    for t in range(100000):
        env.render(close=False)  # turn off animation
        observation, reward, done, info = env.step(action)#perform action
        if observation[0] >= .5:  # reward
            reward = 0
        else:
            reward = -1
        nState = discretizeStates(observation)
        action_prime = eps_greedy(nState,eps=.0)
        action_star = eps_greedy(nState,eps=.0)
        diff = reward + gamma * q_value[(nState, action_prime)] - q_value[(cState, action)]
        #diff = reward + gamma * q[nState[0]][nState[1]][action_prime] - q[cState[0]][cState[1]][action]
        e[(cState, action)] += 1

        #backup
        for i in range(0, 21, 1):
            for j in range(0, 21, 1):
                for k in actions:
                    q_value[((i,j),k)] += alpha*diff*e[((i,j),k)]
                    #q[i][j][k] += alpha*diff*e[((i,j),k)]
                    if action_prime == action_star:
                        e[((i, j), k)] *= gamma*lambda_
                    else:
                        e[((i, j), k)] = 0.
        cState = nState
        action = action_prime

        if observation[0]>=.5:
            print("Episode finished after {} timesteps".format(t+1))
            steps.append(t+1)
            break
for i in range(0, 21, 1):
    for j in range(0, 21, 1):
        for k in actions:
            q[i][j][k] = q_value[((i, j), k)]
pylab.pcolor(q.max(2))
pylab.show()
print steps
#print q.max(2)
#print q.max(2).shape
plt.plot(steps)
