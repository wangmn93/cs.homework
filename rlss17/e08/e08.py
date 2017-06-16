import sys
import numpy as np
from numpy.linalg import norm
from numpy import dot
sys.path.insert(0, '../e07')
sys.path.insert(0,'../../mlss17/e06')
# from e06gmm import gaussian_evaluate
from e07 import cartPole, Rporp,sampleFromPolicy,samplePerturbation,evaluatePolicy
from math import sqrt,exp,pi
from matplotlib import pyplot as plt

def plot(list,title):
    plt.plot(list, label=title)
    plt.legend(loc='lower right')
    plt.show()

def evaluate_gaussian(mean,std,x):
    # return gaussian_evaluate(np.array([mean]),np.array([std]),np.array([x]))
    return 1/sqrt(2*pi*(std**2))*exp(-(x-mean)**2/(2*std**2))

#sample trajectories
def sampleTrajectory(omega,std,M=50,maxSteps=2000):
    trajectories = []
    sumed_reward = []
    for i_episode in range(M):
        trajectory = []
        sum = 0
        observation = np.zeros(4)
        for t in range(maxSteps):
            action = sampleFromPolicy(mean=dot(observation, omega),std = std)
            next_observation, reward = cartPole(observation, action)
            sum += reward
            trajectory.append((observation,action,reward,next_observation))
            observation = next_observation
            if reward == -1:
                sumed_reward.append(sum)
                trajectories.append(trajectory)
                break
    return trajectories,sumed_reward


#
def REINFORCE(trajectories,sumed_reward,omega,std,baseline=False):
    M = len(trajectories)
    gradient = np.zeros(4)
    b1 = np.zeros(4)
    b2 = np.zeros(4)
    temp2 = np.zeros(4)
    for trajectory,R in zip(trajectories,sumed_reward):
        temp = np.zeros(4)
        for step in trajectory:
            state = step[0]
            action = step[1]
            # reward = step[2]
            increment = -1/(std**2)*(dot(omega,state)-action)*state
            temp += increment
        if baseline:
            b1 += np.power(temp,2)*R
            b2 += np.power(temp,2)
        temp2 += temp
        temp *= R
        gradient += temp
    if baseline:
        b1 /= M
        b2 /= M
        b = np.divide(b1,b2)
        gradient -= np.multiply(temp2,b)
        gradient /= M
    else:
        gradient /= M
    return gradient

def GPOMDP(trajectories,omega,std):
    b1k = []
    b2k = []
    temp2 = []
    gradients = []
    M = len(trajectories)
    for trajectory in trajectories:
        for i in range(len(trajectory)):
            temp = np.zeros(4)
            reward = trajectory[i][2]
            for step in trajectory[:i+1]:
                state = step[0]
                action = step[1]
                increment = -1 / (std ** 2) * (dot(omega, state) - action) * state
                temp += increment
            if len(b1k)<=i:
                b1k.append(np.power(temp,2)*reward)
                b2k.append(np.power(temp,2))
                temp2.append(temp)
                gradients.append(temp*reward)
            else:
                b1k[i] += np.power(temp,2)*reward
                b2k[i] += np.power(temp,2)
                temp2[i] += temp
                gradients[i] += temp*reward
    bk = [np.divide(b1k[i],b2k[i]) for j in range(len(b1k))]
    gradient = np.zeros(4)
    for t,b,g in zip(temp2,bk,gradients):
        gradient += (g - np.multiply(t,b))/M

    return gradient







if __name__ == "__main__":
    initState = np.zeros(4)
    omega = np.zeros(4)
    std = sqrt(.5)
    alpha = .5
    steps = np.ones(4) * alpha
    prev_g = np.zeros(4)
    expect_reward_list = []
    for i in range(200):
        trajectories,sumed_reward = sampleTrajectory(omega,std)
        # gradient = REINFORCE(trajectories,sumed_reward,omega,std,baseline=False)
        gradient = GPOMDP(trajectories,omega,std)
        # alpha = 10./(i+1.)
        gradient /= (norm(gradient)+.1)
        delta_omega,prev_g,steps = Rporp(gradient,prev_g,steps,4)
        # omega += alpha*gradient/(norm(gradient)+.1)
        # omega += delta_omega
        omega += alpha*gradient
        expect_reward = evaluatePolicy(omega,10,2000)
        expect_reward_list.append(expect_reward)
        print expect_reward,gradient,omega
    # print evaluate_gaussian(0,sqrt(0.5),0)
    plot(expect_reward_list,"J")