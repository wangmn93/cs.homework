import numpy as np
import math
from numpy.linalg import inv,norm
from numpy import dot
from math import cos,sin
from matplotlib import  pyplot as plt

def cartPole(state,action):
    GRAVITY = 9.8
    MASSCART = 1.0
    m1 = .1
    m2 = MASSCART + m1
    LENGTH = .5
    POLE = MASSCART*LENGTH
    FORCE_MAG = 10.
    TAU = .02

    theta = state[2]
    theta_dot = state[3]
    x = state[0]
    x_dot = state[1]

    eps1 = sampleFromPolicy(0.,std=.01)
    eps2 = sampleFromPolicy(0., std = 0.0001)

    temp = (action+ POLE*theta_dot**2*sin(theta))/m2
    theta_acc = (GRAVITY*sin(theta)-cos(theta)*temp)/(LENGTH*(4./3.-m1*cos(theta)**2/m2))
    x_acc = temp - POLE*theta_acc*cos(theta)/m2
    next_x = x + TAU*x_dot + eps1
    next_x_dot = x_dot + TAU*x_acc + eps1
    next_theta = theta + TAU*theta_dot + eps2
    next_theta_dot = theta_dot + TAU*theta_acc + eps2

    if abs(next_theta)>.21:
        reward = -1
    else:
        reward = 1

    return np.array([next_x,next_x_dot,next_theta,next_theta_dot]),reward

def sampleFromPolicy(mean,std=math.sqrt(0.001)):
    action = 100
    while abs(action)>10:
        action = np.random.normal(loc=mean, scale=std, size=None)
    return action

def samplePerturbation(size):
    return np.random.uniform(-1,1,size=size)

def evaluatePolicy(omega):
    reward_list = []
    for i_episode in range(50):
        observation = np.zeros(4)
        sumOfReward = 0
        for t in range(1000):
            action = sampleFromPolicy(mean=dot(observation,omega))
            observation,reward = cartPole(observation,action)
            sumOfReward += reward
            if reward == -1:
                break
        reward_list.append(sumOfReward)
    return np.mean(np.array(reward_list))

def gradientAccent(old_omega, alpha):

    J = evaluatePolicy(old_omega)
    delta_omega_list = []
    delta_J_list = []
    for i in range(M):
        delta_omega = samplePerturbation(4)
        newOmega = old_omega + delta_omega
        delta_J = evaluatePolicy(newOmega) - J
        delta_omega_list.append(delta_omega)
        delta_J_list.append(delta_J)
    dOMEGA = np.array(delta_omega_list)
    dJ = np.array(delta_J_list)
    gradientFD = dot(dot(inv(dot(dOMEGA.T, dOMEGA)), dOMEGA.T), dJ)
    omega = old_omega+ alpha*gradientFD/(norm(gradientFD)+0.01)
    # print gradientFD
    return omega

if __name__ == "__main__":
    # gamma = 1
    M = 50
    # print samplePerturbation(4)+omega
    omega = np.zeros(4)
    reward_list = []
    # print evaluatePolicy(env,omega)
    for i in range(200):
        alpha = .5
        omega = gradientAccent(omega, alpha)
        expectReward = evaluatePolicy(omega)
        reward_list.append(expectReward)
        print i, expectReward
    plt.plot(reward_list,label="J")
    plt.legend(loc='lower right')
    plt.show()
