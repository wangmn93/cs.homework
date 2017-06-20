import numpy as np
import math
from numpy.linalg import inv,norm
from numpy import dot
from math import cos,sin,exp
from matplotlib import  pyplot as plt

#dynamic of cartpole
def cartPole(state,action):
    GRAVITY = 9.8
    MASSCART = 1.0
    m1 = .1
    m2 = MASSCART + m1
    LENGTH = .5
    POLE = m1*LENGTH
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
    # print mean
    it = 0
    while abs(action)>10:
        # print mean
        if it>10:
            break
        it += 1
        action = np.random.normal(loc=mean, scale=std, size=None)
    return max([-10.,min([action,10.])])
    # return action

def samplePerturbation(size):
    return np.random.uniform(-1,1,size=size)

def evaluatePolicy(omega,n=50,max=1000):
    reward_list = []
    for i_episode in range(n):
        observation = np.zeros(4)
        sumOfReward = 0
        for t in range(max):
            action = sampleFromPolicy(mean=dot(observation,omega))
            observation,reward = cartPole(observation,action)
            sumOfReward += reward
            if reward == -1:
                break
        reward_list.append(sumOfReward)
    return np.mean(np.array(reward_list))

def computeAscentGradient(old_omega,M):
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
    return gradientFD

def Rporp(g_FD,prev_g,steps,k):
    if len(g_FD)!=len(prev_g):
        print "dim not match"
    gradient = np.copy(g_FD)
    for i in range(k):
        product = g_FD[i]*prev_g[i]
        if product > 0 :
            steps[i] *= 1.2
            gradient[i] += steps[i]*g_FD[i]/(abs(g_FD[i])+.001)
            prev_g[i] = g_FD[i]
        elif product < 0:
            steps[i] *= .5
            gradient[i] += steps[i]*g_FD[i]/(abs(g_FD[i])+.001)
            prev_g[i] = 0
        else:
            gradient[i] += steps[i]*g_FD[i]/(abs(g_FD[i])+.001)
            prev_g[i] = g_FD[i]
        steps[i] = max([.01,min([5.,steps[i]])])
    return gradient,prev_g,steps

#wolfe condition
def wolfCondition(alpha,omega,gradient):
    a = alpha
    old_J = evaluatePolicy(omega)
    a_ls = .01
    a_p = 1.2
    a_m = .5
    gradient_direction = gradient/(norm(gradient)+0.001)
    while evaluatePolicy(omega+a*gradient)<old_J+a_ls*dot(gradient,a*gradient_direction):
        a *= a_m
    delta_omega = a * gradient_direction
    a *= a_p
    return delta_omega,a

if __name__ == "__main__":
    M = 50
    omega = np.ones(4)
    reward_list = []
    alpha = .5
    steps = np.ones(4)*alpha
    prev_g = np.zeros(4)
    old_expect = 0
    for i in range(200):
        gradientFD = computeAscentGradient(omega,M)
        if old_expect>1200:
            delta_omega, alpha = wolfCondition(alpha,omega,gradientFD) #wolfe condition+line search
        else:
            # delta_omega,prev_g,steps = Rporp(gradientFD,prev_g,steps,4) #adaptive step-size
            alpha = 10.*exp(-i/5.) #decreasing step-size
            delta_omega = alpha*gradientFD/(norm(gradientFD)+0.01)
        omega += delta_omega
        expectReward = evaluatePolicy(omega)
        old_expect = expectReward
        reward_list.append(expectReward)
        print i, expectReward,alpha,gradientFD,omega
    plt.plot(reward_list,label="J")
    plt.legend(loc='lower right')
    plt.show()
