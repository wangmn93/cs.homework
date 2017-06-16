import sys
import numpy as np
sys.path.insert(0, '../e07')
from e07 import cartPole, Rporp,sampleFromPolicy,samplePerturbation

def sampleTrajectory(M=50):
    trajectories = []
    for i_episode in range(50):
        trajectory = []
        observation = np.zeros(4)
        sumOfReward = 0
        for t in range(1000):
            action = sampleFromPolicy(mean=dot(observation, omega))
            observation, reward = cartPole(observation, action)
            sumOfReward += reward
            if reward == -1:
                break
        reward_list.append(sumOfReward)
    return np.mean(np.array(reward_list))

def computeGradient(trajectories):
    a=0



if __name__ == "__main__":
    a = 0