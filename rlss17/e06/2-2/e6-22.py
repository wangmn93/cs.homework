from ple.games.flappybird import FlappyBird
import numpy as np
from ple import PLE
import random
from numpy.linalg import inv
import  numpy.linalg as lin
import time
from matplotlib import pyplot as plt
import csv
import ast


def LSTDQ(getAction,old_weight,feature_function,k,exp,gamma=.99):
    A_tilde = np.eye((k))*.001
    b_tilde = np.zeros(k)
    for j in exp:
        state,action,reward,next_state = parse(j)
        next_action = getAction(old_weight,next_state,feature_function)
        feature = feature_function(state,action)
        next_feature = feature_function(next_state,next_action)
        diff = feature - gamma * next_feature
        # next_state = transition(state,action)
        # reward = getReward(next_state)
        A_tilde += np.outer(feature, diff.T)
        b_tilde += feature * reward
    #print lin.det(A_tilde)
    #print A_tilde
    weight = np.dot(inv(A_tilde), b_tilde)
    return weight

def getAction(weight,state,feature_function):
    feature_l = feature_function(state,119)
    feature_r = feature_function(state,None)
    if np.dot(weight,feature_l)>np.dot(weight,feature_r):
        action = 119
    else:
        action = None
    return action

def sample(exp_list,n=5000):
    return random.sample(exp_list,n)

#def generateRandomExp(n=5000)



def parse(one_exp):
    state = ast.literal_eval(one_exp[0])
    if one_exp[1] == "":
        action = None
    else:
        action = 119
    # action = ast.literal_eval(exp_list[1][1])
    reward = ast.literal_eval(one_exp[2])
    next_state = ast.literal_eval(one_exp[3])
    return state,action,reward,next_state

def parse2(one_exp):
    return one_exp[0],one_exp[1],one_exp[2],one_exp[3]

def getFeature(state, action):
    if action == 119:
        feature1 = [1]
        feature1.append(state["player_vel"] )
        feature1.append(state["next_pipe_dist_to_player"] )
        feature1.append((state["player_y"]-state["next_pipe_top_y"] ))
        feature1.append(state["player_vel"] * state["next_pipe_dist_to_player"])
        feature1.append(state["player_vel"] * (-state["next_pipe_top_y"] + state["player_y"]) )
        feature1.append(state["next_pipe_dist_to_player"] * (-state["next_pipe_top_y"] + state["player_y"]))
        feature1.append(state["player_vel"] ** 2 )
        feature1.append((-state["next_pipe_top_y"] - state["player_y"]) ** 2 )
        feature1.append(state["next_pipe_dist_to_player"] ** 2 )
    else:
        feature1 = [0]
        feature1 += [0, 0, 0, 0, 0, 0, 0, 0, 0]

    if action == None:
        feature2 = [1]
        feature2.append(state["player_vel"])
        feature2.append(state["next_pipe_dist_to_player"])
        feature2.append((state["player_y"] - state["next_pipe_top_y"]))
        feature2.append(state["player_vel"] * state["next_pipe_dist_to_player"])
        feature2.append(state["player_vel"] * (-state["next_pipe_top_y"] + state["player_y"]))
        feature2.append(state["next_pipe_dist_to_player"] * (-state["next_pipe_top_y"] + state["player_y"]))
        feature2.append(state["player_vel"] ** 2)
        feature2.append((-state["next_pipe_top_y"] - state["player_y"]) ** 2)
        feature2.append(state["next_pipe_dist_to_player"] ** 2)
    else:
        feature2 = [0]
        feature2 += [0, 0, 0, 0, 0, 0, 0, 0, 0]
    return np.array(feature1 + feature2)

if __name__ == "__main__":
    with open('exp.csv', 'rb') as f:
        reader = csv.reader(f)
        exp_list = list(reader)
    print len(exp_list)
    game = FlappyBird()
    p = PLE(game, display_screen=True)
    # actions = p.getActionSet()
    p.init()

    # exp_list = []
    # for i in range(50000):
    #     state = game.getGameState()
    #     action = np.random.choice([119,None])
    #     reward = p.act(action)
    #     nState = game.getGameState()
    #     exp_list.append((state,action,reward,nState))
    #     if p.game_over():
    #         p.reset_game()
    #
    #
    # print len(exp_list)

    weight = np.zeros(20)
    old_weight = np.copy(weight)
    diff = 10000
    i = 1
    for i in range(50):
        print i,
        i += 1
        old_weight = np.copy(weight)
        sample_exp = sample(exp_list[1:],n=50000)
        weight = LSTDQ(getAction,weight,getFeature,20,sample_exp)
        # print weight
        diff = old_weight-weight
        diff = np.dot(diff,diff)
        print diff




    #test


    episode = 0
    score_list = []
    while episode != 100:
        state = game.getGameState()
        action = getAction(weight,state,getFeature)
        p.act(action)
        if p.game_over():
            episode += 1
            score_list.append(game.getScore())
            p.reset_game()
        time.sleep(.2)


    plt.plot(score_list)
    plt.show()



    # print state["player_vel"]
    # print action
    # print reward
    # print next_state