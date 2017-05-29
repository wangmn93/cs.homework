from ple.games.flappybird import FlappyBird
import numpy as np
from ple import PLE
import random
from numpy.linalg import inv
import  numpy.linalg as lin
import time
from matplotlib import pyplot as plt
from threading import Thread
import csv

def moving_average(a, n=10) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def getFeature(state):

    feature = (
        int(state["player_vel"]),
        int(state["next_pipe_dist_to_player"]/10),
        int(state["player_y"]/10 -(state["next_pipe_top_y"])/10),
        int(state["next_pipe_bottom_y"]/10 - state["player_y"]/10)

    )

    return feature


def getFeature2(state,action):
    if action == 119:
        feature1 = [1]
        feature1.append(state["player_vel"]/100)
        feature1.append(state["next_pipe_dist_to_player"]/100)
        feature1.append((state["next_pipe_bottom_y"] - state["player_y"])/100)
        feature1.append(state["player_vel"]*state["next_pipe_dist_to_player"]/10000)
        feature1.append(state["player_vel"]*(state["next_pipe_bottom_y"] - state["player_y"])/10000)
        feature1.append(state["next_pipe_dist_to_player"]*(state["next_pipe_bottom_y"] - state["player_y"])/10000)
        feature1.append(state["player_vel"]**2/10000)
        feature1.append((state["next_pipe_bottom_y"] - state["player_y"])**2/10000)
        feature1.append(state["next_pipe_dist_to_player"]**2/10000)
    else:
        feature1 = [1]
        feature1 += [0,0,0,0,0,0,0,0,0]

    if action == None:
        feature2 = [1]
        feature2.append(state["player_vel"]/100)
        feature2.append(state["next_pipe_dist_to_player"]/100)
        feature2.append((state["next_pipe_bottom_y"] - state["player_y"])/100)
        feature2.append(state["player_vel"] * state["next_pipe_dist_to_player"]/10000)
        feature2.append(state["player_vel"] * (state["next_pipe_bottom_y"] - state["player_y"])/10000)
        feature2.append(state["next_pipe_dist_to_player"] * (state["next_pipe_bottom_y"] - state["player_y"])/10000)
        feature2.append(state["player_vel"] ** 2/10000)
        feature2.append((state["next_pipe_bottom_y"] - state["player_y"]) ** 2/10000)
        feature2.append(state["next_pipe_dist_to_player"] ** 2/10000)
    else:
        feature2 = [1]
        feature2 += [0, 0, 0, 0, 0, 0, 0, 0, 0]
    return np.array(feature1+feature2)

def eps_greedy(current_state,actions,q_value,e,eps=.0):
    # initialize q value to 0
    for a in actions:
        if (current_state,a) not in q_value:
            q_value[(current_state,a)]=0
        if (current_state,a) not in e:
            e[(current_state, a)] = 0
    # eps greedy
    if random.uniform(0, 1) < eps:
        action = random.choice(actions)
    else:
        i_action = random.choice(actions)#random initial action choice
        maxValue = q_value[(current_state, i_action)]
        action = i_action
        for i in actions:
            if q_value[(current_state, i)] > maxValue:
                maxValue = q_value[(current_state, i)]
                action = i
    return  action

def LSTDQ3(getAction,old_weight,feature_function,k,exp,gamma=.9):
    A_tilde = np.eye((k))*.001
    b_tilde = np.zeros(k)
    for j in exp:
        state = j[0]
        action = j[1]
        reward = j[2]
        next_state = j[3]
        next_action = getAction(old_weight,next_state,feature_function)
        feature = feature_function(state,action)
        next_feature = feature_function(next_state,next_action)
        diff = feature - gamma * next_feature
        A_tilde += np.outer(feature, diff.T)
        b_tilde += feature * reward
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


def train(q):
    print "start"
    game = FlappyBird()
    p = PLE(game,display_screen=False)
    actions = p.getActionSet()
    p.init()
    actions = [119, None]
    #reward = p.act(p.NOOP)
    #q = {} #q value
    e = {}
    gamma = 1
    alpha = .2
    alive_time = 0
    alive_list = []
    score_list = []
    all_exp = []
    lambda_ = .7
    #print eps_greedy(current_feature,actions,q,.3)
    currentWeight = np.zeros(20)
    for kk in range(100):
        episodes = 0
        exp = []
        while episodes != 100:
            # test eps greedy
            p.reset_game()
            current_feature = getFeature(game.getGameState())
            for i in range(1000):
                #print i
                alive_time += 1
                cS = game.getGameState()
                current_feature = getFeature(game.getGameState())
                #print current_feature,

                action = eps_greedy(current_feature,actions,q,e,.1)#greedy action
                reward = p.act(action)

                #print reward
                nS = game.getGameState()
                next_feature = getFeature(game.getGameState())
                maxAction = eps_greedy(next_feature, actions, q,e)

                exp.append((cS,action,reward,nS))
                # all_exp.append((cS,action,reward,nS))
                #update
                diff = reward + gamma*q[(next_feature,maxAction)]-q[(current_feature,action)]
                q[(current_feature,action)] += alpha*diff
                # for key in e:
                #     q[key] += alpha * diff * e[key]
                #     e[key] *= gamma*lambda_


                current_feature = next_feature

                if p.game_over():
                    episodes += 1
                    score_list.append(game.getScore())
                    alive_list.append(alive_time)
                    alive_time = 0

                    # current_state = game.getGameState()
                    # current_feature = getFeature(current_state)
                    break
            # if i>9000:
            #     time.sleep(.3)

        print len(exp),
        #
        print kk
        # currentWeight = LSTDQ3(getAction,currentWeight,getFeature2,20,exp)
        # print currentWeight

    # with open('exp2.csv', 'wb') as out:
    #     csv_out = csv.writer(out)
    #     csv_out.writerow(['current', 'action','reward','next'])
    #     for row in all_exp:
    #         csv_out.writerow(row)

    p.reset_game()
    for i in range(1000):
        cS = game.getGameState()
        action = eps_greedy(getFeature(cS),actions,q,e)
        p.act(action)
        if p.game_over():
            p.reset_game()
        time.sleep(.1)

    # p.reset_game()
    # for i in range(1000):
    #     cS = game.getGameState()
    #     action = getAction(currentWeight,cS,getFeature2)
    #     p.act(action)
    #     if p.game_over():
    #         p.reset_game()
    #     time.sleep(.1)


    plt.figure()
    plt.plot(score_list)
    plt.figure()
    plt.plot(moving_average(alive_list,20))
    plt.show()

if __name__ =="__main__":
    q ={}
    #train(q)
    try:
        for i in range(1):
            t = Thread(target=train, args=(q,))
            t.start()
    except:
        print "Error: unable to start thread"



