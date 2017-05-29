from ple.games.flappybird import FlappyBird
import numpy as np
from ple import PLE
import random
import time
from matplotlib import pyplot as plt

def getState(state):
    temp = (
        int(state["player_vel"]),
        int(state["next_pipe_dist_to_player"]/20),
        int((state["player_y"] - (state["next_pipe_top_y"]))/20)+40

    )

    return temp

def getFeature(state):
    feature = [1]

    return feature

def eps_greedy(current_state,actions,q_value,eps=.0):
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

if __name__ == "__main__":
    # initialize game
    game = FlappyBird()
    p = PLE(game, display_screen=True)
    actions = p.getActionSet()
    p.init()

    q_value = {}
    e = {}
    q = np.zeros((30, 100, 2))
    for i in range(30):
        for j in range(100):
            for action in actions:
                q_value[((i,j),action)] = 0.
                e[((i,j),action)] = 0.

    gamma = 1.
    alpha = .2
    lambda_ = .8
    alive_time = 0
    alive_list = []

    cS = getState(game.getGameState())
    for i in range(1000):
        alive_time += 1
        print cS,
        action = eps_greedy(cS,actions,q_value)
        action = 119
        print action
        p.act(action)
        e[(cS, action)] += 1
        if p.game_over():
            reward = -1000
        else:
            reward = 1

        nS = getState(game.getGameState())
        maxAction = eps_greedy(nS,actions,q_value)
        diff = reward + gamma * q_value[(nS, maxAction)] - q_value[(cS, action)]
        # backup
        for i in range(30):
            for j in range(100):
                for k in actions:
                    q_value[((i, j), k)] += alpha * diff * e[((i, j), k)]
                    # q[i][j][k] += alpha*diff*e[((i,j),k)]

                    e[((i, j), k)] *= gamma * lambda_



        q_value[(cS, action)] += alpha*diff
        cS = nS
        # cS = getState(game.getGameState())
        if p.game_over():
            alive_list.append(alive_time)
            alive_time = 0
            p.reset_game()

        # if i>4000:
        #     time.sleep(.01)

    for i in range(30):
        for j in range(100):
            for action in actions:
                if action == 119:
                    a = 0
                else:
                    a = 1
                q[i][j][a] = q_value[((i,j),action)]
    print len([key for key in q_value if q_value[key]!=0])
    plt.figure()
    plt.plot(alive_list)
    #print q
    plt.pcolor(q.max(2))
    plt.show()

