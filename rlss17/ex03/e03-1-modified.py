# encoding=utf8
import random
# map
map = [1,0,0,0,0,0,1]
# actions
RIGHT = 1
LEFT = -1
ACTIONS = [RIGHT,LEFT]

def validActions(current):
    actions = list(ACTIONS)
    for action in actions:
        if current+action > 6 or current+action<0:
            actions.remove(action)
    return actions

def transition(current,action):
    next = current+action
    if isDone(next) and next == 6:
        reward = 1
    else:
        reward = 0
    return next,reward

def takeAction(current):
    actions = validActions(current)
    return  random.choice(actions)

def actionToSymbol(action):
    if action == LEFT:
        return u"◄"
    elif action == RIGHT:
        return u'►'

def isDone(next):
    if map[next] == 1:
        return True
    else:
        return False

if __name__=="__main__":
    lambda_ = .1
    gama_ = 1
    current = 3
    # for i in range(1,5):
    #     print actionToSymbol(takeAction(current))

    # initialize value
    values = [0,0,0,0,0,0,1]

    # for i in range(0,100,1):
    #     if isDone(current):
    #         break
    #     takeAction(current)
    while True:
        print str(current+1),
        if isDone(current):
            break
        action = takeAction(current)
        old = current
        print actionToSymbol(action),
        current,reward = transition(current,action)
        values[old] += lambda_*()
        print '[%d]'%reward,

