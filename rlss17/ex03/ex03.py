import numpy as np
import matplotlib.pyplot as plt
class env:

    def __init__(self):
        self.actions = dict()
        self.transition = dict()

    # add states and available actions t oenvironment
    def add_state(self,state,actions):

        self.actions[state] = actions

    def add_transition(self,old_state,action,new_states,probability_list):
        if len(new_states) == len(probability_list):
            self.transition[(old_state,action)] = [new_states,probability_list]
        else:
            print "Error states count does not match with probability count"

    def doTransition(self,old_state, action):

        if self.actions[old_state] != None:
            if action in self.actions[old_state]:
                transition = self.transition[(old_state, action)]
                return [self.reward(transition[0][0]),np.random.choice(transition[0], 1, p=transition[1])[0]]
            else:
                print "invalid action %s in state %s" % (action,old_state )
        elif self.done(old_state):
            #print "terminate"
            #transition = self.transition[(old_state, action)]
            return [self.reward(old_state),old_state]
        else:
            print "no action in state %s"%(old_state)

    def reward(self,current_state):
        if current_state == "R":
            return 1
        else:
            return 0

    def done(self,current_state):
        if self.actions[current_state] == None:
            return True
        else:
            return  False

class agent:
    def __init__(self,actions,initial_state):
        self.actions = actions
        self.current_state = initial_state
        self.value_function = dict()
        self.alpha = .1
        self.gama = 1

    def reset(self, state):
        self.current_state=state
    def take_action(self):
        # random walk
        return np.random.choice(self.actions, 1, p=[.5,.5])
        # return ["right"]

    def transition(self,new_state):
        self.current_state = new_state

    def TD_0(self,old_state, action, reward, new_state):
        old_value = self.value_function[old_state]
        self.value_function[old_state] = old_value+self.alpha*\
                                                   (reward+self.gama*self.value_function[new_state]-old_value)

if __name__ == "__main__":
    # set up env
    env = env()
    actions = ["left","right"]
    env.add_state("L",None)
    env.add_state("R", None)
    env.add_state("A", actions)
    env.add_state("B", actions)
    env.add_state("C", actions)
    env.add_state("D", actions)
    env.add_state("E",actions)
    env.add_transition("L",None,["L"],[1])
    env.add_transition("R", None, ["R"], [1])
    env.add_transition("A", "left", ["L"], [1])
    env.add_transition("B", "left", ["A"], [1])
    env.add_transition("C", "left", ["B"], [1])
    env.add_transition("D", "left", ["C"], [1])
    env.add_transition("E", "left", ["D"], [1])

    env.add_transition("A", "right", ["B"], [1])
    env.add_transition("B", "right", ["C"], [1])
    env.add_transition("C", "right", ["D"], [1])
    env.add_transition("D", "right", ["E"], [1])
    env.add_transition("E", "right", ["R"], [1])


    # print env.doTransition("R","left")

    #set agent
    agent = agent(actions,"C")
    agent.value_function["A"]=.5
    agent.value_function["B"] = .5
    agent.value_function["C"] = .5
    agent.value_function["D"] = .5

    agent.value_function["E"] = .5
    agent.value_function["R"] = 1
    agent.value_function["L"] = .0
    for i in range(1,500,1):

        # print agent.take_action()
        count = 0


        while True:

            count +=1
            current_state = agent.current_state
            action = agent.take_action()[0]
            transition = env.doTransition(current_state,action)
            reward = transition[0]
            new_state = transition[1]
            # print current_state,action,transition
            agent.current_state = transition[1]
            agent.TD_0(current_state,action,reward,agent.current_state)
            if env.done(agent.current_state):
                break

        # print "%d steps finished"%count
        agent.reset("C")
    print agent.value_function
    states = ["A","B","C","D","E"]
    values = [agent.value_function[s] for s in states]
    plt.plot(values,'b-')
    plt.plot([1./6,2./6,3./6,4./6,5./6],'r')
    plt.show()
