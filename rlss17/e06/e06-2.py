from ple.games.flappybird import FlappyBird
import numpy as np
from ple import PLE

class NaiveAgent():
    def __init__(self, actions):
        self.actions = actions
    def pickAction(self, reward, obs):
        return self.actions[np.random.randint(0, len(self.actions))]

game = FlappyBird()
p = PLE(game, display_screen=True)
agent = NaiveAgent(p.getActionSet())
p.init()
reward = p.act(p.NOOP)
for i in range(1500):
    obs = p.getScreenRGB()
    print game.getGameState()
    reward = p.act(agent.pickAction(reward, obs))