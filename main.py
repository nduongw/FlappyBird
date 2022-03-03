from ple import PLE
from ple.games.flappybird import FlappyBird

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

import numpy as np
import matplotlib.pyplot as plt


learning_rate = 0.002
gamma = 0.98

class Env:
  def __init__(self):
    self.game = FlappyBird(pipe_gap=125)
    self.env = PLE(self.game, fps=30, display_screen=False, force_fps=False)
    self.env.init()
    self.env.getGameState = self.game.getGameState # maybe not necessary

    # by convention we want to use (0,1)
    # but the game uses (None, 119)
    self.action_map = self.env.getActionSet() #[None, 119]

  def step(self, action):
    action = self.action_map[action]
    reward = self.env.act(action)
    done = self.env.game_over()
    obs = self.get_observation()
    # don't bother returning an info dictionary like gym
    return obs, reward, done

  def reset(self):
    self.env.reset_game()
    return self.get_observation()

  def get_observation(self):
    # game state returns a dictionary which describes
    # the meaning of each value
    # we only want the values
    obs = self.env.getGameState()
    return np.array(list(obs.values()))

  def set_display(self, boolean_value):
    self.env.display_screen = boolean_value



class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.data = []
        # self.obs = obs
        # self.action_space = action_space

        self.fc1 = nn.Linear(8, 128)
        # self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(128, 2)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=0)

        return x
    
    def input_data(self, data):
        self.data.append(data)


def train(model, optimizer):
    R = 0
    optimizer.zero_grad()

    for r, prob in model.data[::-1]:
        R = r + gamma * R
        loss = -torch.log(prob) * R
        loss.backward()

    optimizer.step()

    model.data = []

env = Env()
net = Model().to('cuda')
optimizer = optim.Adam(net.parameters(), lr=learning_rate)

# agent = NaiveAgent(env.actions)

score = 0.0

for i in range(10000):
    env.reset()
    s = env.get_observation()
    done = False

    while not done:
        prob = net(torch.from_numpy(s).float().to('cuda'))
        m = Categorical(prob)
        do = m.sample().item()

        s_prime, reward, done = env.step(do)
        # print(reward)
        net.input_data([reward, prob[do]])

        s = s_prime
        score += reward
    
    train(net, optimizer)

    if i % 20 == 0 and i != 0:
        print('episode {} : score : {:.2f}'.format(i, score / 20))
        # total.append(score / 20)
        score = 0

# for i in range(5000):
#    if env.env.game_over():
#            env.reset()

#    observation = env.get_observation()
#    action = agent.pickAction(reward, observation)
#    _, reward, done = env.step(action) 