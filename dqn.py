from ple import PLE
from ple.games.flappybird import FlappyBird

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

import numpy as np
import matplotlib.pyplot as plt

from model import DQNModel
from replay_memory import Memory
from param import Param
from agent import Agent

class Env:
    def __init__(self):
        self.game = FlappyBird(pipe_gap=125)
        self.env = PLE(self.game, fps=30,frame_skip=1, \
            display_screen=False, force_fps=False)
        self.env.init()
        
        self.obs_space = len(np.array(list(self.env.getGameState())))
        self.actions = self.env.getActionSet()
    
    def step(self, do):
        action = self.actions[do]
        reward = self.env.act(action)
        done = self.env.game_over()
        obs = self.get_observation()
        return obs, reward, done
    
    def get_observation(self):
        obs = self.env.getGameState()

        return np.array(list(obs.values()))
    
    def get_obs_rgb(self):
        return self.env.getScreenRGB()
    
    def init(self):
        self.env.init()
    
    def reset(self):
        self.env.reset_game()

        return self.get_observation()
        # return self.get_obs_rgb()
    
    def set_display(self, bool):
        self.env.display_screen = bool

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

env = Env()
net = DQNModel(len(env.actions), env.obs_space).to(device)
target_net = DQNModel(len(env.actions), env.obs_space).to(device)
memory = Memory()
optimizer = optim.Adam(net.parameters(), lr=Param.learning_rate)

target_net.load_state_dict(net.state_dict())
agent = Agent(net, target_net, optimizer, memory)

score = 0.0
score_list = []
for i in range(12000):
    epsilon = max(0.01, 0.08 - 0.01 * (i / 200))
    env.reset()
    s = env.get_observation()
    done = False

    while not done:
        a = agent.choose_action(torch.from_numpy(s).float().to(device), epsilon)
        s_prime, r, done = env.step(a)
        done_mask = 0.0 if done else 1.0
        memory.add([s, a, r/100.0, s_prime, done_mask])
        s = s_prime
        score += r

        if done:
            break
    
    if memory.size() > 2000:
        agent.train()
    
    if i % 20 == 0 and i != 0:
        print('n_episode: {}, score: {:.1f}, n_buffer : {}, eps: {:.1f}%'\
            .format(i, score / 20, memory.size(), epsilon*100))
        agent.target_model.load_state_dict(agent.model.state_dict())
        score_list.append(score / 20)
        score = 0.0

torch.save(agent.model.state_dict(), 'trained_model.pt')
