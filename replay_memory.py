from collections import deque
import random
import torch
import numpy as np

from param import Param

class Memory:
    def __init__(self):
        self.buffer = deque(maxlen=Param.buffer_limit)
        self.device = 'cuda'
    
    def add(self, transition):
        self.buffer.append(transition)
    
    def sample(self):
        mini_batch = random.sample(self.buffer, Param.batch_size)
        s_lst, a_lst, r_lst, s_prime_lst, done_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_lst.append([done])

        nps_lst = np.array(s_lst)
        npa_lst = np.array(a_lst)
        npr_lst = np.array(r_lst)
        nps_prime_lst = np.array(s_prime_lst)
        npdone_lst = np.array(done_lst)

        return torch.tensor(nps_lst, dtype=torch.float).to(self.device), \
            torch.tensor(npa_lst).to(self.device), \
            torch.tensor(npr_lst, dtype=torch.float).to(self.device), \
            torch.tensor(nps_prime_lst, dtype=torch.float).to(self.device), \
            torch.tensor(npdone_lst, dtype=torch.float).to(self.device)    
        
        # return torch.tensor(s_lst, dtype=torch.float, device='cuda'), torch.tensor(a_lst, device='cuda'), \
        #     torch.tensor(r_lst, device='cuda'), torch.tensor(s_prime_lst, dtype=torch.float, device='cuda'), \
        #         torch.tensor(done_lst, device='cuda')

    def size(self):
        return len(self.buffer)