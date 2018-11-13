import numpy as np
import random as rand
import torch

class Robot:

    def __init__(self, size_x, size_y, obs_radius, nrobot, nhidden, eps, gamma):
        self.dim_x = size_x                   # Dimensions of the grid world
        self.dim_y = size_y
        self.obs_radius = obs_radius          # How far can the robot observe
        self.nrobot = nrobot                  # Number of robots in the world
        self.nhidden = nhidden                # Number of hidden units of the Q net
        self.timestep = 0                     # Current timestep
        self.eps = eps
        self.gamma = gamma

        self.pos = np.zeros(2, dtype=int)    # Position of the robot
        # self.other_pos = [[0,0] for i in range(self.nrobot - 1)]   # Postion of the other rovers
        self.target_pos = [-1, -1]
        self.Qnet = torch.nn.Sequential(torch.nn.Linear(2*nrobot+2+self.dim_x*self.dim_y+1, nhidden), torch.nn.Tanh(),
                    torch.nn.Linear(nhidden, nhidden), torch.nn.Tanh(),
                    torch.nn.Linear(nhidden, 1), torch.nn.Tanh())
        self.Qnet.double()
        self.targ_net = torch.nn.Sequential(torch.nn.Linear(2*nrobot+2+self.dim_x*self.dim_y+1, nhidden), torch.nn.Tanh(),
                    torch.nn.Linear(nhidden, nhidden), torch.nn.Tanh(),
                    torch.nn.Linear(nhidden, 1), torch.nn.Tanh())
        self.targ_net.double()
        self.opt = torch.optim.Adam(self.Qnet.parameters(), lr = 1e-4)

    def reset(self):
        self.pos = [0, 0]
    
    def get_pos(self):
        return self.pos

    def set_pos(self, pos):
        self.pos = pos

    # Inputted state should be the states of all robots in order then the 
    # position of the target then observed state of the world
    def rand_action(self, state, eps):
        if rand.random() < eps:     # Take random action
            # Moves [0, 1, 2, 3] corresponds to up, right, down, left respectively
            moves = [0, 1, 2, 3]
            if self.pos[0] == 0:
                moves.remove(3)
            if self.pos[0] == self.dim_x - 1:
                moves.remove(1)
            if self.pos[1] == 0:
                moves.remove(2)
            if self.pos[1] == self.dim_y - 1:
                moves.remove(0)
            return rand.choice(moves)
        else:       # Use Q network
            acts = np.zeros(3)
            for i in range(3):
                acts[i] = self.Qnet(torch.from_numpy(np.append(state, i)))
            return np.argmax(acts)

    # Function to update the Q network. Note that state should include the states of all of the robots
    # (in order), target_pos, the observed state of the world, then the action (0-3) in that order
    def update_net(self, state, reward):
        x = torch.from_numpy(state)
        pred_q = self.Qnet(x)
        targ_q = reward + self.gamma * self.targ_net(x).detach()
        loss_fn = torch.nn.MSELoss()
        loss = loss_fn(pred_q, targ_q)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

    def check_goal(self, targ):
        if (self.pos == targ).all():
            return True
        return False


class ground_robot(Robot):
    def __init__(self, size_x, size_y, nrobot, nhidden, eps, gamma):
        Robot.__init__(self, size_x, size_y, 1, nrobot, nhidden, eps, gamma)

class UAV(Robot):
    def __init__(self, size_x, size_y, nrobot, nhidden, eps, gamma):
        Robot.__init__(self, size_x, size_y, 4, nrobot, nhidden, eps, gamma)