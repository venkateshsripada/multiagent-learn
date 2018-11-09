import numpy as np
import torch

class Robot:

    def __init__(self, size_x, size_y, obs_radius, nrobot, nhidden):
        self.dim_x = size_x                   # Dimensions of the grid world
        self.dim_y = size_y
        self.obs_radius = obs_radius          # How far can the robot observe
        self.nrobot = nrobot                  # Number of other robots in the world
        self.nhidden = nhidden                # Number of hidden units of the Q net
        self.timestep = 0                     # Current timestep

        self.pos = [0,0]    # Position of the robot
        self.other_pos = [[0,0] for i in range(self.nrobot - 1)]   # Postion of the other rovers
        self.target_pos = [-1, -1]
        self.Qnet = torch.nn.Sequential(torch.nn.Linear(2*nrobot+self.dim_x*self.dim_y+2, nhidden), torch.nn.Tanh(),
                    torch.nn.Linear(nhidden, nhidden), torch.nn.Tanh(),
                    torch.nn.Linear(nhidden, 4), torch.nn.Tanh())

    def reset(self):
        self.pos = [0, 0]
    
    def get_pos(self):
        return self.pos

    def set_pos(self, pos):
        self.pos = pos

    def rand_action(self, pos):
        # Moves [0, 1, 2, 3] corresponds to up, right, down, left respectively
        moves = [0, 1, 2, 3]
        if pos[0] == 0:
            moves.remove(3)
        if pos[0] == self.dim_x - 1:
            moves.remove(1)
        if pos[1] == 0:
            moves.remove(2)
        if pos[1] == self.dim_y - 1:
            moves.remove(0)
        return random.choice(moves)
        direct = random.choice(moves)
      

    def check_goal(self):
        for i in range(self.nrover):
            if self.rover_pos[i] == self.target_pos:
                return True
        return False


class ground_robot(Robot):
    def __init__(self, size_x, size_y, nrobot, nhidden):
        Robot.__init__(self, size_x, size_y, 1, nrobot, nhidden)

class UAV(Robot):
    def __init__(self, size_x, size_y, nrobot, nhidden):
        Robot.__init__(self, size_x, size_y, 4, nrobot, nhidden)