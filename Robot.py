import numpy as np
import math
import random as rand
import copy
import torch
from torch.autograd import Variable

class Qnet(torch.nn.Module):
    def __init__(self, D_in, H):
        super(Qnet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, H)
        self.linear3 = torch.nn.Linear(H, 1)
        
        torch.nn.init.normal_(self.linear1.weight)
        torch.nn.init.normal_(self.linear1.bias)
        torch.nn.init.normal_(self.linear2.weight)
        torch.nn.init.normal_(self.linear2.bias)
        torch.nn.init.normal_(self.linear3.weight)
        torch.nn.init.normal_(self.linear3.bias)

    def forward(self, x):
        tanh_act = torch.nn.Tanh()
        relu_act = torch.nn.ReLU()
        mid1 = self.linear1(x)
        act1 = tanh_act(mid1)
        mid2 = self.linear2(act1)
        act2 = tanh_act(mid2)
        mid3 = self.linear3(act2)
        act3 = relu_act(mid3)

        return act3

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
        self.max_val = (size_x*size_y+size_x+size_y)

        self.pos = np.zeros(2, dtype=int)    # Position of the robot
        self.target_pos = [9, 9]
        self.Qnet = Qnet(2*nrobot+2+self.dim_x*self.dim_y+1, nhidden)
        self.targ_net = Qnet(2*nrobot+2+self.dim_x*self.dim_y+1, nhidden)
        # self.Qnet = torch.nn.Sequential(
        #             torch.nn.Linear(2*nrobot+2+self.dim_x*self.dim_y+1, nhidden),
        #             torch.nn.Tanh(),
        #             torch.nn.Linear(nhidden, nhidden), 
        #             torch.nn.Tanh(),
        #             torch.nn.Linear(nhidden, 1), 
        #             torch.nn.ReLU()
        #             )
        # for param in self.Qnet.parameters():
            # torch.nn.init.normal_(param)
        # self.targ_net = torch.nn.Sequential(torch.nn.Linear(2*nrobot+2+self.dim_x*self.dim_y+1, nhidden), torch.nn.Tanh(),
        #             torch.nn.Linear(nhidden, nhidden), torch.nn.Tanh(),
        #             torch.nn.Linear(nhidden, 1), torch.nn.ReLU())
        # for param in self.targ_net.parameters():
        #     torch.nn.init.normal_(param)
        self.opt = torch.optim.Adam(self.Qnet.parameters(), lr = 1e-4)
        self.buff_state = [torch.Tensor(2*nrobot+2+self.dim_x*self.dim_y+1) for i in range(2000)]
        self.buff_reward = [torch.Tensor(1) for i in range(2000)]
        self.buff_count = 0
        self.buff_filled = False

    def reset(self):
        self.pos = [0, 0]
    
    def get_pos(self):
        return self.pos

    def set_pos(self, pos):
        self.pos = pos

    # Inputted state should be the states of all robots in order then the 
    # position of the target then observed state of the world
    def rand_action(self, state, eps, do_soft=True):
        if do_soft:
            # Soft max
            acts = np.zeros(4)
            # Get outputs, normalize to [0, self.max_val], then take exp
            for i in range(4):
                acts[i] = np.exp(self.forward(self.Qnet, torch.Tensor(np.append(state, i))).data.numpy()/self.max_val)
            # acts = np.exp(self.forward(self.Qnet, torch.Tensor(state)).data.numpy()/self.max_val)
            acts = acts / sum(acts)
            sum_acts = np.zeros(4)
            sum_acts[0] = acts[0]
            for i in range(1,4):
                sum_acts[i] = acts[i] + sum_acts[i-1]
            samp = rand.random()
            # output_act = 0
            for i in range(4):
                if samp < sum_acts[i]:
                    return i
            #         output_act = i
            #         break
            # return output_act
        else:
            if rand.random() < eps:     # Take random action
                # Moves [0, 1, 2, 3] corresponds to up, right, down, left respectively
                moves = [0, 1, 2, 3]
                if self.pos[0] == 0:
                    moves.remove(3)
                if self.pos[0] == self.dim_x - 1:
                    moves.remove(1)
                if self.pos[1] == 0:
                    moves.remove(0)
                if self.pos[1] == self.dim_y - 1:
                    moves.remove(2)
                return rand.choice(moves)
            else:       # Use Q network
                acts = np.zeros(4)
                for i in range(4):
                    acts[i] = self.forward(self.Qnet, torch.Tensor(np.append(state, i))).data.numpy()
                # acts = self.forward(self.Qnet, torch.Tensor(state)).data.numpy()
                #print(acts)
                max_act = np.argmax(acts)
                return max_act

    # Function to update the Q network. Note that state should include the states of all of the robots
    # (in order), target_pos, the observed state of the world, then the action (0-3) in that order
    def update_net(self, state, reward):
        # Update buffer
        self.buff_state[self.buff_count] = torch.Tensor(state)
        self.buff_reward[self.buff_count] = torch.Tensor((reward,))
        self.buff_count = self.buff_count + 1
        if self.buff_count == 1000:
            self.buff_filled = True
            self.buff_count = 0
        batch_size = 128
        choice_lim = len(self.buff_state)
        if not self.buff_filled:
            choice_lim = self.buff_count
            if self.buff_count < batch_size:
                batch_size = self.buff_count
        # Sample batch
        samp_ind = np.random.choice(choice_lim, batch_size, replace=False)
        batch_states = [None]*batch_size
        batch_rewards = [None]*batch_size
        for i in range(batch_size):
            batch_states[i] = self.buff_state[samp_ind[i]]
            batch_rewards[i] = self.buff_reward[samp_ind[i]] 
        # Update network
        loss_fn = torch.nn.MSELoss()
        for i in range(batch_size):
            pred_q = self.forward(self.Qnet, batch_states[i])
            targ_q = batch_rewards[i] + self.gamma * self.forward(self.targ_net, batch_states[i])
            loss = loss_fn(pred_q, targ_q)        
            self.opt.zero_grad()
            loss.backward()
            prev_w = np.copy(self.Qnet.state_dict()['linear3.weight'].data.numpy())
            self.opt.step()
            new_w = np.copy(self.Qnet.state_dict()['linear3.weight'].data.numpy())
            diff = np.linalg.norm(new_w-prev_w)
            if torch.norm(list(self.Qnet.parameters())[0].grad) == 0:
                print("WARNING: GRAD IS ZERO")
            if torch.any(torch.isnan(self.Qnet.linear3.weight)) == True:
                print("IS NAN")
                for param in self.Qnet.parameters():
                    print(param.grad)
                print("loss: ", loss)
                print("pred_q: ", pred_q)
                print("targ_q: ", targ_q)
                sleep(10)

    def forward(self, model, state):
        # out = model(state)
        # return out*self.max_val
        return model(state)


    def check_goal(self, targ):
        if (self.pos == targ).all():
            return True
        return False
    
    def grad_norm(self):
        def hook(grad):
            print(torch.norm(grad))
        return hook


class ground_robot(Robot):
    def __init__(self, size_x, size_y, nrobot, nhidden, eps, gamma):
        Robot.__init__(self, size_x, size_y, 1, nrobot, nhidden, eps, gamma)

class UAV(Robot):
    def __init__(self, size_x, size_y, nrobot, nhidden, eps, gamma):
        Robot.__init__(self, size_x, size_y, 4, nrobot, nhidden, eps, gamma)
