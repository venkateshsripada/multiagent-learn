import random
from random import randint
from operator import add
import numpy as np
import math
import time
import torch
import Robot
import matplotlib.pyplot as plt

class Parameters:
    def __init__(self):

        #Rover domain
        self.dim_x = 10
        self.dim_y = 5 #HOW BIG IS THE ROVER WORLD
        self.action_dim = 4
        self.nrover = 1
        self.T = 100
        self.nrollout = 10
        self.obs_radius = 15 #OBSERVABILITY FOR EACH ROVER
        self.act_dist = 1.5 #DISTANCE AT WHICH A POI IS CONSIDERED ACTIVATED (OBSERVED) BY A ROVER
        self.angle_res = 30 #ANGLE RESOLUTION OF THE QUASI-SENSOR THAT FEEDS THE OBSERVATION VECTOR
        self.num_poi = 10 #NUM OF POIS
        self.num_rover = 4 #NUM OF ROVERS
        self.num_timestep = 25 #TIMESTEP PER EPISODE

class GridWorld:
    def __init__(self, size_x, size_y, T, niter, nground, nUAV, targ_pos):
        #Rover domain
        self.dim_x = size_x                             # x dimension of the grid world
        self.dim_y = size_y                             # y dimension of the grid world
        self.nrover = nground + nUAV                    # Number of rovers
        self.nground = nground                          # Number of ground robots
        self.nUAV = nUAV                                # Number of UAV robots
        self.T = T                                      # Length of each episode
        self.timestep = 0                               # Current timestep
        self.niter = niter                              # Number of iterations to train
        self.targ_pos = targ_pos                        # Position of the target
        self.obs_states = np.zeros((size_y, size_x))    # Current observed states. 0=hidden, 1=observed
        self.num_obs = 0                                # Number of states that have currently been observed
        self.rovers = [0 for i in range(self.nrover)]   # Array of Task_Rover objects
        
        # Construct the rovers
        for i in range(nground):
            self.rovers[i] = Robot.ground_robot(self.dim_x, self.dim_y, self.nrover, 64, .1, .9)
        for i in range(nUAV):
            self.rovers[nground + i] = Robot.UAV(self.dim_x, self.dim_y, self.nrover, 64, .1, .9)

    def reset(self):
        self.timestep = 0
        for i in range(self.nrover):
            self.rovers[i].pos = np.zeros(2, dtype=int)
        self.targ_pos = np.array([random.randrange(self.dim_x), random.randrange(self.dim_y)])
        self.num_obs = 0
        

    # Function to visualize the current state of the grid world
    def visualize(self):
        grid = [['x' for _ in range(self.dim_x)] for _ in range(self.dim_y)]
        
        drone_symbol_bank = ['1', '2', '3', '4']
        for i in range(self.dim_x):
            for j in range(self.dim_y):
                if self.obs_states[j, i] == 1:
                    grid[j][i] = '-'
        for i in range(self.nrover):
            rov_pos = self.rovers[i].get_pos()
            x = int(rov_pos[0])
            y = int(rov_pos[1])
            if grid[y][x] == 'x' or grid[y][x] == '-':
                grid[y][x] = str(i)
            else:
                grid[y][x] += str(i)

        # Draw in target ('xT' means the target is hidden)
        x = int(self.targ_pos[0])
        y = int(self.targ_pos[1])
        sym = 'T'
        if self.obs_states[y, x] == 0:   # Target is hidden
            sym = 'xT'
        grid[y][x] = sym

        for row in grid:
            print(row)
        print()

    # Update the observed state of the world given that the $index-th robot made move $move
    # Todo: make more efficient, if know the move the robot made previously, don't have to check
    # the entire radius, just the changed states
    def update_obs(self, move, index):
        obs_len = self.rovers[index].obs_radius
        # Loop through y
        for i in range(max(0, self.rovers[index].pos[1]-obs_len), min(self.dim_y, self.rovers[index].pos[1]+obs_len)):
            # Loop through x
            for j in range(max(0, self.rovers[index].pos[0]-obs_len), min(self.dim_x, self.rovers[index].pos[0]+obs_len)):
                if self.obs_states[i, j] == 0:
                    self.obs_states[i, j] = 1
                    self.num_obs += 1


    # Updates state of the world given the array of robot actions
    # For actions: 0=up, 1=right, 2=down, 3=left
    def step(self, action, visual = False):
        self.timestep += 1

        reached_goal = False
        for i in range(self.nrover):
            move = [-1, 0]
            if action[i] == 0:
                move = [0, -1]
            elif action[i] == 1:
                move = [1, 0]
            elif action[i] == 2:
                move = [0, 1]
            self.rovers[i].pos = self.rovers[i].pos + np.array(move)#list(map(add, self.rovers[i].pos, move))
            self.update_obs(move, i)
            # Check pos limits, make sure not out of bounds
            if self.rovers[i].pos[0] < 0:
                self.rovers[i].pos[0] = 0
            if self.rovers[i].pos[0] >= self.dim_x:
                self.rovers[i].pos[0] = self.dim_x - 1
            if self.rovers[i].pos[1] < 0:
                self.rovers[i].pos[1] = 0
            if self.rovers[i].pos[1] >= self.dim_y:
                self.rovers[i].pos[1] = self.dim_y - 1

            if self.rovers[i].check_goal(self.targ_pos):
                reached_goal = True

        if visual:
            self.visualize()
            input()
        return reached_goal
    
    def reward(self):
        obs_reward = 2*self.num_obs / (self.dim_x*self.dim_y)
        rewards = np.ones(self.nrover)*obs_reward
        # If target pos is known, ground robot gets additional reward
        if (self.obs_states[self.targ_pos[1], self.targ_pos[0]] == 1):
            rewards[0] += (20 - np.linalg.norm(self.rovers[0].pos - self.targ_pos))
        return rewards

    def train(self):
        obs_rewards = np.zeros(self.niter)
        for k in range(self.niter):
            self.reset()
            for j in range(self.T):
                # Get actions
                acts = np.zeros(self.nrover)
                state = self.rovers[0].pos
                for i in range(1, self.nrover):
                    state = np.append(state, self.rovers[i].pos)
                if (self.obs_states[self.targ_pos[1], self.targ_pos[0]] == 0):
                    state = np.append(state, [-1,-1])
                else:
                    state = np.append(state, self.targ_pos)
                state = np.append(state, self.obs_states.flatten())
                for i in range(self.nrover):
                    acts[i] = self.rovers[i].rand_action(state, .1)
                done = self.step(acts)
                if not done:
                    rew = self.reward()
                    for i in range(self.nrover):
                        self.rovers[i].update_net(np.append(state, acts[i]), rew[i])
                else:
                    break
            if k % 10 == 0:
                print("Iteration: " + str(k))
                self.eval()
                for i in range(self.nrover):
                    self.rovers[i].targ_net.load_state_dict(self.rovers[i].Qnet.state_dict())
            obs_rewards[k] = max(self.reward())
            print(obs_rewards[k])
        for i in range(self.nrover):
            torch.save(self.rovers[i].Qnet.state_dict(), "./models/model"+str(i)+".pth")
        return obs_rewards

    def eval(self, visual=False):
        self.reset()
        self.targ_pos = np.array([self.dim_x-1, self.dim_y-1])
        done = False
        counter = 0
        while not done and counter < 100:
            counter += 1
            acts = np.zeros(self.nrover)
            state = self.rovers[0].pos
            for i in range(1, self.nrover):
                state = np.append(state, self.rovers[i].pos)
            if (self.obs_states[self.targ_pos[1], self.targ_pos[0]] == 0):
                state = np.append(state, [-1,-1])
            else:
                state = np.append(state, self.targ_pos)
            state = np.append(state, self.obs_states.flatten())
            for i in range(self.nrover):
                acts[i] = self.rovers[i].rand_action(state, 0.0)
            # print(acts)
            done = self.step(acts, visual)
        print("Time to capture: " + str(self.timestep))


    def render(self):
        # Visualize
        grid = [['-' for _ in range(self.dim_x)] for _ in range(self.dim_y)]

        drone_symbol_bank = ["0", "1", '2', '3', '4', '5', '6', '7', '8', '9', '10', '11']
        # Draw in rover path
        for rover_id in range(self.params.num_rover):
            for time in range(self.params.num_timestep):
                x = int(self.rover_path[rover_id][time][0])
                y = int(self.rover_path[rover_id][time][1])
                # print x,y
                grid[x][y] = drone_symbol_bank[rover_id]

        # Draw in food
        for loc, status in zip(self.poi_pos, self.poi_status):
            x = int(loc[0])
            y = int(loc[1])
            marker = 'I' if status else 'A'
            grid[x][y] = marker

        for row in grid:
            print (row)
        print()

        print ('------------------------------------------------------------------------')

    def test_model(self, path):
        for i in range(self.nrover):
            self.rovers[i].Qnet.load_state_dict(torch.load(path+str(i)+".pth"))
        self.eval(True)


if __name__ == '__main__':
    
    env = GridWorld(10, 10, 100, 50, 1, 3, [9, 9])
    # acts = [1, 2, 2, 1]
    # env.step(acts, True)
    # env.reset()
    rews = env.train()
    plt.plot(range(50), rews)
    plt.xlabel("Iterations")
    plt.ylabel("Final Reward")
    plt.title("Learning curve of DQN")
    plt.draw()
    plt.savefig("./testfig.png")
    # print(rews)
    # env.test_model("./models/model")
 



