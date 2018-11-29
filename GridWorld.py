import random
from random import randint
from operator import add
import numpy as np
import copy
import math
import time
import torch
import threading
import Robot
import matplotlib.pyplot as plt

class myThread (threading.Thread):
    def __init__(self, rover, state, reward):
        threading.Thread.__init__(self)
        self.rover = rover
        self.state = state
        self.reward = reward

    def run(self):
        # threadLock.acquire()
        self.rover.update_net(self.state, self.reward)
        # threadLock.release()

class GridWorld:
    def __init__(self, size_x, size_y, T, niter, nground, nUAV, targ_pos, filename):
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
        self.filename = filename                        # Filename of model to save to

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
        self.obs_states = np.zeros((self.dim_x, self.dim_y))
        observed = self.update_obs(0, 1)
        self.num_obs = 0
        for s in observed:
            if self.obs_states[s[0], s[1]] == 0:
                self.obs_states[s[0], s[1]] = 1
                self.num_obs = self.num_obs + 1

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
        change_states = set()   # Set of new states that the rover observed
        obs_len = self.rovers[index].obs_radius
        # Loop through y
        for i in range(max(0, self.rovers[index].pos[1]-obs_len), min(self.dim_y, self.rovers[index].pos[1]+obs_len+1)):
            # Loop through x
            for j in range(max(0, self.rovers[index].pos[0]-obs_len), min(self.dim_x, self.rovers[index].pos[0]+obs_len+1)):
                if self.obs_states[i, j] == 0:
                    change_states.add((i, j))
                    # self.obs_states[i, j] = 1
                    # self.num_obs += 1
        return change_states


    # Updates state of the world given the array of robot actions
    # For actions: 0=up, 1=right, 2=down, 3=left
    def step(self, action, visual = False):
        self.timestep += 1
        hit_wall = [False]*self.nrover
        change_states = [None] * self.nrover
        diff_obs = np.zeros(self.nrover)
        for i in range(self.nrover):
            move = [-1, 0]
            if action[i] == 0:
                move = [0, -1]
            elif action[i] == 1:
                move = [1, 0]
            elif action[i] == 2:
                move = [0, 1]
            self.rovers[i].pos = self.rovers[i].pos + np.array(move)#list(map(add, self.rovers[i].pos, move))
            change_states[i] = self.update_obs(move, i)
            # Check pos limits, make sure not out of bounds
            if self.rovers[i].pos[0] < 0:
                self.rovers[i].pos[0] = 0
                hit_wall[i] = True
            if self.rovers[i].pos[0] >= self.dim_x:
                self.rovers[i].pos[0] = self.dim_x - 1
                hit_wall[i] = True
            if self.rovers[i].pos[1] < 0:
                self.rovers[i].pos[1] = 0
                hit_wall[i] = True
            if self.rovers[i].pos[1] >= self.dim_y:
                self.rovers[i].pos[1] = self.dim_y - 1
                hit_wall[i] = True
        reached_goal = self.rovers[0].check_goal(self.targ_pos)
        # Update observe states
        for i in range(self.nrover):
            states = change_states[i]
            for s in states:
                if self.obs_states[s[0], s[1]] == 0:
                    self.obs_states[s[0], s[1]] = 1
                    self.num_obs = self.num_obs + 1
        if visual:
            self.visualize()
            input()
        return reached_goal, change_states, hit_wall

    def global_rew(self, num_obs, dist, hit_wall, do_targ=True):
        out = num_obs
        # out -= 1    # Penalty for doing nothing
        if (self.obs_states[self.targ_pos[1], self.targ_pos[0]] == 1) and do_targ:
            out = -dist
        out += sum(hit_wall)*(-1)
        return out

    # Do difference rewards
    def diff_reward(self, change_states, num_change, dist, hit_wall):
        # Calculate global reward
        rewards = np.ones(self.nrover)*self.global_rew(num_change, dist, hit_wall)
        # Calculate which states each rover found themselves
        for i in range(self.nrover):
            temp_set = change_states[i]
            for j in set(range(self.nrover)) - set([i]):
                temp_set = temp_set - change_states[j]
            diff_obs = len(temp_set)
            do_targ = True
            if (self.targ_pos[1], self.targ_pos[0]) in temp_set:
                do_targ = False
            change_hit = False
            if hit_wall[i] == True:
                hit_wall[i] = False
                change_hit=True
            if i == 0:
                rewards[i] -= self.global_rew(num_change-diff_obs, 0, hit_wall, do_targ)
            else:
                rewards[i] -= self.global_rew(num_change-diff_obs, dist, hit_wall, do_targ)
            rewards[i] -= 1
            if change_hit:
                hit_wall[i] = True
        return rewards

    def eval_fn(self, reached_goal):
        if not reached_goal:
            return 1
        else:
            min_time = np.linalg.norm(self.targ_pos, ord=1)
            #return self.timestep
            return (self.timestep - min_time) / (self.T-min_time)


    def train(self, do_time=False):
        print("start training")
        evals = np.zeros(self.niter)
        eps=.4
        targ_count = 0
        # rewards = np.zeros(self.niter)
        for k in range(self.niter+1):
            self.reset()
            iter_t = time.clock()
            total_rew = 0
            for j in range(self.T):
                # Form the input state
                t = time.clock()
                state = self.rovers[0].pos
                for i in range(1, self.nrover):
                    state = np.append(state, self.rovers[i].pos)
                if (self.obs_states[self.targ_pos[1], self.targ_pos[0]] == 0):
                    state = np.append(state, [-1,-1])
                else:
                    state = np.append(state, self.targ_pos)
                state = np.append(state, self.obs_states.flatten())
                if do_time:
                    print("Formed state: ", time.clock() - t)
                # Get actions
                t = time.clock()
                acts = np.zeros(self.nrover)
                #acts[0] = self.rovers[0].rand_action(state, eps, False)
                for i in range(self.nrover):
                    acts[i] = self.rovers[i].rand_action(state, eps, False)
                if do_time:
                    print("Computed states: ", time.clock() - t)
                #acts[1] = 0
                #acts[2] = 0
                #acts[3] = 0
                prev_dist = np.linalg.norm(self.rovers[0].pos - self.targ_pos, ord=1)
                done, change_states, hit_wall = self.step(acts)
                new_dist = np.linalg.norm(self.rovers[0].pos - self.targ_pos, ord=1)
                if not done:
                    #if targ_count == 200:
                    #    for i in range(self.nrover):
                    #        self.rovers[i].targ_net.load_state_dict(self.rovers[i].Qnet.state_dict())
                    #    targ_count = 0
                    total_states = set()
                    for i in range(self.nrover):
                        total_states = total_states|change_states[i]
                    rew = self.diff_reward(change_states, len(total_states), new_dist-prev_dist, hit_wall)
                    #rew = np.zeros(4)
                    #rew[0] = self.global_rew(0, new_dist-prev_dist, hit_wall)
                    t = time.clock()
                    #self.rovers[0].update_net(np.append(state, acts[0]), rew[0])
                    true_state = state
                    if self.rovers[0].do_pad:
                        true_state = self.rovers[0].pad_state(state)
                    for i in range(self.nrover):
                        self.rovers[i].update_net(np.append(true_state, acts[i]), rew[i])
                    if do_time:
                        print("Updated networks: ", time.clock()-t)
                    #targ_count += 1
                else:
                    break
            for i in range(self.nrover):
                self.rovers[i].targ_net.load_state_dict(self.rovers[i].Qnet.state_dict())
            if k % 5 == 0:
                eval_t = time.clock()
                evals[int(k/10)] = self.eval()
                print("EVAL OF ALL TARG_POS: "+str(evals[int(k/10)])+"\t Time: "+str(time.clock()-eval_t))
                for i in range(self.nrover):
                    torch.save(self.rovers[i].Qnet.state_dict(), "./models/"+self.filename+str(i)+".pth")
            # evals[k] = self.eval_fn(done)
            eps = eps*.98
            # rewards[k] = total_rew
            print("Iteration " + str(k) + ": Eval = " + str(self.timestep)+"\tTime = " + str(round(time.clock() - iter_t, 4))
                    +"\tTarg_pos: [" + str(self.targ_pos[0])+", "+str(self.targ_pos[1])+"]")
        for i in range(self.nrover):
            torch.save(self.rovers[i].Qnet.state_dict(), "./models/"+self.filename+str(i)+".pth")
        return evals

    def eval(self, visual=False):
        counter = 0
        for i in range(self.dim_x):
            for j in range(self.dim_y):
                self.reset()
                self.targ_pos = np.array([i, j])
                done = False
                if [i, j] == [0, 0]:
                    done = True
                while not done and self.timestep < 100:
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
                        acts[i] = self.rovers[i].rand_action(state, 0, False)
                    #print(acts)
                    done, change_states, hit_wall = self.step(acts, visual)
                if done:
                    counter += 1
        return counter


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

def train_whole(loadfile = ""):
    filename = "15"
    env = GridWorld(15, 15, 350, 250, 1, 3, [14, 14], filename)
    if loadfile:
        for i in range(env.nrover):
            env.rovers[i].Qnet.load_state_dict(torch.load(loadfile+str(i)+".pth"))
    print(filename)
    print("T: "+str(env.T)+"\tniter: "+str(env.niter))
    rews = env.train()
    plt.plot(range(int(env.niter/5)), rews)
    plt.xlabel("Iterations")
    plt.ylabel("Final Reward")
    plt.title("Learning curve of DQN")
    plt.draw()
    plt.savefig(filename+".png")

if __name__ == '__main__':

    #env = GridWorld(10, 10, 200, 100, 1, 3, [9, 9])
    #print(env.rovers[0].pos[0], env.rovers[0].pos[1])
    #acts = [0, 1, 1, 0]
   #  for i in range(10):
    # env.step(acts)
      # print(env.reward())
    torch.set_num_threads(2)
    start_t = time.clock()
    train_whole()
    print("Total time: ", time.clock()-start_t)
    #env = GridWorld(10, 10, 200, 100, 1, 3, [9, 9])
    #env.test_model("./models/big_model")




