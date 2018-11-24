# Implementing a Deep Q Network for ROB 538 Final Project
# Created by Ashwin Vinoo om 11/10/2018
# Thanks to Jeremy Dao for a lot of insights developed by reading his codes
#
# Button 'a' toggles the updating of the plot to enhance training speedups
# Button 'z' helps to play/pause the plots. In pause mode the entire code waits for 'x' to be pressed
# Button 'x' helps to move to the next state of the grid world in pause mode only

# importing all the necessary modules
import tensorflow as tf
import matplotlib.patches as plot_patches
from matplotlib.font_manager import FontProperties
from matplotlib import pyplot as plot, colors as colors
import numpy
import random
import keyboard
import tkinter
import tkFileDialog
import itertools
import math

# ---------- Hyper-Parameters ----------
# the problem cases to be executed
problem_cases = [1]
# The row count of the grid upon which agents move
final_grid_row_count = 50
# The column count of the grid upon which agents move
final_grid_column_count = 50
# The different robot types used
robot_types = ['drone', 'medibot']
# The number of robots of each type initialized
robot_count = [1, 1]
# The number of rescuees initialized
rescuee_count = 1
# The reward robots are given for hitting obstacles
obstacle_penalty = -1
# The reward given to the medibot for acquiring a rescuee
medibot_rescuee_acquire_reward = 10
# The reward given to the medibot for bringing the rescuee back to the base
medibot_rescuee_delivery_reward = 10
# The reward given to the drone per unknown cell discovered
drone_area_reward = 1
# Neural network learning rate
neural_network_learning_rate = 0.3
# The exploration vs exploitation trade-off
neural_network_trade_off = 0.15
# The decay rate for the trade-off per episode
neural_network_decay_rate = 0.999
# The minimum value of trade-off
neural_network_trade_off_min = 0.01
# Neural network hidden layer node amount configuration (excludes input size and output size)
neural_network_hidden_layer_configuration = [50, 50, 50]
# --------------------------------------

# ----------------------------------- Robot Class -----------------------------------
class Robot:
    # The init function acts as the constructor for the class
    def __init__(self, robot_type_, robot_position_, grid_dimensions_, robot_count_, observation_radius_,
                 movement_interval_):
        # Specifies the robot type
        self.robot_type_ = robot_type_
        # Specifies the status of the robot. Can be used to denote that the mode of the robot changed due to an event
        self.robot_status_ = 0
        # Initializing the position of the robot
        self.robot_position_ = robot_position_
        # Initializing the grid dimensions
        self.grid_dimensions_ = grid_dimensions_
        # The number of turns after which the drone can move (acts as a proxy for relative speeds amongst robots)
        self.movement_interval_ = movement_interval_
        # The relative observation points that can be viewed by the robot assuming it is at (0,0)
        self.relative_observation_points_ = []
        # Looping through the coordinates around the robot that can fall under its observation radius
        for i in range(math.floor(-observation_radius_), math.ceil(observation_radius_)+1):
            for j in range(math.floor(-observation_radius_), math.ceil(observation_radius_)+1):
                # Check if the point is within the observation radius assuming robot is at (0,0)
                if math.sqrt(math.pow(i, 2)+math.pow(j, 2)) <= observation_radius_:
                    # Add that point to the relative observation points list
                    self.relative_observation_points_.append([i, j])
        # Stores the absolute observation points that can be viewed by the robot from its current position
        self.absolute_observation_points_ = self.relative_observation_points_

    @ staticmethod
    # This function simply returns a random action to be taken up by the robot
    def robot_random_action():
        # Returns a random number between 0 and 3. 0 is up, 1 is down, 2 is left and 3 is right
        return random.randint(0, 3)

    # This function updates the robot status and environment based on its chosen action and returns the reward attained
    def update_robot_status(self, action_, environment_):
        # Accessing global variables stored within the hyper parameter section
        global obstacle_penalty, medibot_rescuee_acquire_reward, medibot_rescuee_delivery_reward, drone_area_reward
        # This stores the reward to be returned. For every movement -1 reward
        reward = -1
        # Robot moves upward
        if action_ == 0:
            # Penalty for hitting the walls
            if self.robot_position_[0] == self.grid_dimensions_[0]:
                reward += obstacle_penalty
            else:
                self.robot_position_[0]+=1
        # Robot moves downward
        elif action_ == 1:
            # Penalty for hitting the walls
            if self.robot_position_[0] == 0:
                reward += obstacle_penalty
            else:
                self.robot_position_[0]-=1
        # Robot moves leftwards
        elif action_ == 2:
            # Penalty for hitting the walls
            if self.robot_position_[1] == self.grid_dimensions_[1]:
                reward += obstacle_penalty
            else:
                self.robot_position_[1]+=1
        # Robot moves rightwards
        elif action_ == 3:
            # Penalty for hitting the walls
            if current_row_ == 0:
                reward += obstacle_penalty
            else:
                self.robot_position_[1]-=1

        # Calculates the number of absolute observation points that the robot sees around it
        observation_point_count_ = len(self.relative_observation_points_)
        # Creates an empty numpy array that can be used to store absolute observation points
        self.absolute_observation_points_ = []
        # Adding other points to the observation points that lie in the grid limits
        for i_ in range(0, observation_point_count_):
            absolute_point = [relative_observation_points_[i_][0] + self.robot_position_[0],
                              relative_observation_points_[i_][1] + self.robot_position_[1]]
            # Checks if the absolute observation points lie outside the grid
            if absolute_point[0] in range(0, self.grid_dimensions[0]) and \
               absolute_point[1] in range(0, self.grid_dimensions[1]):
                # Updating the observable row and column
                self.absolute_observation_points_.append(absolute_point)
                # Setting the grid observation map as visible at that point. This resets each step
                environment_.grid_observation_map_[absolute_point[0], absolute_point[1]] = 1
                # Setting the grid visibility map as known. This resets each episode
                environment_.grid_visibility_map_[absolute_point[0], absolute_point[1]] = 1

        # Gives rewards to the medibot for reaching a rescuee without already carrying another rescuee
        if self.robot_type_ == 'medibot' and self.robot_status == 0 and \
           self.robot_position_ in environment_.rescuee_locations_:
            reward += medibot_rescuee_acquire_reward
            # Changes status to one to denote that a rescuee is being carried
            self.robot_status_ = 1
            # Removes the rescuee from the environment
            environment_.rescuee_locations_.remove(self.robot_position_)
        # Gives rewards to the medibot for bringing the rescuee back to the base
        elif self.robot_type_ == 'medibot' and self.robot_status == 1 and \
             tuple(self.robot_position_) in environment_.base_locations_:
            reward += medibot_rescuee_delivery_reward
            # Changes status to one to denote that a rescuee is being carried
            self.robot_status_ = 0
        # Gives rewards to the drone based on the area covered
        elif self.robot_type_ == 'drone':
            for point_ in self.absolute_observation_points_:
                if environment_.grid_visibility_map_[point_[0],point_[1]] == 0:
                    # Increasing the reward for the new area uncovered
                    reward += drone_area_reward

        # Returns the reward for the action taken
        return reward

# ----------------------------------- DQN Network Class -----------------------------------
class NeuralNetwork:
    def __init__(self):

        # Accessing the global variables from the hyper-parameters section
        global neural_network_learning_rate
        global robot_types
        global neural_network_hidden_layer_configuration

        # ------ Section of the neural network that is dedicated for taking decisions -----
        # Storing the learning rate within the class
        self.neural_net_learning_rate_ = neural_network_learning_rate
        # input state size = 4 for base + 4 for rescuee + 1 for internal state + 4 for each robot type
        self.neural_net_input_size_ = 9 + 2 * len(robot_types)
        # The output size is 4 because each robot can take only 4 actions
        self.neural_net_output_size_ = 4

        # Setting the neural network input placeholders so that values can be inserted in dynamically at runtime
        self.neural_net_input_ = tf.placeholder(tf.float32,[None,self.neural_net_input_size_], name='neural_net_input_')

        # This is needed to store the output from the previous network layer while linking layers together
        neural_net_previous_layer_output_ = self.neural_net_input_
        # Creating a chain of hidden layers from the input to the output
        for i_ in range(0, len(neural_network_hidden_layer_configuration)):
            neural_net_previous_layer_output_ = tf.layers.dense(inputs = neural_net_previous_layer_output_,
                                                        units = neural_network_hidden_layer_configuration[i_],
                                                        activation = tf.nn.elu,
                                                        kernel_initializer=tf.contrib.layers.xavier_initializer())

        # Finally the output is linked to the hidden layer before it. Output is the predicted q values
        self.predicted_q_values_ = tf.layers.dense(inputs = neural_net_previous_layer_output_,
                                                   kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                   units = self.neural_net_output_size_ ,
                                                   activation=None)

        # Now we find which output has maximum value and we use that to decide the action taken
        self.predicted_action_ = tf.argmax(self.predicted_q_values_)

        # ------ Section of the neural network that is dedicated for Q learning purposes -----
        # We create a placeholder to store the action that we have taken in the end. Format [1,0,0,0] or similar
        self.action_taken_ = tf.placeholder(tf.float32, [None, 4], name="actions_taken_")

        # This is the target Q value that we want the neural network to match for the prediction of chosen action
        self.target_q_value_ = tf.placeholder(tf.float32, [None], name="target_q_value")

        # This is the q value of the action taken
        self.q_value_action_taken_ = tf.reduce_sum(tf.multiply(self.predicted_q_values_, self.action_taken_))

        # The loss is the mean square error between our predicted Q value and the target Q value
        self.training_error_ = tf.reduce_mean(tf.square(self.target_q_value_ - self.q_value_action_taken_))

        # Optimizer when invoked adjusts the weights and biases automatically by trying to minimize our
        self.optimizer_ = tf.train.AdamOptimizer(self.neural_net_learning_rate_).minimize(self.training_error_)

# ----------------------------------- Environment Class to Define the Grid World -----------------------------------


# This class is responsible for handling the state information within the grid world
class Environment:
    # The init function acts as the constructor for the class
    def __init__(self, grid_dimensions_, robot_types_, robot_count_, rescuee_count_):
        # Storing the row and column count of the environment to be generated
        self.grid_dimensions_ = grid_dimensions_
        # Initializing the entire map visibility to be unknown initially. Black zones
        self.grid_visibility_map_ = numpy.zeros((grid_row_count_, grid_column_count_))
        # These are the zones that the agents are observing currently. Yellow Zones
        self.grid_observation_map_ = numpy.zeros((grid_row_count_, grid_column_count_))
        # Generating all combinations of rows and columns in the grid world
        grid_combinations_ = list(itertools.product(range(0, grid_row_count_), range(0, grid_column_count_)))
        # Converting a list of tuples to a list of lists for better mutability
        grid_combinations_ = [list(elem) for elem in grid_combinations_]
        # randomly samples from the grid combinations to get the position of the base and rescuees
        self.rescuee_locations_ = random.sample(grid_combinations_, rescuee_count_+1)
        # Getting a random location for the base
        self.base_location_ = self.rescuee_locations.pop()
        # Creating an empty list to which robot class instances may be appended
        self.robot_list_ = []
        # Looping through the different robot types and the number of robots of each type
        for i_ in robot_types_:
            for _ in robot_count_:
                if robot_types_[i_] == 'medibot':
                    # Adding a medibot class object to the robot list
                    robot_list.append(Robot('medibot', self.base_location_, grid_dimensions_, robot_count_, 1.5, 3))
                elif robot_types_[i_] == 'drone':
                    # Adding a drone class object to the robot list
                    robot_list.append(Robot('drone', self.base_location_, grid_dimensions_, robot_count_, 4.5, 1))

    # Reset function to be called after every step is taken within an observation
    def environment_step_reset(self):
        # Resetting the observation map to zeros for every new step
        self.grid_observation_map_ = numpy.zeros((grid_row_count_, grid_column_count_))

    # Returns the grid data for plotting based on the current state of the environment
    def environment_grid_data(self):
        # We add the unknown zones first to the grid data (0 for unknown and 1 for known as per plot initializer)
        grid_data = self.grid_visibility_map_
        # Then we add the observed zones around each robot (14 is for yellow observed zones as per plot initializer)
        grid_data += self.grid_observation_map_*14
        # Now we add the location of each rescuee (16 is for orange rescuee zone as per plot initializer)
        for rescuee_location_ in self.rescuee_locations_:
            grid_data[rescuee_location_[0],rescuee_location_[1]] = 16
        # Now we add in the robots to the grid
        for robot_ in self.robot_list_:
            # Checking if a robot has been added to the same coordinates previously
            if grid_data[robot_.robot_position_[0], robot_.robot_position_[1]] in [8,10]:
                # 20 specifies multiple robots are overlapping as per plot initializer
                grid_data[robot_.robot_position_[0], robot_.robot_position_[1]] = 20
            elif robot_.robot_type_ == 'medibot':
                # 8 specifies medibot as per plot initializer if medibot isn't carrying anyone
                if robot_.robot_status_ == 0:
                    grid_data[robot_.robot_position_[0], robot_.robot_position_[1]] = 8
                else:
                    grid_data[robot_.robot_position_[0], robot_.robot_position_[1]] = 18
            elif robot_.robot_type_ == 'drone':
                # 10 specifies medibot as per plot initializer
                grid_data[robot_.robot_position_[0], robot_.robot_position_[1]] = 10
        # Now we add the location of the base (6 is for dark blue base zone as per plot initializer)
        grid_data[self.base_location_[0], self.base_location_[1]] = 6
        # Returns the grid data based on the current state of the environment
        return grid_data

# ----------------------------------- Grid Map Visualization Block -----------------------------------

# Display pause mode indicates whether a halt is placed on images being drawn
display_pause_mode = False
# This stops the plot from being updated completely, thereby speeding up the rate of training
plot_update_halt_mode = False
# Checks if the 'z' button on the keyboard is released
plot_release = True
# Checks if the 'a' button on the keyboard is released
update_release = True


# Initializes the grid map based on the dimensions
def plot_initializer(grid_row_count, grid_column_count):
    # Close the current figure
    plot.close('all')
    # Creates a matrix of zeros of unsigned integer format to represent the grid being displayed
    grid_map = numpy.zeros((grid_row_count, grid_column_count), numpy.uint8)
    # Loads the python interface to Tkinter GUI
    tkinter_root = tkinter.Tk()
    # Hides the GUI elements
    tkinter_root.withdraw()
    # Creates a colormap in to represent all the information on the screen
    color_map = colors.ListedColormap(['#000000', '#FFFFFF', '#654321', '#003B5C', '#FF0000', '#00FF00', '#0000FF',
                                       '#FFFF00', '#F58231', '#A90000', '#551A8B'])
    # Specifies the boundaries that help divide up the colors during plotting
    bounds = [-1, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
    # Links the colors to sections between two adjacent numbers of the the bounds array
    boundary_norm = colors.BoundaryNorm(bounds, color_map.N)
    # Creates a 2D image to plot
    plot_image = plot.imshow(grid_map, interpolation='nearest', cmap=color_map, norm=boundary_norm)
    # Get the axis handle of the current plot
    axis = plot.gca()
    # Drawing horizontal grid lines
    for x in range(grid_row_count + 1):
        axis.axhline(x - 0.5, lw=1.5, color='black')
    # Drawing vertical grid lines
    for y in range(grid_column_count + 1):
        axis.axvline(y - 0.5, lw=1.5, color='black')
    # Create a title for the plot
    plot.title("Robot Rescue Simulations with Multiple Robot Types", fontsize=18)
    # Specifying the tick interval in the x axis
    plot.xticks(numpy.arange(0, grid_column_count, 1), fontsize=8)
    # Specifying the tick interval in the y axis
    plot.yticks(numpy.arange(0, grid_row_count, 1), fontsize=8)
    # Invert the Y-axis for easier human understandability
    axis.invert_yaxis()
    # The label for the x-axis is specified
    plot.xlabel('Grid World Column Number', fontsize=12)
    # The label for the y-axis is specified
    plot.ylabel('Grid World Row Number', fontsize=12)
    # The different patches we will add to the legend
    unknown_patch = plot_patches.Patch(color='#000000', label='Unknown Area')
    known_patch = plot_patches.Patch(color='#FFFFFF', label='Known Area')
    obstacle_patch = plot_patches.Patch(color='#654321', label='Obstacle')
    base_patch = plot_patches.Patch(color='#003B5C', label='Base')
    medibot_patch = plot_patches.Patch(color='#FF0000', label='Medibot')
    drone_patch = plot_patches.Patch(color='#00FF00', label='Drone')
    bridgebot_patch = plot_patches.Patch(color='#0000FF', label='Bridgebot')
    observable_patch = plot_patches.Patch(color='#FFFF00', label='Observable Area')
    rescuee_patch = plot_patches.Patch(color='#F58231', label='Rescuee')
    medibot_loaded_patch = plot_patches.Patch(color='#A90000', label='Medibot Loaded')
    overlap_patch = plot_patches.Patch(color='#551A8B', label='Bots Overlap')
    # Put a legend below current axis
    axis.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=5, fontsize=9,
                handles=[unknown_patch, known_patch, obstacle_patch, base_patch, medibot_patch, drone_patch,
                         bridgebot_patch, observable_patch, rescuee_patch, medibot_loaded_patch, overlap_patch,
                         mission_patch])
    # Pause the plot for 10 milliseconds to allow for the GUI event loop and new data plotting to occur
    plot.pause(0.01)
    return plot_image


# This function will help to plot the grid and handle keyboard interrupts that can control it
def plot_maintainer(plot_image, grid_data):
    # Global values outside the scope of the function are declare here
    global plot_release
    global display_pause_mode
    global update_release
    global plot_update_halt_mode
    # checks if 'a' has been pressed indicating a stop/start in the updating of the plots completely
    if keyboard.is_pressed('a') and update_release:
        plot_update_halt_mode = not plot_update_halt_mode
        update_release = False
    elif not keyboard.is_pressed('a') and not update_release:
        update_release = True
    # If we are in plot update halt mode don't continue with plotting the grid world
    if not plot_update_halt_mode and update_release:
        # Sets the plot image with the grid data
        plot_image.set_data(grid_data)
        # Displays the plot
        plot.draw()
        # Pause the plot for 50 milliseconds to allow for the GUI event loop and new data plotting to occur
        plot.pause(0.01)
        # Hold the current image in place if 'z' is pressed on the keyboard. Press 'Z' again to cancel this effect
        if keyboard.is_pressed('z') and plot_release:
            display_pause_mode = not display_pause_mode
            plot_release = False
        elif not keyboard.is_pressed('z') and not plot_release:
            plot_release = True
        # What to do in the display pause mode. Press x to move to the next image. Press z to cancel the mode entirely
        while display_pause_mode and plot_release:
            # Toggles the display pause mode off
            if keyboard.is_pressed('z'):
                display_pause_mode = False
                plot_release = False
                break
            if keyboard.is_pressed('a'):
                plot_update_halt_mode = True
                update_release = False
                display_pause_mode = False
                plot_release = False
                break
            if keyboard.is_pressed('x'):
                while keyboard.is_pressed('x'):
                    pass
                break

# ------------------------------------------------------------------------------------------------

# ----------------------------------- Problem One - Deep Q Learning Alone -----------------------------------
# Training the agents to capture the target using the DQN network in a 50X50 grid

# Check if the current problem case is to be executed
if 1 in problem_cases:



# ----------------------------------- Problem Two - Curriculum Learning -----------------------------------

# Check if the current problem case is to be executed
if 1 in problem_cases: