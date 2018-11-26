# This class describes methods to be used to transfer learning from one neural network to another bigger one
#
# Simply initialize the class with the input network, input grid size, output grid size
# Then call transfer_learning() with specified function variables to transfer the learning
# Returns a list [output neural network, training success]
#
# Created by Ashwin Vinoo om 11-24-2018
# Based on work done by Jeremy Dao
# Version 1.0

# importing all the necessary modules we will use in this class
from torch.optim import Adam
from torch.nn import MSELoss
from torch import Tensor
from Robot import Qnet
from random import randint, shuffle
from math import pow
from numpy import append as np_append, uint8 as np_uint8, array as np_array, empty as np_empty, ndarray as np_ndarray


# ----------------------------------------- Transfer Learning Class Definition ----------------------------------------


# This class holds methods to implement transfer learning
class TransferLearning:
    # This is the constructor for the class
    def __init__(self, input_neural_net, input_grid_dimension, output_grid_dimension):

        # Can't transfer learning if the input neural network has been been trained on greater row numbers
        if input_grid_dimension[0] > output_grid_dimension[0]:
            raise Exception("Transfer learning error: input row count {} is greater than output row count {}"
                            .format(input_grid_dimension[0], output_grid_dimension[0]))

        # Can't transfer learning if the input neural network has been been trained on greater column numbers
        if input_grid_dimension[1] > output_grid_dimension[1]:
            raise Exception("Transfer learning error: input column count {} is greater than output column count {}"
                            .format(input_grid_dimension[1], output_grid_dimension[1]))

        # Storing the input grid dimension within the class
        self.input_grid_dimension = input_grid_dimension

        # Storing the output grid dimension within the class
        self.output_grid_dimension = output_grid_dimension

        # Storing the number of robots used in both neural networks
        self.network_robot_count = int((input_neural_net.linear1.in_features -
                                        input_grid_dimension[0]*input_grid_dimension[1] - 3)/2)

        # Stores the neural network that should train the larger network
        self.input_neural_net = input_neural_net

        # Stores the state size of the input neural network
        self.input_neural_net_state_size = input_neural_net.linear1.in_features

        # Calculating the input state size for the output neural network
        self.output_neural_net_state_size = (input_neural_net.linear1.in_features -
                                             input_grid_dimension[0]*input_grid_dimension[1] +
                                             output_grid_dimension[0]*output_grid_dimension[1])

        # Stores the output neural network. The number of hidden layers is made equivalent to that of input neural net
        self.output_neural_net = Qnet(self.output_neural_net_state_size, self.input_neural_net.linear1.out_features)

    # This function is called to perform transfer learning and return the output neural network
    def transfer_learning(self, max_iterations, accepted_mean_square_error=0.1, batch_size=5000, learning_rate=1e-4):

        # Adam optimizer optimizes the parameters (weights and biases) at the learning rate specified
        output_network_optimizer = Adam(self.output_neural_net.parameters(), lr=learning_rate)

        # This is the row count difference between that of input and output grids
        row_difference = self.output_grid_dimension[0] - self.input_grid_dimension[0]

        # This is the column count difference between that of input and output grids
        column_difference = self.output_grid_dimension[1] - self.input_grid_dimension[1]

        # We cycle through iterations of each batch of training the output neural network until the max iteration
        for _ in range(0, max_iterations):

            # Training list - each element in the list contains [input state for output neural net, target value]
            training_list = []

            # Shuffling through batches and then calculating the Mean square error for the entire batch
            for batch in range(0, batch_size):

                # Creating a matrix to hold the observation state of the input neural network map (0 or 1)
                output_network_known_state = np_array([[randint(0, 1) for _ in range(0, self.output_grid_dimension[1])]
                                                      for _ in range(0, self.output_grid_dimension[0])])

                # This is used to store the robot and target state of the input neural network
                input_network_state = np_empty((self.network_robot_count+1)*2, dtype=np_uint8)

                # Creating positions for all of the robots and target of the input neural network randomly
                for i in range(0, (self.network_robot_count+1)*2, 2):
                    input_network_state[i] = randint(0, self.input_grid_dimension[0]-1)
                    input_network_state[i+1] = randint(0, self.input_grid_dimension[1]-1)

                # Creates a backup copy of the input state
                input_network_state_memory = input_network_state

                # Sliding the input network state window over different sections of the output network state
                for i in range(0, row_difference):
                    for j in range(0, column_difference):

                        # Creating a matrix to hold the observation state of the input neural network map (0 or 1)
                        input_network_known_state = output_network_known_state[i:(i+self.input_grid_dimension[0]),
                                                                               j:(j+self.input_grid_dimension[1])]

                        # This is used to store the robot and target state of the output neural network
                        output_network_state = np_empty((self.network_robot_count+1)*2, dtype=np_uint8)

                        # Extending the input position states across the moving window within the output grid dimensions
                        for k in range(0, (self.network_robot_count+1)*2, 2):
                            output_network_state[k] = input_network_state[k]+i
                            output_network_state[k+1] = input_network_state[k+1]+j

                        # Now we flatten data in the input network state, a 2-D matrix to 1-D and append
                        input_network_state = np_append(input_network_state,
                                                        np_ndarray.flatten(input_network_known_state))

                        # Now we flatten data in the output network state, a 2-D matrix to 1-D and append
                        output_network_state = np_append(output_network_state,
                                                         np_ndarray.flatten(output_network_known_state))

                        # Looping through the 4 possible actions each robot can take and then appending them to state
                        for k in range(0, 4):
                            # Adding an action completes the input state for the input neural network
                            input_network_state_tensor = Tensor(np_append(input_network_state, k))
                            # Adding an action completes the input state for the output neural network
                            output_network_state_tensor = Tensor(np_append(output_network_state, k))
                            # Getting the Q value predicted by the input neural network for the given state
                            input_network_predicted_value = self.input_neural_net.forward(input_network_state_tensor)
                            # Now we know the value the output neural network is to be trained towards for its given
                            # input. Add both of them to the training list so that batch training can occur later
                            training_list.append([output_network_state_tensor, input_network_predicted_value])

                        # Restoring the input state from memory
                        input_network_state = input_network_state_memory

            # Shuffling the training data before feeding it in for training
            shuffle(training_list)
            # Initializing the current MSE loss
            sum_square_error = 0.0
            # Using the batch of state and target data for training the output neural network
            for batch in range(0, batch_size):
                # Obtaining the completed input states for the output neural network
                output_network_state_tensor = training_list[batch][0]
                # Obtaining the target predictions that the output neural network should be trained towards
                predicted_target_value = training_list[batch][1]
                # Getting the Q value predicted by the output neural network for the given input state
                output_network_predicted_value = self.output_neural_net.forward(output_network_state_tensor)
                # Adding the current square error to the sum of square errors
                sum_square_error += pow((output_network_predicted_value - predicted_target_value), 2)
                # Represents the function that can calculate training error
                training_error_function = MSELoss()
                # Our goal is to reduce the mean square error loss between the target prediction and that of network
                training_error = training_error_function(output_network_predicted_value, predicted_target_value)
                # Clears the gradients of all optimized torch tensors
                output_network_optimizer.zero_grad()
                # During the backwards pass, gradients from each replica are summed into the original module
                training_error.backward()
                # Training actually happens here. Performs a single optimization step of weights and biases
                output_network_optimizer.step()

            # Dividing the sum of square errors by the batch size to get the mean square error
            current_mean_square_error = sum_square_error/batch_size

            print(current_mean_square_error)

            # Checks if the MSE for the entire batch is within acceptable levels and then returns the output neural net
            if current_mean_square_error <= accepted_mean_square_error:
                # we return a list where true indicates that we achieved the accepted mean square error criteria
                return [self.output_neural_net, True]

        # Failed to completely train the output neural network. Return a list with second element false to indicate this
        return [self.output_neural_net, False]

# ----------------------------------------------------------------------------------------------------------------------

