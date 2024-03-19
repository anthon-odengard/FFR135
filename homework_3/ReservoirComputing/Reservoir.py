import numpy
import numpy as np
import sklearn.linear_model

class Reservoir:

    def __init__(self, input_weights, reservoir_connection_weight, output_shape):
        self.input_weights = input_weights
        self.reservoir_connection_weights = reservoir_connection_weight
        self.output_weights = 0
        self.reservoir_states = np.zeros((reservoir_connection_weight.shape[0], 1))
        self.training_output = np.zeros(output_shape)

    def update_reservoir_state(self, inputs):
        self.reservoir_states = np.zeros((self.reservoir_connection_weights.shape[0], inputs.shape[0]+1))
        i = 0
        for input in inputs:
            reservoir_contribution = np.matmul(self.reservoir_connection_weights, self.reservoir_states[:, i])
            input_contribution = np.matmul(self.input_weights, input)
            local_field = reservoir_contribution + input_contribution
            reservoir_new_state = np.tanh(local_field)
            self.reservoir_states[:, i+1] = reservoir_new_state
            i = i + 1
        return self.reservoir_states

    def predict_output(self):

        output = np.matmul(self.output_weights, self.reservoir_states[:, -1])
        reservoir_contribution = np.matmul(self.reservoir_connection_weights, self.reservoir_states[:, -1])
        input_contribution = np.matmul(self.input_weights, output)
        local_field = reservoir_contribution + input_contribution
        reservoir_new_state = np.tanh(local_field)
        self.reservoir_states = np.c_[self.reservoir_states, reservoir_new_state]
        return output

    def train_output_weights(self, training_set):
        self.reservoir_states = np.zeros((self.reservoir_states.shape[0], 1))
        training_set_transpose = np.transpose(training_set)
        self.update_reservoir_state(training_set_transpose[:-1, :])
        targets = training_set
        ridge1 = np.matmul(targets, self.reservoir_states.T)
        ridge2 = np.linalg.inv((np.matmul(self.reservoir_states, self.reservoir_states.T) + 0.01 * np.identity(500)))
        ridge = np.matmul(ridge1,ridge2)
        print(ridge)
        self.output_weights = ridge

    def iterate_reservoir_state(self, inputs, number_of_predictions):
        self.reservoir_states = np.zeros((self.reservoir_states.shape[0], 1))

        inputs_transpose = np.transpose(inputs)
        outputs = np.zeros((self.output_weights.shape[0], number_of_predictions))
        self.update_reservoir_state(inputs_transpose[:-1, :])
        t = 0
        while t < number_of_predictions:
            input = self.predict_output()
            outputs[:, t] = input
            t = t + 1
        return outputs





