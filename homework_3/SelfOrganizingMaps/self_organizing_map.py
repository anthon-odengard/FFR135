import numpy as np


class self_organizing_map:

    def __init__(self, weight_matrix):
        self.weight_matrix = weight_matrix
        self.output = 0

    def evaluate_network(self, input):
        output = np.matmul(self.weight_matrix, input)
        return output

    def winning_neuron(self, input):
        distance_matrix = self.weight_matrix - input
        distance_matrix = np.linalg.norm(distance_matrix, axis=2)

        return distance_matrix

    def get_minimum_distance_index(self, distance_matrix):
        minimum_index = np.unravel_index(np.argmin(distance_matrix), distance_matrix.shape)
        return minimum_index

    def get_maximum_value_index(self, output):
        maximum_index = np.unravel_index(np.argmax(output), output.shape)
        return maximum_index

    def calculate_euclidean_distance(self, output):
        index_matrix = np.moveaxis(np.mgrid[:self.weight_matrix.shape[0], :self.weight_matrix.shape[0]], 0, -1)
        distance_matrix = index_matrix - output
        distance_matrix = np.linalg.norm(distance_matrix, axis=2)
        return distance_matrix

    def neighbourhood_function(self, sigma, euclidean_distance_matrix):
        euclidean_distance_squared = np.square(euclidean_distance_matrix)
        sigma_factor = -1/(2*np.power(sigma, 2))
        exponent = sigma_factor * euclidean_distance_squared
        neigbourhood_matrix = np.exp(exponent)
        return neigbourhood_matrix

    def update_weights(self, input, eta, neighbourhood_matrix):
        dw = np.zeros(shape=self.weight_matrix.shape)
        size_weight_matrix = len(self.weight_matrix) - 1

        for i in range(0, size_weight_matrix):
            for j in range(0, size_weight_matrix):
                dw[i, j, :] = eta * neighbourhood_matrix[i, j] * (input - self.weight_matrix[i, j, :])

        self.weight_matrix = self.weight_matrix + dw

    def train_network(self, training_set, epochs, eta, sigma, sigma_decay, eta_decay):

        for epoch in range(0, 20):
            print(epoch)
            sigma_decayed = sigma * np.exp(-sigma_decay * epoch)
            eta_decayed = eta * np.exp(-eta_decay * epoch)

            for input in training_set:
                output = self.winning_neuron(input)
                min_index = self.get_minimum_distance_index(output)
                euclidian_matrix = self.calculate_euclidean_distance(min_index)
                neighbourhood_matrix = self.neighbourhood_function(sigma_decayed, euclidian_matrix)
                self.update_weights(input, eta_decayed, neighbourhood_matrix)

    def generate_heat_map(self, inputs):

        heat_map = np.zeros((len(inputs), 2))
        i = 0
        for input in inputs:
            output = self.winning_neuron(input)
            min_index = self.get_minimum_distance_index(output)
            heat_map[i] = min_index
            i = i+1
        return heat_map




