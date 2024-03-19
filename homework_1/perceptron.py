import numpy as np


class Perceptron:

    def __init__(self, weights, threshold):
        self.weights = weights
        self.threshold = threshold

    def evaluate_perceptron(self, inputs):
        size_input_space = len(inputs)-1
        b = self.weights * inputs[0:size_input_space]
        b = np.sum(b) - self.threshold
        if b == 0:
            signum_b = 1
        else:
            signum_b = np.sign(b)
        return signum_b

    def evaluate_perceptron_list(self, inputs):
        outputs = np.zeros(len(inputs))
        i = 0
        for input in inputs:
            signum_b = self.evaluate_perceptron(input)
            outputs[i] = signum_b
            i = i + 1
        return outputs

    def update_weights(self, pattern, output, eta):
        size_input_space = len(pattern)-1
        inputs = pattern[0:size_input_space]
        error = pattern[-1] - output
        dw = eta * error * inputs
        self.weights = self.weights + dw

    def evaluate_function_space(self, function_space, weights, threshold, eta):
        print('started evaluation')
        linearly_separable_functions = 0

        for function in function_space:
            self.weights = weights
            self.threshold = threshold
            self.train_network(function, eta, eta, 20)
            outputs = self.evaluate_perceptron_list(function)
            if (function[:, -1] == outputs).all():
                linearly_separable_functions = linearly_separable_functions + 1
            else:
                pass

        return linearly_separable_functions

    def update_thresholds(self, pattern, output, eta):
        error = (pattern[-1] - output)
        d_theta = -eta * error
        self.threshold = self.threshold + d_theta

    def get_thresholds(self):
        return self.threshold

    def get_weights(self):
        return self.weights

    def train_network(self, pattern, eta, epochs):
        for i in range(0, epochs):
            for loc_input in pattern:
                output = self.evaluate_perceptron(loc_input)
                self.update_weights(loc_input, output, eta)
                self.update_thresholds(loc_input, output, eta)
