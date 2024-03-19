import numpy as np


class FF_neural_network:

    def __init__(self, weights_hidden_layer, weights_output_layer, thresholds_hidden_layer, thresholds_output_layer):

        self.weights_hidden_layer = weights_hidden_layer
        self.weights_output_layer = weights_output_layer
        self.thresholds_hidden_layer = thresholds_hidden_layer
        self.thresholds_output_layer = thresholds_output_layer
        self.dw_hidden_layer = 0
        self.dw_output_layer = 0

    def get_weights(self):
        return self.weights_hidden_layer, self.weights_output_layer

    def get_thresholds(self):
        return self.thresholds_hidden_layer, self.thresholds_output_layer

    def feed_forward(self, pattern):
        input = np.expand_dims(pattern[0:-1], axis=1)
        local_fields_hidden_layer = np.matmul(self.weights_hidden_layer, input) - self.thresholds_hidden_layer
        hidden_layer_activation = np.tanh(local_fields_hidden_layer)
        local_field_output_layer = np.matmul(self.weights_output_layer, hidden_layer_activation) - self.thresholds_output_layer
        output = np.tanh(local_field_output_layer)

        return input, local_fields_hidden_layer, hidden_layer_activation, local_field_output_layer, output

    def tanh_prime(self, local_fields):
        tanh = np.tanh(local_fields)
        tanh_prime = np.ones((local_fields.shape)) - np.power(tanh, 2)
        return tanh_prime

    def output_error(self, target, output, local_field):
        error = (target-output) * self.tanh_prime(local_field)
        return error

    def update_weighs(self, dw, w):
        new_weights = dw + w
        return new_weights

    def update_threshold(self, d_theta, theta):
        new_threshold = theta + d_theta
        return new_threshold

    def backpropagation_mini_batch(self, patterns, eta, alpha):
        dw_hidden_layer = 0
        dw_output_layer = 0
        d_theta_hidden_layer = 0
        d_theta_output_layer = 0
        for pattern in patterns:
            inputs, local_fields_hidden_layer, hidden_layer_activation, local_fields_output_layer, output = self.feed_forward(pattern)
            target = pattern[-1]
            output_error = self.output_error(target, output, local_fields_output_layer)
            transpose_output_weights = np.transpose(self.weights_output_layer)
            error_hidden_layer = np.multiply((transpose_output_weights * output_error), self.tanh_prime(local_fields_hidden_layer))

            dw_hidden_layer = dw_hidden_layer + eta * np.multiply(error_hidden_layer, np.transpose(inputs))
            dw_output_layer = dw_output_layer + eta * output_error * np.transpose(hidden_layer_activation)
            d_theta_hidden_layer = d_theta_hidden_layer -eta * error_hidden_layer
            d_theta_output_layer = d_theta_output_layer -eta * output_error

        self.weights_hidden_layer = self.update_weighs(dw_hidden_layer, self.weights_hidden_layer) + self.dw_hidden_layer * alpha
        self.weights_output_layer = self.update_weighs(dw_output_layer, self.weights_output_layer) + self.dw_output_layer * alpha
        self.dw_hidden_layer = dw_hidden_layer
        self.dw_output_layer = dw_output_layer
        self.thresholds_hidden_layer = self.update_threshold(d_theta_hidden_layer, self.thresholds_hidden_layer)
        self.thresholds_output_layer = self.update_threshold(d_theta_output_layer, self.thresholds_output_layer)
        
        