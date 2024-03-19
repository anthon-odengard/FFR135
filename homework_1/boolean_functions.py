import numpy as np
import itertools
from perceptron import Perceptron as perceptron
import matplotlib.pyplot as plt
#sample a boolean function


def generate_binary_inputspace(dimensions):
    binary_inputspace = np.array(list(itertools.product([0, 1], repeat=dimensions)))
    return binary_inputspace


def boolean_function_sampler(number_of_samples, number_of_dimensions):

    input_space = generate_binary_inputspace(number_of_dimensions)
    number_of_samples = min(number_of_samples, np.power(2, pow(2,number_of_dimensions)))
    number_of_entries = input_space.shape[0]
    sampled_functions = np.zeros((number_of_samples, number_of_entries, number_of_dimensions+1))
    sampled_outputs = np.zeros((number_of_samples, number_of_entries,1))
    input_space = generate_binary_inputspace(number_of_dimensions)
    j = 0
    last_element = 0

    while last_element == 0:

        random_output = np.random.randint(2, size=(number_of_entries, 1)) * 2 - 1
        random_function = np.append(input_space, random_output, axis=1)

        if not any((random_output == sample).all() for sample in sampled_outputs):
            sampled_functions[j] = random_function
            sampled_outputs[j] = random_output
            j = j + 1

        last_element = sampled_functions.flatten()[-1]
        if j%10 == 0:
            print(j)

    return sampled_functions


def initiate_weights(number_of_weights):

    distribution_mean = 0
    distribution_variance = 1/number_of_weights
    weight_matrix = np.random.normal(distribution_mean, distribution_variance, number_of_weights)

    return weight_matrix


# Testing
eta = 0.05
threshold = 0
nr_of_samples = np.power(10,4)

weights_2D = initiate_weights(2)
weights_3D = initiate_weights(3)
weights_4D = initiate_weights(4)
weights_5D = initiate_weights(5)
threshold = 0

perceptron_2D = perceptron(weights_2D, threshold)
perceptron_3D = perceptron(weights_3D, threshold)
perceptron_4D = perceptron(weights_4D, threshold)
perceptron_5D = perceptron(weights_5D, threshold)

boolean_funtion_space_2D = boolean_function_sampler(nr_of_samples, 2)
boolean_funtion_space_3D = boolean_function_sampler(nr_of_samples, 3)
boolean_funtion_space_4D = boolean_function_sampler(nr_of_samples, 4)
boolean_funtion_space_5D = boolean_function_sampler(nr_of_samples, 5)

nr_linearly_seperable_2D = perceptron_2D.evaluate_function_space(boolean_funtion_space_2D, weights_2D, threshold, eta)
nr_linearly_seperable_3D = perceptron_3D.evaluate_function_space(boolean_funtion_space_3D, weights_3D, threshold, eta)
nr_linearly_seperable_4D = perceptron_4D.evaluate_function_space(boolean_funtion_space_4D, weights_4D, threshold, eta)
nr_linearly_seperable_5D = perceptron_5D.evaluate_function_space(boolean_funtion_space_5D, weights_5D, threshold, eta)

print(nr_linearly_seperable_2D)
print(nr_linearly_seperable_3D)
print(nr_linearly_seperable_4D)
print(nr_linearly_seperable_5D)


