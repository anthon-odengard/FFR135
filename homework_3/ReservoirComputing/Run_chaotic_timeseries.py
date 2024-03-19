import numpy as np
import matplotlib.pyplot
import pandas as pd

import Reservoir


training_set = np.genfromtxt('training-set.csv', delimiter=',')
validation_set = np.genfromtxt('test-set.csv', delimiter=',')
print(training_set.shape)
input_size = 3
number_of_reservoir_neurons = 500
input_weight_variance = 0.002
reservoir_weight_variance = 2/500



input_weights = np.random.normal(0, np.sqrt(input_weight_variance), (number_of_reservoir_neurons, input_size))
reservoir_connection_weights = np.random.normal(0, np.sqrt(reservoir_weight_variance), size=(number_of_reservoir_neurons,

                                                                              number_of_reservoir_neurons))

reservoir = Reservoir.Reservoir(input_weights, reservoir_connection_weights, 3)

reservoir.train_output_weights(training_set)


outputs = reservoir.iterate_reservoir_state(validation_set, 500)
predictions = pd.DataFrame(outputs)
predictions.to_csv('predicitons.csv', index=False, header=False)
y_coordinates = pd.DataFrame(outputs[1])
print(y_coordinates.transpose())
y_coordinates.to_csv('y_coordinate.csv', index=False, header=False)
print()