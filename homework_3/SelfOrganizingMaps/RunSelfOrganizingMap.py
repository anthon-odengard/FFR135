import numpy as np
import sys
import self_organizing_map
np.set_printoptions(threshold=sys.maxsize)
import matplotlib.pyplot as plt
iris_label = np.genfromtxt('iris-labels.csv', delimiter=',')
iris_data = np.genfromtxt('iris-data.csv', delimiter=',')
iris_max = np.max(iris_data.flatten())
training_data = iris_data/iris_max

eta = 0.1
eta_decay = 0.01
sigma = 10
sigma_decay = 0.05
epochs = 10

weight_matrix = np.random.uniform(low=0, high=1, size=(40, 40, 4))

map = self_organizing_map.self_organizing_map(weight_matrix)
heat_map_untrained = map.generate_heat_map(training_data)
map.train_network(training_data, epochs, eta, sigma, sigma_decay,eta_decay)
heat_map_trained = map.generate_heat_map(training_data)

figure, axis = plt.subplots(1, 2)
scatter_1 = axis[0].scatter(heat_map_untrained[:, 0], heat_map_untrained[:, 1], c=iris_label)
scatter_2 = axis[1].scatter(heat_map_trained[:, 0],heat_map_trained[:, 1], c=iris_label)

axis[0].set_xlabel('x1')
axis[0].set_ylabel('x2')
axis[0].legend(*scatter_1.legend_elements(), loc="lower left", title="Classes")
axis[1].set_xlabel('x1')
axis[1].set_ylabel('x2')
axis[1].legend(*scatter_1.legend_elements(), loc="lower left", title="Classes")
axis[0].set_title('Random weights')
axis[1].set_title('Trained weights')
figure.suptitle('Self organizing map of iris data-set', fontsize=16)
plt.show()
