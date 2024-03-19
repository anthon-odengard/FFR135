import matplotlib.pyplot as plt
import numpy as np
from patterns import *
# Patterns
NUMBER_OF_NEURONS = 160
training_patterns = [x1, x2, x3, x4, x5]
question_patterns = [q1, q2, q3]

def hebb_weight_matrix(patterns):
    patterns = format_patterns(patterns)
    weight_matrix = np.matmul(patterns.transpose(), patterns) * 1/NUMBER_OF_NEURONS
    np.fill_diagonal(weight_matrix, 0)
    return weight_matrix


def format_patterns(patterns):
    pattern_index = 0
    for pattern in patterns:
        patterns[pattern_index] = np.array(pattern).flatten()
        pattern_index = pattern_index+1
    patterns = np.array(patterns)
    return patterns


def calculate_next_state(weight_matrix, state_t):
    state_t = np.array(state_t).flatten()
    state_t = state_t.transpose()
    b_t = np.matmul(weight_matrix, state_t)
    b_t = np.where(b_t == 0, 1, b_t)
    state_t_plus_1 = np.sign(b_t)
    return state_t_plus_1


def plot_state(state):
    state = state.reshape((16,10))
    plt.imshow(state, cmap='binary')
    plt.show()


def type_writer_update(weight_matrix, state): #Klart???
    number_of_rows = len(state)
    new_state = np.array(state).reshape((16, 10))
    #plot_state(new_state)
    for i in range(number_of_rows):
        temporary_state = calculate_next_state(weight_matrix, new_state)
        temporary_state = temporary_state.reshape((16, 10))
        new_state[i] = temporary_state[i]
    #plot_state(new_state)
    return new_state


# Weight matrix
weight_matrix = hebb_weight_matrix(training_patterns)
# Update_states
results = []

for i, pattern in enumerate(question_patterns):
    results.append(type_writer_update(weight_matrix, pattern))

# Print resulting patterns
for result in results:
    print(result.tolist())

# Plot states
for result in results:
    plot_state(result)


