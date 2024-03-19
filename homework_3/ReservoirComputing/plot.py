
import matplotlib.pyplot as plt
import numpy as np

predictions = np.genfromtxt('predicitons.csv', delimiter=',')
print(predictions.shape)
ax = plt.figure().add_subplot(projection='3d')
ax.plot(*predictions, lw=0.5)
ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
ax.set_zlabel("Z Axis")
ax.set_title("Lorenz Attractor")

plt.show()