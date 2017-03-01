import numpy as np
import matplotlib.pyplot as plt
from math import cos, sin, radians

steps = 100000
step_size = 0.01

# System variables
g = 9.81
alpha = radians(41)
mu = 62.083
m = 0.2663
k1 = 10.8
k2 = 11.05


def function(pos, vel):
    return g * (mu * cos(alpha) - sin(alpha)) - (1 / m) * (k1 * pos + k2 * (pos ** 3))


# Initial values
t_0 = 0
v_0 = 0
x_0 = 15.06

time_length = 8
h = time_length / steps

time = np.linspace(0.8, time_length, steps)
position = np.zeros(steps)
velocity = np.zeros(steps)

for n in range(1, steps - 1):
    velocity[n + 1] = h*function(position[n], velocity[n]) + velocity[n]
    position[n + 1] = h*velocity[n] + position[n]

# Experimental data
data = np.genfromtxt("data.txt", delimiter=",")
timestamps = [point[0] for point in data]
positions = [(point[1] - x_0) for point in data]

plt.figure()
plt.plot(time, position, label="Numerical")
plt.plot(timestamps, positions, label="Experimental")
plt.xlabel(r'$t$')
plt.ylabel(r'$x(t)$')
plt.legend(loc="lower right")
plt.grid()
plt.show()
