import numpy as np
import matplotlib.pyplot as plt
from math import cos, sin, radians

steps = 10000
step_size = 0.001

# System variables
g = 9.81
alpha = radians(41)
mu = 1.8
m = 0.2663
k1 = 6.8
k2 = 0.3
x_end = 0


def function(pos):
    return g * (-(mu * cos(alpha)) + sin(alpha)) - (1 / m) * (k1 * pos + k2 * (pos ** 3))


# Experimental data
data = np.genfromtxt("data.txt", delimiter=",")
steady_state_pos = data[len(data) - 1][1]
timestamps = [point[0] for point in data]
positions = [point[1] - steady_state_pos for point in data]

# Initial values
t_0 = 0
v_0 = 0
x_0 = positions[0] - steady_state_pos

#x_end = positions[len(positions) - 1]
print(x_end)

time_length = 8
h = time_length / steps

time = np.linspace(0.8, time_length, steps)
position = np.zeros(steps)
velocity = np.zeros(steps)

position[0] = x_0

for n in range(1, steps - 1):
    velocity[n + 1] = h * function(position[n]) + velocity[n]
    position[n + 1] = h * velocity[n] + position[n]

plt.figure()
plt.plot(time, position, label="Numerical")
plt.plot(timestamps, positions, label="Experimental")
plt.xlabel(r'$t$')
plt.ylabel(r'$x(t)$')
plt.legend(loc="lower right")
plt.grid()
plt.show()
