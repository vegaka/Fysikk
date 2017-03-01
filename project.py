import numpy as np
import matplotlib.pyplot as plt
from math import cos, sin, radians

steps = 100
step_size = 0.1

# System variables
g = 9.81
alpha = radians(41)
mu = 0.9
m = 0.2663
k1 = 0.5
k2 = 0.2


def function(pos):
    return g * (mu * cos(alpha) - sin(alpha)) - (1 / m) * (k1 * pos + k2 * (pos ** 3))


# Initial values
t_0 = 0
v_0 = 0
x_0 = 15.06

time = np.zeros(steps + 1)
position = np.zeros(steps + 1)
velocity = np.zeros(steps + 1)

time[0] = t_0
position[0] = x_0
velocity[0] = v_0

for n in range(steps):
    # Euler's method
    x_next = position[n] + step_size * velocity[n]
    func_val = function(position[n])
    v_next = velocity[n] + step_size * func_val

    time[n + 1] = time[n] + step_size
    position[n + 1] = x_next
    velocity[n + 1] = v_next

plt.figure()
plt.plot(time, position)
plt.xlabel(r'$t$')
plt.ylabel(r'$x(t)$')
plt.grid()
plt.show()
