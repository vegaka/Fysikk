import numpy as np
import matplotlib.pyplot as plt
from math import cos, sin, radians

steps = 1000
step_size = 0.001

# System variables
g = 9.81
alpha = radians(41)
mu = 2.8
m = 0.2663
k1 = 6.8
k2 = 0.3
x_end = 0


def function(pos, vel):
    if vel > 0:
        return g * (-mu * cos(alpha) - sin(alpha)) + (1 / m) * (-k1 * pos - k2 * (pos ** 3))
    else:
        return g * (mu * cos(alpha) - sin(alpha)) + (1 / m) * (-k1 * pos - k2 * (pos ** 3))

# Experimental data
data = np.genfromtxt("data.txt", delimiter=",")
steady_state_pos = data[len(data) - 1][1]
timestamps = [point[0] for point in data]
positions = [point[1] - steady_state_pos for point in data]


# Initial values
t_0 = 0
v_0 = 0
x_0 = positions[0]

#x_end = positions[len(positions) - 1]
print(x_0)

time_length = 8
h = time_length / steps

time = np.linspace(0.8, time_length, steps)
position = np.zeros(steps)
velocity = np.zeros(steps)
acceleration = np.zeros(steps)

position[0] = x_0

for n in range(0, steps - 1):
	#acceleration[n + 1] = h* function(position[n]) + acceleration[n]
	velocity[n + 1] = h * function(position[n], velocity[n]) + velocity[n]
	position[n + 1] = h * velocity[n] + position[n]

#for i in range(0, len(velocity)-7, 6):
#	print(str(velocity[i]) + "\t" + str(velocity[i+1]) + "\t" + str(velocity[i+2]) + "\t" + str(velocity[i+3]) \
#		+ "\t" + str(velocity[i+4]) + "\t" + str(velocity[i+5]) + "\t" + str(velocity[i+5]))

plt.figure()
plt.plot(time, position, label="Numerical")
#plt.plot(time, velocity, label="Velocity")
#plt.plot(timestamps, positions, label="Experimental")
plt.xlabel(r'$t$')
plt.ylabel(r'$x(t)$')
plt.legend(loc="lower right")
plt.grid()
plt.show()
