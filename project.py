import numpy as np
import matplotlib.pyplot as plt
from math import cos, sin, radians
from matplotlib.widgets import Slider

steps = 10000
step_size = 0.001

# System variables
g = 9.81
alpha = radians(54.5)
mu = 1.37
#mu = 0.84
m = 0.3342
#k1 = 0.56290714285714285714285714285714
k1 = 56.452
k2 = 0.17
x_end = 0


def function(pos, vel):
    if vel > 0:
        return g * (-mu * cos(alpha) - sin(alpha)) + (1 / m) * (-k1 * pos - k2 * (pos ** 3))
    else:
        return g * (mu * cos(alpha) - sin(alpha)) + (1 / m) * (-k1 * pos - k2 * (pos ** 3))

# Experimental data
data = np.genfromtxt("data3.csv", delimiter=",")
steady_state_pos = data[len(data) - 1][1]
timestamps = [point[0] for point in data]
positions = [(point[1] - steady_state_pos) for point in data]


# Initial values
t_0 = 0
v_0 = 0
x_0 = positions[0]

#x_end = positions[len(positions) - 1]
print(x_0)

time_length = timestamps[len(timestamps) - 1]
h = time_length / steps
#h = 0.002

time, h = np.linspace(0.065, time_length, steps, retstep=True)
position = np.zeros(steps)
velocity = np.zeros(steps)
acceleration = np.zeros(steps)

def calc():
    position = np.zeros(steps)
    velocity = np.zeros(steps)
    acceleration = np.zeros(steps)
    position[0] = x_0

    for n in range(0, steps - 1):
        #acceleration[n + 1] = h* function(position[n]) + acceleration[n]
        velocity[n + 1] = h * function(position[n], velocity[n]) + velocity[n]
        position[n + 1] = h * velocity[n] + position[n]

    return position

#for i in range(0, len(velocity)-7, 6):
#	print(str(velocity[i]) + "\t" + str(velocity[i+1]) + "\t" + str(velocity[i+2]) + "\t" + str(velocity[i+3]) \
#		+ "\t" + str(velocity[i+4]) + "\t" + str(velocity[i+5]) + "\t" + str(velocity[i+5]))


fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.25)

l, = plt.plot(time, calc(), label="Numerical")
plt.plot(timestamps, positions, label="Experimental")
plt.xlabel(r'$t$')
plt.ylabel(r'$x(t)$')
plt.legend(loc="lower right")
plt.grid()

axk1 = plt.axes([0.2, 0.05, 0.65, 0.03])
axk2 = plt.axes([0.2, 0.1, 0.65, 0.03])
axmu = plt.axes([0.2, 0.15, 0.65, 0.03])

sk1 = Slider(axk1, 'k1', 0.001, 60.0, valinit=k1)
sk2 = Slider(axk2, 'k2', 0.0001, 5.0, valinit=k2)
smu = Slider(axmu, 'mu', 0.01, 50.0, valinit=mu)

def update(val):
    global k1, k2, mu
    k1 = sk1.val
    k2 = sk2.val
    mu = smu.val
    l.set_ydata(calc())
    fig.canvas.draw_idle()
sk1.on_changed(update)
sk2.on_changed(update)
smu.on_changed(update)


plt.show()