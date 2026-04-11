## from https://github.com/Younes-Toumi/Youtube-Channel/blob/main/Simulation%20with%20Python/Heat%20Equation/heatEquation2D.py

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.widgets import Button, Slider

# Defining our problem

a = 110
length = 50 #mm
time = 20 #seconds
nodes = 50

# Initialization 

dx = length / (nodes-1)
dy = length / (nodes-1)

dt = min(   dx**2 / (4 * a),     dy**2 / (4 * a))

t_nodes = int(time/dt) + 1

u = np.zeros((nodes, nodes)) + 30 # Plate is initially as 20 degres C

# Boundary Conditions 

u[0, :] = np.linspace(-100, 100, nodes)
u[-1, :] = np.linspace(-100, 100, nodes)
u[:, 0] = np.linspace(0, -100, nodes)
u[:, -1] = np.linspace(0, -100, nodes)

#L = 50
#u[L-15:L-1,L-1] = 40
#u[0, 40:] = 5
#u[5:15,0] = 90

# Visualizing with a plot

fig, axis = plt.subplots()
pcm = axis.pcolormesh(u, cmap=plt.cm.jet, vmin=-100, vmax=100)
plt.colorbar(pcm, ax=axis)

# Simulating

counter = 0

data = []

while counter < time :

    w = u.copy()
    try:
        for i in range(1, nodes - 1):
            for j in range(1, nodes - 1):

                dd_ux = (w[i-1, j] - 2*w[i, j] + w[i+1, j])/dx**2
                dd_uy = (w[i, j-1] - 2*w[i, j] + w[i, j+1])/dy**2

                u[i, j] = dt * a * (dd_ux + dd_uy) + w[i, j]

    except KeyboardInterrupt:
        print(counter, counter/time)
    
    counter += dt
    data.append(u.copy())

sliderax = fig.add_axes([0, 0, 0.65, 0.03])
slider = Slider(ax=sliderax, label="T", valmin=0, valmax=time, valinit=0)

def updateplot(var):
    idx = int((slider.val/time) * (len(data)-1))
    axis.set_title("Distribution at t: {:.3f} [s].".format(slider.val))

slider.on_changed(updateplot)
plt.show()