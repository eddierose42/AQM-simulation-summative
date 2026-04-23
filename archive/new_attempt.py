## numerical solver for 2d time-dependent heat equation with neumann boundaries using implicit euler method

import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.animation as anim
from matplotlib.widgets import Slider

Lx = 0.6
Ly = 0.6

Nx = 50
Ny = 50

D = 0.005 # constant for  air = 2e-5 (m^2 / s)
u0 = 20
Tmax = 200

dx = Lx/(Nx-1)
dy = Ly/(Ny-1)
dt = 0.005

g = lambda x,y,t: 0
g2 = lambda x,y,t: 1

def construct_b(g=g, t=0, Nx=Nx, Ny=Ny, dx=dx, dy=dy):
    b = np.zeros(Nx*Ny)
    n_idx = lambda i, j: j*Nx + i

    i = np.arange(0,Nx)
    j = 0
    n = n_idx(i,j)
    x, y = dx*i, dy*j
    b[n] += -2 * dy * g(x,y,t)

    i = np.arange(0,Nx)
    j = Ny-1
    n = n_idx(i,j)
    x, y = dx*i, dy*j
    b[n] += 2 * dy * g(x,y,t)

    i = 0
    j = np.arange(0,Ny)
    n = n_idx(i,j)
    x, y = dx*i, dy*j
    b[n] += -2 * dx * g(x,y,t)

    i = Nx-1
    j = np.arange(0,Ny)
    n = n_idx(i,j)
    x, y = dx*i, dy*j
    b[n] += 2 * dx * g(x,y,t)

    return b

def construct_matrix(Nx=Nx, Ny=Ny, dx=dx, dy=dy, dt=dt):
    A = np.zeros((Nx*Ny, Nx*Ny))

    rx = D * dt / dx**2
    ry = D * dt / dy**2
    A_const = 1 + 2 * (rx + ry)

    for j in range(0, Ny):
        for i in range(0, Nx):
            n = j*Nx + i

            A[n][n] = A_const

            if i == 0: #left
                A[n][n+1] = -2*rx

            elif i == Nx-1: #right
                A[n][n-1] = -2*rx

            else:
                A[n][n-1] = -rx
                A[n][n+1] = -rx
            
            if j == 0: #bottom
                A[n][n+Nx] = -2*ry

            elif j == Ny-1: #top
                A[n][n-Nx] = -2*ry
            
            else:
                A[n][n-Nx] = -ry
                A[n][n+Nx] = -ry

    return A

A = construct_matrix()
A_inv = np.linalg.inv(A)
b = construct_b(g)

u = np.zeros([Ny, Nx]) #+ u0
u[:, 3:7] = 100
data = [u.copy()]

max_iters = Tmax / dt
count = 0

print(f"\nRunning for {int(max_iters)} iterations")

while count < max_iters+1:
    u = (A_inv @ (u.flatten() - b)).reshape(Ny, Nx)
    data.append(u)
    count += 1

heat_change = data[-1].sum() - data[0].sum()
print(f"\nHeat change = {round(heat_change, 1)} ({round((heat_change/data[0].sum())*100, 1)})%\n")


## GUI

fig, axis = plt.subplots()
pcm = axis.pcolormesh(data[0], cmap=plt.cm.jet, vmin=np.array(data).min(), vmax=np.array(data).max())#vmin=-100, vmax=100)
plt.colorbar(pcm, ax=axis)

sliderax = fig.add_axes([0, 0, 0.65, 0.03])
slider = Slider(ax=sliderax, label="T", valmin=0, valmax=max_iters+1, valinit=0) #max_iters+1 = len(data)

animation_running = False
animation_speed = 0.05

def updateplot(val):
    pcm.set_array(data[int(val)])
    axis.set_title("Distribution at t: {:.3f} [s].".format(int(val)*dt))
    
def key_press(event):
    if event.key == " ":
        animate()

def animate():
    global animation_running
    animation_running = not animation_running

    while animation_running and slider.val < max_iters-1:
        slider.set_val(slider.val+1)
        plt.pause(animation_speed)
    
    if animation_running:
        animation_running = False

slider.on_changed(updateplot)
fig.canvas.mpl_connect('key_press_event', key_press)
plt.show()