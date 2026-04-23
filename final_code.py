## numerical solver for 2d time-dependent heat equation with neumann boundaries using implicit euler method

import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.widgets import Slider

Lx = 5 #m
Ly = 3 #m

D = 2e-5 # constant for  air = 2e-5 (m^2 / s)
u0 = 20 # room temperature (°C)
Tmax = 20000 # time max (s)


Nx = 50
Ny = 30
#Nx = (Lx/dx)+1
#Ny = (Ly/dy)+1

dx = Lx/(Nx-1)
dy = Ly/(Ny-1)
dt = 0.5

k = -0.01*dt # constant for badly insulated wall

g2 = lambda x,y,t: -k # top - heat loss from badly insulated wall
g3 = lambda x,y,t: 0 # right - insulated wall
g4 = lambda x,y,t: 0 # bottom - insulated wall
g1 = lambda x,y,t: 0 # top - heat loss from badly insulated wall

# left - piecewise including log burner

def g1(x,y,t):
    if not isinstance(y, np.ndarray):
        y = np.array(y)
    dxdt =  (dt*5e3) / (-0.025*0.25)    * -1   #35000 Q / -kA (conductivity of air; area=0.25)
    new_y = np.where((y > 1) & (y <= 2), dxdt, y)
    return new_y 

def construct_b(t=0, g1=g1, g2=g2, g3=g3, g4=g4, Nx=Nx, Ny=Ny, dx=dx, dy=dy):
    b = np.zeros(Nx*Ny)
    n_idx = lambda i, j: j*Nx + i

    # bottom
    i = np.arange(0,Nx)
    j = 0
    n = n_idx(i,j)
    x, y = dx*i, dy*j
    b[n] += -2 * dy * g4(x,y,t) 

    # top
    i = np.arange(0,Nx)
    j = Ny-1
    n = n_idx(i,j)
    x, y = dx*i, dy*j
    b[n] += 2 * dy * g2(x,y,t) * -1 ######

    # left
    i = 0
    j = np.arange(0,Ny)
    n = n_idx(i,j)
    x, y = dx*i, dy*j
    b[n] += -2 * dx * g1(x,y,t) 

    # right
    i = Nx-1
    j = np.arange(0,Ny)
    n = n_idx(i,j)
    x, y = dx*i, dy*j
    b[n] += 2 * dx * g3(x,y,t) * -1 ######

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
b = construct_b()

u = np.zeros([Ny, Nx]) + u0

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

X = np.linspace(0,Lx,Nx+1)
Y = np.linspace(0,Ly,Ny+1)

fig, axis = plt.subplots()
pcm = axis.pcolormesh(X, Y, data[0], cmap=plt.cm.jet, vmin=np.array(data).min(), vmax=np.array(data).max())#vmin=-100, vmax=100)
axis.set_aspect('equal')
plt.colorbar(pcm, ax=axis, shrink=0.65)

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