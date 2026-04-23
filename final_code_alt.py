## numerical solver for 2d time-dependent heat equation with neumann boundaries using implicit euler method

import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.widgets import Slider
import time

def construct_matrix(Nx, Ny, dx, dy, dt, D):
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

def construct_b(Nx, Ny, dx, dy, dt, D, g1, g2, g3, g4, t=0):
    b = np.zeros(Nx*Ny)
    n_idx = lambda i, j: j*Nx + i
    rx = D * dt / dx**2
    ry = D * dt / dy**2

    # bottom
    i = np.arange(0,Nx)
    j = 0
    n = n_idx(i,j)
    x = dx*i
    b[n] += 4 * ry * dy * g4(x,t)

    # top
    i = np.arange(0,Nx)
    j = Ny-1
    n = n_idx(i,j)
    x = dx*i
    b[n] += - 4 * ry * dy * g2(x,t)  *-1 ## I dont understand why we put -1 here

    # left
    i = 0
    j = np.arange(0,Ny)
    n = n_idx(i,j)
    y = dy*j
    b[n] += 4 * rx * dx * g1(y,t)

    # right
    i = Nx-1
    j = np.arange(0,Ny)
    n = n_idx(i,j)
    y = dy*j
    b[n] += -4 * rx * dx * g3(y,t) * -1 ## I dont understand why we put -1 here
    return b

def RUN_SIMULATION(simulation_data, time_dependent_b=False, measure_heat_change=False):
    Lx, Ly, Nx, Ny, dx, dy, dt, D, g1, g2, g3, g4, max_time, u0 = simulation_data.values()
    X = np.linspace(0,Lx,Nx)
    Y = np.linspace(0,Ly,Ny)

    A = construct_matrix(Nx, Ny, dx, dy, dt, D)
    A_inv = np.linalg.inv(A)
    b = construct_b(Nx, Ny, dx, dy, dt, D, g1, g2, g3, g4, t=0)

    u = np.zeros([Ny, Nx]) + u0(X, Y)
    data = [u.copy()]

    max_iters = max_time / dt
    count = 0

    print(f"\nRunning for {round(max_iters)} iterations")

    while count < max_iters+1:
        if time_dependent_b:
            b = construct_b(Nx, Ny, dx, dy, dt, D, g1, g2, g3, g4, t=count*dt)
        u = (A_inv @ (u.flatten() + b)).reshape(Ny, Nx)
        data.append(u)
        count += 1

    if measure_heat_change:
        heat_change = data[-1].sum() - data[0].sum()
        print(f"\nHeat change = {round(heat_change, 1)} ({round((heat_change/data[0].sum())*100, 1)})%\n")
    
    return data

def GUI(data, Lx, Ly, Nx, Ny, max_time, dt, speed=0.05):
    X = np.linspace(0,Lx,Nx+1)
    Y = np.linspace(0,Ly,Ny+1)
    max_iters = max_time / dt

    fig, axis = plt.subplots()
    pcm = axis.pcolormesh(X, Y, data[0], cmap=plt.cm.jet, vmin=min(0,np.array(data).min()), vmax=max(100, np.array(data).max()))
    axis.set_aspect('equal')
    axis.set_xlabel('X (m)')
    axis.set_ylabel('Y (m)')
    axis.set_title('Heat distribution at t: 0.0 (s)')
    colorbar = plt.colorbar(pcm, ax=axis, shrink=0.6)
    colorbar.set_label('Temperature (°C)', rotation=270, labelpad=10)

    sliderax = fig.add_axes([0, 0, 0.65, 0.03])
    slider = Slider(ax=sliderax, label="T", valmin=0, valmax=max_iters+1, valinit=0) #max_iters+1 = len(data)

    global animation_running
    animation_running = False
    animation_speed = speed

    def updateplot(val):
        pcm.set_array(data[int(val)])
        secs = int(val)*dt
        min, sec = divmod(secs, 60)
        hour, min = divmod(min, 60)
        text = '%d:%02d:%02d' % (hour, min, sec)
        axis.set_title(f"Distribution at t: {text} [hh:mm:ss].")
        
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
    return fig, axis

## set up simulation data

Lx = 5 # m
Ly = 3
height = 2.4
dx = 0.01 # 10cm
dy = 0.01
dt = 0.05 # s

def u0(x, y):
    x, y = np.meshgrid(x, y)
    return np.where((x <= 0.5) & (1 < y) & (y < 2), 200, 20)

sim2 = {
    # dimensions of room in x- and y- directions (m)
    "Lx": Lx,
    "Ly": Ly,

    # grid and time spacing
    "Nx": int((Lx/dx)+1),
    "Ny": int((Ly/dy)+1),
    "dx": dx,
    "dy": dy,
    "dt": 0.5,

    # diffusion constant for  air (m^2 / s)
    "D": 2e-5,

    # boundary conditions - insulated walls
    "g1": lambda a,t:0,
    "g2": lambda a,t:0,
    "g3": lambda a,t:0,
    "g4": lambda a,t:0,

    "max_time": 2*60*60, # length of simulation (s)
    "u0": u0 # initial condition
}

data2 = RUN_SIMULATION(sim2, measure_heat_change=True)
GUI(data2, sim2["Lx"], sim2["Ly"], sim2["Nx"], sim2["Ny"], sim2["max_time"], sim2["dt"], speed=0.0000001)
plt.show()
print(data2[-1].mean() - 20)