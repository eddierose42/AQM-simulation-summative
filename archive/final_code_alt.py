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
    return fig, axis



def left_boundary(y,t): # includes fire/heater
    if not isinstance(y, np.ndarray):
        y = np.array(y)

    t_h = t/1200
    #dudx = 4.614e-05*t_h**6 - 0.002889*t_h**5 + 0.07141*t_h**4 - 0.878*t_h**3 + 5.519*t_h**2 - 15.94*t_h + 14.11

    if t < 1200:
        dudx = dt*3000
    elif t > 2400:
        dudx = dt*-3000
    else:
        dudx=0
    new_y = np.where((y >= 1) & (y <= 2), dudx, y)
    return new_y

def right_boundary(y,t): # includes radiator
    if not isinstance(y, np.ndarray):
        y = np.array(y)

    dudx = np.sin(t)
    #dudx = dt * (1e3 / 0.025 * rad_surface_area +  1e3 / 15 * rad_surface_area) # conduction + convection. radiator is 1kW
    new_y = np.where(y < 1, dudx, y)
    return new_y


Lx = 5 # m
Ly = 3
height = 2.4
dx = 0.1 # 10cm
dy = 0.1
dt = 0.05 # s
Ctop = -2
Cright = 0
Cbottom = 0

## set up simulation data

sim1 = {
    # dimensions of room in x- and y- directions (m)
    "Lx": Lx,
    "Ly": Ly,
    "Nx": int((Lx/dx)+1),
    "Ny": int((Ly/dy)+1),
    "dx": dx,
    "dy": dy,
    "dt": 200,

    "D": 2e-5, # diffusion constant for  air (m^2 / s)

    # boundary conditions
    "g1": left_boundary,
    "g2": lambda x,t: Ctop*dt, # top - heat loss from badly insulated wall
    "g3": right_boundary, # right - insulated wall
    "g4": lambda x,t: Cbottom*dt, # insulated wall

    "max_time": 2000, # length of simulation (s)
    "u0": lambda x, y: 20 # initial condition - set uniform room temperature of 20°C
}

#data1 = RUN_SIMULATION(sim1, time_dependent_b=True)
#print("Finished")
#GUI(data1, sim1["Lx"], sim1["Ly"], sim1["Nx"], sim1["Ny"], sim1["max_time"], sim1["dt"])
#plt.show()
#print("done")


def initial_condition(x, y):
    x, y = np.meshgrid(x, y)
    return np.where((2 < x) & (x < 3) & (1 < y) & (y < 2), 200, 20)


sim2 = {
    # dimensions of room in x- and y- directions (m)
    "Lx": Lx,
    "Ly": Ly,
    "Nx": int((Lx/dx)+1),
    "Ny": int((Ly/dy)+1),
    "dx": dx,
    "dy": dy,
    "dt": 0.5,
    "D": 2e-5, # diffusion constant for  air (m^2 / s)
    # boundary conditions - insulated walls
    "g1": lambda a,t:0,
    "g2": lambda a,t:0,
    "g3": lambda a,t:0,
    "g4": lambda a,t:0,
    "max_time": 3*60*60, # length of simulation (s)
    "u0": initial_condition
}

data2 = RUN_SIMULATION(sim2, measure_heat_change=True)
GUI(data2, sim2["Lx"], sim2["Ly"], sim2["Nx"], sim2["Ny"], sim2["max_time"], sim2["dt"])
plt.show()