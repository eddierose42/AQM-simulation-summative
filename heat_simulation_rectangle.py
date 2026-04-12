import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

Lx = 0.6
Ly = 0.6

Nx = 50
Ny = 50
D = 2e-5 # constant for air. units m^2 / s
u0 = 20
Tmax = 2000

dx = Lx/(Nx-1)
dy = Ly/(Ny-1)
dt = min(dx**2 / (4*D), dy**2 / (4*D)) #CFL condition

u = np.zeros([Ny, Nx]) + u0
max_iters = int(round(Tmax/dt))

## boundary / initial conditions

F_bottom = lambda x: 100
F_top = lambda x: 0
F_left = lambda y: 100
F_right = lambda y: 0
X = np.linspace(0, Lx, Nx)
Y = np.linspace(0, Ly, Ny)

#u[0, :] = F_bottom(X)
#u[-1, :] = F_top(X)
#u[:, 0] = F_left(Y)
#u[:, -1] = F_right(Y)

#u[20:30, -1] = 50
#u[20:30, 0] = 0


#mpx, mpy = int(Nx/2), int(Ny/2) #midpoint
#u[mpy-5:mpy+5, mpx-5:mpx+5] = 100
u[20:30, 20:30] = 100

def construct_matrix_dirichlet(): # using global variables
    A = np.zeros([Nx*Ny, Nx*Ny])

    A_const = 2 * D * dt * (-1/dx**2 - 1/dy**2) + 1
    Bx = D * dt / dx**2
    By = D * dt / dy**2
    
    for j in range(0, Ny):
        for i in range(0, Nx):
            n = j*Nx + i

            if j == 0 and i == 0: #bottom-left corner
                A[n][n] = 1

            elif j == 0 and i == Nx-1: #bottom-right corner
                A[n][n] = 1

            elif j == Ny-1 and i == 0: #top-left corner
                A[n][n] = 1

            elif j == Ny-1 and i == Nx-1: #top-right corner
                A[n][n] = 1

            elif j == 0: #bottom
                A[n][n] = 1

            elif j == Ny-1: #top
                A[n][n] = 1

            elif i == 0: #left
                A[n][n] = 1

            elif i == Nx-1: #right
                A[n][n] = 1

            else: #interior points
                A[n][n] = A_const #U(i,j)
                A[n][n-Nx] = By #U(i,j-1)
                A[n][n-1] = Bx #U(i-1,j)
                A[n][n+1] = Bx #U(i+1,j)
                A[n][n+Nx] = By #U(i,j+1)

            ## wraparound?
    return A

def construct_matrix_homogenous_neumann():
    A = np.zeros([Nx*Ny, Nx*Ny])

    A_const = 2 * D * dt * (-1/dx**2 - 1/dy**2) + 1
    Bx = D * dt / dx**2
    By = D * dt / dy**2
    
    for j in range(0, Ny):
        for i in range(0, Nx):
            n = j*Nx + i

            if j == 0 and i == 0: #bottom-left corner
                A[n][n] = A_const
                A[n][n+1] = 2*Bx
                A[n][n+Nx] = 2*By

            elif j == 0 and i == Nx-1: #bottom-right corner
                A[n][n] = A_const
                A[n][n-1] = 2*Bx
                A[n][n+Nx] = 2*By

            elif j == Ny-1 and i == 0: #top-left corner
                A[n][n] = A_const
                A[n][n+1] = 2*Bx
                A[n][n-Nx] = 2*By

            elif j == Ny-1 and i == Nx-1: #top-right corner
                A[n][n] = A_const
                A[n][n-1] = 2*Bx
                A[n][n-Nx] = 2*By

            elif j == 0: #bottom
                A[n][n] = A_const
                A[n][n-1] = Bx
                A[n][n+1] = Bx
                A[n][n+Nx] = 2*By

            elif j == Ny-1: #top
                A[n][n] = A_const
                A[n][n-Nx] = 2*By
                A[n][n-1] = Bx
                A[n][n+1] = Bx

            elif i == 0: #left
                A[n][n] = A_const
                A[n][n-Nx] = By
                A[n][n+1] = 2*Bx
                A[n][n+Nx] = By

            elif i == Nx-1: #right
                A[n][n] = A_const
                A[n][n-Nx] = By
                A[n][n-1] = 2*Bx
                A[n][n+Nx] = By

            else: #interior points
                A[n][n] = A_const #U(i,j)
                A[n][n-Nx] = By #U(i,j-1)
                A[n][n-1] = Bx #U(i-1,j)
                A[n][n+1] = Bx #U(i+1,j)
                A[n][n+Nx] = By #U(i,j+1)

    return A

def vec_to_grid(r):
    array = np.zeros([Ny, Nx])
    for n in range(Nx*Ny):
        i = n % Nx
        j = n // Nx
        array[j,i] = r[n]
    return array

def grid_to_vec(u):
    r = np.zeros(Nx*Ny)
    for n in range(Nx*Ny):
        i = n % Nx
        j = n // Nx
        r[n] = u[j,i]
    return r


data = [u.copy()]
A_mat = construct_matrix_homogenous_neumann()
U_vec = grid_to_vec(u)

print(f"\nRunning for {max_iters} iterations")
count = 0


while count < max_iters+1:
    U_vec = A_mat @ U_vec
    new_u = vec_to_grid(U_vec)
    data.append(new_u.copy())
    u = new_u
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