import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

Lx = 0.6
Ly = 0.6

Nx = 50
Ny = 50
D = 2e-5 # constant for air. units m^2 / s
u0 = 20
Tmax = 4000

dx = Lx/(Nx-1)
dy = Ly/(Ny-1)
dt = 200 #1 / (2*D* (1/dx**2 + 1/ dy**2)) #CFL condition

u = np.zeros([Ny, Nx]) + u0
max_iters = int(round(Tmax/dt))

## boundary / initial conditions

F_bottom = lambda x: 100
F_top = lambda x: 100
F_left = lambda y: 100
F_right = lambda y: 0
X = np.linspace(0, Lx, Nx)
Y = np.linspace(0, Ly, Ny)

u[0, :] = F_bottom(X)
u[-1, :] = F_top(X)
u[:, 0] = F_left(Y)
u[:, -1] = F_right(Y)
#mpx, mpy = int(Nx/2), int(Ny/2) #midpoint
#u[mpy-5:mpy+5, mpx-5:mpx+5] = 100


def construct_laplacian_matrix_interiod(): # implicit euler
    A = np.zeros([Nx*Ny, Nx*Ny])

    rx = D * dt / dx**2
    ry = D * dt / dy**2
    A_const = 1 + 2 * (rx + ry)
    
    for j in range(1, Ny-1):
        for i in range(1, Nx-1):
            n = j*Nx + i
            A[n][n] = A_const #U(i,j)
            A[n][n-Nx] = -ry #U(i,j-1)
            A[n][n-1] = -rx #U(i-1,j)
            A[n][n+1] = -rx #U(i+1,j)
            A[n][n+Nx] = -ry #U(i,j+1)

    return A

def construct_laplacian_matrix(): # using global variables
    A = np.zeros([Nx*Ny, Nx*Ny])

    rx = D * dt / dx**2
    ry = D * dt / dy**2
    A_const = 1 + 2 * (rx + ry)
    
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
                A[n][n-Nx] = -ry #U(i,j-1)
                A[n][n-1] = -rx #U(i-1,j)
                A[n][n+1] = -rx #U(i+1,j)
                A[n][n+Nx] = -ry #U(i,j+1)

            ## wraparound?
    return A

def construct_laplacian_matrix_homo_neumann(): # using global variables
    A = np.zeros([Nx*Ny, Nx*Ny])

    rx = D * dt / dx**2
    ry = D * dt / dy**2
    A_const = 1 + 2 * (rx + ry)
    
    for j in range(0, Ny):
        for i in range(0, Nx):
            n = j*Nx + i

            if j == 0 and i == 0: #bottom-left corner
                A[n][n] = u[j,i]

            elif j == 0 and i == Nx-1: #bottom-right corner
                A[n][n] = u[j,i]

            elif j == Ny-1 and i == 0: #top-left corner
                A[n][n] = u[j,i]

            elif j == Ny-1 and i == Nx-1: #top-right corner
                A[n][n] = u[j,i]

            elif j == 0: #bottom
                A[n][n] = u[j,i]

            elif j == Ny-1: #top
                A[n][n] = u[j,i]

            elif i == 0: #left
                A[n][n] = u[j,i]

            elif i == Nx-1: #right
                A[n][n] = u[j,i]

            else: #interior points
                A[n][n] = A_const #U(i,j)
                A[n][n-Nx] = -ry #U(i,j-1)
                A[n][n-1] = -rx #U(i-1,j)
                A[n][n+1] = -rx #U(i+1,j)
                A[n][n+Nx] = -ry #U(i,j+1)

            ## wraparound?
    return A


def boundary_vector_dirichlet():
    b = np.zeros(Nx*Ny)
    n_idx = lambda i, j: j*Nx + i

    # bottom
    i = np.arange(0,Nx)
    j = 0
    n = n_idx(i, j)
    b[n] = u[j,i]

    # top
    i = np.arange(0,Nx)
    j = Ny-1
    n = n_idx(i, j)
    b[n] = u[j,i]

    # left
    i = 0
    j = np.arange(0,Ny)
    n = n_idx(i, j)
    b[n] = u[j,i]

    # right
    i = Nx-1
    j = np.arange(0,Ny)
    n = n_idx(i, j)
    b[n] = u[j,i]

    return b
    
def boundary_vector_homogenous_neumann():
    b = np.zeros(Nx*Ny)
    n_idx = lambda i, j: j*Nx + i

    # bottom
    i = np.arange(0,Nx)
    j = 0
    n = n_idx(i, j)
    b[n] = u[j+1,i]

    # top
    i = np.arange(0,Nx)
    j = Ny-1
    n = n_idx(i, j)
    b[n] = u[j-1,i]

    # left
    i = 0
    j = np.arange(0,Ny)
    n = n_idx(i, j)
    b[n] = u[j,i+1]

    # right
    i = Nx-1
    j = np.arange(0,Ny)
    n = n_idx(i, j)
    b[n] = u[j,i-1]

    return b

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
#A_mat = construct_laplacian_matrix()
#A_inv = np.linalg.inv(A_mat)
#b_vec = boundary_vector_dirichlet()
#U_vec = grid_to_vec(u)

print(f"\nRunning for {max_iters} iterations")
count = 0

while count < max_iters+1:
    A_mat = construct_laplacian_matrix_homo_neumann()
    A_inv = np.linalg.inv(A_mat)
    #b_vec = boundary_vector_dirichlet()
    U_vec = grid_to_vec(u)

    #b_vec = boundary_vector_homogenous_neumann() # comment out for different BCs
    
    U_vec = (A_inv @ U_vec)
    new_u = vec_to_grid(U_vec) #+ b_vec # b(n+1). add dt * F inside the brackets for source term
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