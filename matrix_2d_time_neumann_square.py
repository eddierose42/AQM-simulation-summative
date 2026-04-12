import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import time

lenX = lenY = L = 10 # could do unit square
N = 50

D = 1 # 110
u0 = 0
Tmax = 1000

h = L/(N-1)
dt = 0.4 * (h**2 / (4*D))
u = np.zeros((N, N)) + u0
max_iters = int(round(Tmax/dt))

## boundary conditions

mp = int(N/2) # midpoint
#u[mp-5:mp+5, mp-5:mp+5] = 100

u[:, 0] = 100
#u[mp-5:mp+5, -1] = 40
#u[:, 0] = 100


def CalculMat2D_time_homogenous_neumann(N):
    A = np.zeros([N**2, N**2])
    r = D*dt/h**2

    for n in range(N**2):
        i = n % N 
        j = n // N
        A[n][n] = 1 - 4*r

        if i == 0: #left
            A[n][n+1] = 2*r 
        elif i == N-1: #right
            A[n][n-1] = 2*r
        else:
            A[n][n+1] = r # i+1, j
            A[n][n-1] = r # i-1, j

        if j == 0: #bottom
            A[n][n+N] = 2*r 
        elif j == N-1: #top
            A[n][n-N] = 2*r # i, j-1
        else:
            A[n][n+N] = r # i, j+1
            A[n][n-N] = r
        
    return  A

def vec_to_grid(r, N):
    u = np.zeros([N, N])
    for n in range(N**2): # convert vector indexed 1-dimensionally to 2d array
        i = n % N 
        j = n // N
        u[j,i] = r[n]
    return u

def grid_to_vec(u, N):
    vec = np.zeros(N**2)
    for n in range(N**2): # index 1-dimensionally using n
        i = n % N 
        j = n // N
        vec[n] = u[j,i]
    return vec

data = [u.copy()]
A_mat = CalculMat2D_time_homogenous_neumann(N)
t1 = time.time()

steady_state_resolution = 13 # decimal places
running = True
run_count = True
n_iter = 0

## ADD estimate time based on max_iters, N, dt etc

if run_count:
    print(f"\nRunning for max {max_iters} iterations:")


U_vec = grid_to_vec(u, N)

while running and (run_count and n_iter < max_iters+1):
    try:
        #U_vec = grid_to_vec(u, N)
        U_vec = A_mat @ U_vec

        new_u = vec_to_grid(U_vec, N)
        data.append(new_u.copy())
        
        if np.all(np.round(new_u, steady_state_resolution) == np.round(u, steady_state_resolution)):
            print(f"\nSteady state reached, n={n_iter}, t={round(time.time()-t1, 2)}")
            max_iters = n_iter
            running=False
     
        u = new_u
        n_iter += 1 

    except KeyboardInterrupt:
        timetaken = round(time.time()-t1, 2)
        print(timetaken)
        if run_count:
            complete = n_iter/max_iters
            left = round((timetaken/complete) * (1-complete), 2)
            print(f"{round(complete, 2)*100}% done, taken {timetaken}s, {left}s estimated left")


delta = data[-1].sum() - data[0].sum()
print(f"\nLeakage: {delta} ({round((delta / data[0].sum())*100, 2)}%)\n")

## Run simulation

fig, axis = plt.subplots()
pcm = axis.pcolormesh(data[0], cmap=plt.cm.jet, vmin=np.array(data).min(), vmax=np.array(data).max())#vmin=-100, vmax=100)
plt.colorbar(pcm, ax=axis)

sliderax = fig.add_axes([0, 0, 0.65, 0.03])
slider = Slider(ax=sliderax, label="T", valmin=0, valmax=max_iters+1, valinit=0) #max_iters+1 = len(data)

buttonax = fig.add_axes([0.85, 0, 1, 0.015])
button = Button(ax=buttonax, label="Play")
animation_running = False
animation_speed = dt

def updateplot(val):
    pcm.set_array(data[int(val)])
    axis.set_title("Distribution at t: {:.3f} [s].".format(int(val)*dt))
    #fig.canvas.draw_idle()

def animate(event=None):
    global animation_running
    animation_running = not animation_running

    while animation_running and slider.val < max_iters-1:
        slider.set_val(slider.val+1)
        plt.pause(animation_speed)
    
    if animation_running:
        animation_running = False

def key_press(event):
    if event.key == " ":
        animate()

slider.on_changed(updateplot)
button.on_clicked(animate)
fig.canvas.mpl_connect('key_press_event', key_press)
plt.show()