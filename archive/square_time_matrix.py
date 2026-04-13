import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import time

lenX = lenY = L = 20 # 50
N = 60

D = 1 # 110
u0 = 30
Tmax = 20

#dx = 1/(lenX-1)
#dy = 1/(lenY-1)
#dt = min(dx**2 / (4 * D), dy**2 / (4 * D)) ## from video

h = L/(N-1)
dt = (h**2 / (4*D))
u = np.zeros((N, N)) + u0
max_iters = int(round(Tmax/dt))

## boundary conditions

u[0, :] = 0 #np.linspace(-100, 100, N)
u[-1, :] = 100 #np.linspace(-100, 100, N)
u[:, 0] = 0 #np.linspace(0, -100, N)
u[:, -1] = 30 #np.linspace(0, -100, N)

def right_boundary(): # dirchlet
    return 1

def left_boundary(): # dirchlet
    return 1

def top_boundary(): # dirchlet
    return 1

def bottom_boundary(): # dirchlet
    return 1

def CalculMat2D_time(N):
    A = np.zeros([N**2, N**2])
    r = D*dt/h**2
    
    for n in range(N**2):
        i = n % N 
        j = n // N

        if i == 0: #left boundary
            A[n][n] = left_boundary()

        elif i == N-1: #right boundary
            A[n][n] = right_boundary()

        elif j == 0: #bottom
            A[n][n] = top_boundary()

        elif j == N-1: #top
            A[n][n] = bottom_boundary()

        else:
            A[n][n] = 1 - 4*r

            if n > N-1:
                A[n][n-N] = r # u(n-N)

            if n <= (N**2)-N-1: 
                A[n][n+N] = r # u(n+N)

            # for u(n-1) and u(n+1), also prevent 'wraparound' values being added
            if n > 0 and n % N != 0: 
                A[n][n-1] = r # u(n-1)

            if n < N**2-1 and (n+1) % N != 0:
                A[n][n+1] = r # u(n+1)

    return  A

def vec_to_grid(r, N):
    u = np.zeros([N, N])
    for n in range(N**2): # convert vector indexed 1-dimensionally to 2d array
        i = n % N 
        j = n // N
        u[j][i] = r[n]
    return u

def grid_to_vec(u, N):
    vec = np.zeros(N**2)
    for n in range(N**2): # index 1-dimensionally using n
        i = n % N 
        j = n // N
        vec[n] = u[j,i]
    return vec

data = [u.copy()]
A_mat = CalculMat2D_time(N)
t1 = time.time()

steady_state_resolution = 2 # decimal places
running = True
n_iter = 0

## ADD estimate time based on max_iters, N, dt etc

while running:
    try:
        U_vec = grid_to_vec(u, N)
        B_vec = A_mat @ U_vec
        new_u = vec_to_grid(B_vec, N)
        data.append(new_u.copy())

        if np.all(np.round(new_u, steady_state_resolution) == np.round(u, steady_state_resolution)):
            print(f"Steady state reached, n={n_iter}, t={round(time.time()-t1, 2)}")
            max_iters = n_iter
            running=False

        u = new_u
        n_iter +=1 

    except KeyboardInterrupt:
        timetaken = round(time.time()-t1, 2)
        print(timetaken)
        #complete = round(n_iter/max_iters, 2)
        #left = round((timetaken/complete) * (1-complete), 2)
        #print(f"{complete*100}% done, taken {timetaken}s, {left}s estimated left")

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