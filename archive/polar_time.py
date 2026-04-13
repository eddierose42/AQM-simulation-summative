import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

Nr = 20
Ntheta = 36
N = Nr * Ntheta
Tmax = 10000

i = j = 0
dr = 1/(Nr-1)
dtheta = 2*np.pi/Ntheta #### ??? 2*np.pi
D = 1
dt = 0.1 * min(dr**2, (dr*dtheta)**2) / D #min(dr**2 / (4 * D), dtheta**2 / (4 * D)) ## FROM VIDEO
u = np.zeros([Nr, Ntheta]) + 40

## BOUNDARY CONDITIONS

## USING ARRAY
u[0, :] = 100 ## TOP ROW IS CENTRE POINT
#u[Nr-1, :] = 100
u[Nr-1, :] = 0 ## TOP ROW IS CENTRE POINT

## ADD FUNCTION TRANSLATING BOUNDARY CONDITIONS FROM POLAR COORDS TO ARRAY

data = [u.copy()]

for iteration in range(Tmax):
    u_arr = u.copy()
    
    ##

    for i in range(1,Nr-1):
        for j in range(1,Ntheta-1):
            dr_term = (1/(i*dr*dr)) * (((i*dr + dr/2) * ((u_arr[i+1,j] - u_arr[i,j])/dr)) - ((i*dr - dr/2) * ((u_arr[i,j] - u_arr[i-1,j])/dr)))
            dtheta_term = ((u_arr[i,j+1]) - 2*u_arr[i,j] + u_arr[i,j-1]) / ((i*dr)**2 * (dtheta**2))
            u[i,j] = u_arr[i,j] + (D * dt * (dr_term + dtheta_term))
        
        #wrap around
        #j=0
        dr_term = (1/(i*dr*dr)) * (((i*dr + dr/2) * ((u_arr[i+1,0] - u_arr[i,0])/dr)) - ((i*dr - dr/2) * ((u_arr[i,0] - u_arr[i-1,0])/dr)))
        u[i,0] = u_arr[i,0] + (D * dt * (dr_term + ((u_arr[i,1]) - 2*u_arr[i,0] + u_arr[i,Ntheta-1]) / ((i*dr)**2 * (dtheta**2))))
        #j=Ntheta-1
        dr_term = (1/(i*dr*dr)) * (((i*dr + dr/2) * ((u_arr[i+1,Ntheta-1] - u_arr[i,Ntheta-1])/dr)) - ((i*dr - dr/2) * ((u_arr[i,Ntheta-1] - u_arr[i-1,Ntheta-1])/dr)))
        u[i,Ntheta-1] = u_arr[i,Ntheta-1] + (D * dt * (dr_term + ((u_arr[i,0]) - 2*u_arr[i,Ntheta-1] + u_arr[i,Ntheta-2]) / ((i*dr)**2 * (dtheta**2))))

    data.append(u.copy()) 


print("calculations done")


fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)

theta = np.arange(Ntheta) * dtheta
r = np.arange(Nr) * dr
thetas, rs = np.meshgrid(theta, r)
pcm = ax.pcolormesh(thetas, rs, data[0], shading="auto", vmin=0)
plt.colorbar(pcm, ax=ax)

sliderax = fig.add_axes([0.25, 0.1, 0.65, 0.03])
slider = Slider(ax=sliderax, label="t", valmin=0, valmax=Tmax, valinit=0) #0

def updateplot(var):
    idx = int((slider.val/Tmax) * (len(data)-1))
    pcm.set_array(data[idx].ravel())
    ax.set_title("Distribution at t: {:.3f} [s].".format(slider.val))

slider.on_changed(updateplot)
plt.show()