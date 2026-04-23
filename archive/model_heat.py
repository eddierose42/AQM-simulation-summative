import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

fig, axis=plt.subplots()
img = np.asarray(Image.open('AQM/summative 2/AQM-simulation/wood_burner_heat.png'))
imgplot = plt.imshow(img, extent=[0,18,0,14])
axis.set_ylabel("kW")
axis.set_xlabel("HRS")

Xdata = []
Ydata = []

def onclick(event):
    print(f"x={event.xdata}, y={event.ydata}")
    Xdata.append(event.xdata)
    Ydata.append(event.ydata)
    axis.plot(event.xdata, event.ydata, 'x', color="red")
    fig.canvas.draw_idle()

def onpress(event):
    if event.key == " ":
        coefficients = np.polyfit(Xdata, Ydata, 7)
        f = np.poly1d(coefficients)
        x = np.linspace(min(Xdata),max(Xdata),100)
        der = np.polyder(f)
        axis.plot(x, f(x))
        fig.canvas.draw_idle()
        print(f"\n{f}\n{der}")

fig.canvas.mpl_connect('button_press_event', onclick)
fig.canvas.mpl_connect('key_press_event', onpress)
plt.show()