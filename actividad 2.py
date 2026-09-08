import numpy as np
from numpy import sin, cos
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# -----------------------------------------------------
# System parameters
# -----------------------------------------------------
g = 9.81
L1 = 1
L2 = 1
m1 = 1
m2 = 1

# Forzamiento del pivote: xp(t) = A*cos(w*t)
A = 1       # amplitud
w = 1       # frecuencia angular del pivote

# -----------------------------------------------------
# Initial conditions
# -----------------------------------------------------
th10 = 0.0
th20 = np.pi / 2
ome10, ome20 = 0.0, 0.0

# -----------------------------------------------------
# method parameters
# -----------------------------------------------------
tmax = 40
dt = 0.01
STRIDE = 2

# -----------------------------------------------------
# Dynamics: Double pendulum with horizontally oscillating pivot
# xp(t) = A*cos(w*t)  ->  aporta un término de forzamiento en las
# dos ecuaciones (obtenido del lagrangiano, ver eq1/eq2 en Mathematica)
# -----------------------------------------------------
def dyn(t, y):
    th1, w1, th2, w2 = y
    d = th1 - th2
    sd = sin(d)
    cd = cos(d)
    den = m1 + m2 * sd**2

    forcing = A * w**2 * cos(w * t)

    a1 = (forcing * ((2*m1 + m2)*cos(th1) - m2*cos(th1 - 2*th2))
          - 2*m2*sd*(L1*cd*w1**2 + L2*w2**2)
          - g*(2*m1 + m2)*sin(th1) - g*m2*sin(th1 - 2*th2)) / (2*L1*den)

    a2 = (sd * ((m1 + m2)*(L1*w1**2 + g*cos(th1) + forcing*sin(th1))
                + L2*m2*cd*w2**2)) / (L2*den)

    return np.array([w1, a1, w2, a2])

# -----------------------------------------------------
# Fourth-order Runge-Kutta method
# -----------------------------------------------------
def rk4(f, t, y, h):
    k1 = h * f(t, y)
    k2 = h * f(t + h/2, y + k1/2)
    k3 = h * f(t + h/2, y + k2/2)
    k4 = h * f(t + h, y + k3)
    return y + (k1 + 2*k2 + 2*k3 + k4) / 6

# -----------------------------------------------------
# Integration using RK4
# -----------------------------------------------------
n = int(tmax / dt)
t = np.linspace(0, n * dt, n + 1)
y = np.empty((n + 1, 4))
y[0] = np.array([th10, ome10, th20, ome20])
for i in range(n):
    y[i + 1] = rk4(dyn, t[i], y[i], dt)

# -----------------------------------------------------
# Separate variables after integration
# -----------------------------------------------------
th1 = y[:, 0]
ome1 = y[:, 1]
th2 = y[:, 2]
ome2 = y[:, 3]

# -----------------------------------------------------
# Kinematics
# -----------------------------------------------------
xp = A * cos(w * t)                       # posición del pivote (se mueve)
x1, y1 = xp + L1 * sin(th1), -L1 * cos(th1)
x2, y2 = x1 + L2 * sin(th2), y1 - L2 * cos(th2)

# -----------------------------------------------------
# Figure
# -----------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 7))
R = L1 + L2 + A + 0.2
ax.set(xlim=(-R, R), ylim=(-R, R), aspect="equal",
       title="Péndulo doble con pivote oscilante (RK4)")
ax.title.set_fontsize(16)
ax.tick_params(axis="both", labelsize=12)
ax.grid(alpha=0.3)

rail, = ax.plot([], [], "-", lw=3, color="gray")   # riel del pivote
line, = ax.plot([], [], "o-", lw=1.3, color="black")
trace, = ax.plot([], [], "-", lw=1, alpha=0.6, color="red")
clock = ax.text(0.05, 0.93, "", transform=ax.transAxes, fontsize=12)

# -----------------------------------------------------
# Create Animation
# -----------------------------------------------------
def animate(i):
    rail.set_data([xp[i] - 0.2, xp[i] + 0.2], [0, 0])
    line.set_data([xp[i], x1[i], x2[i]], [0, y1[i], y2[i]])
    trace.set_data(x2[:i+1], y2[:i+1])
    clock.set_text(f"t = {t[i]:.1f} s")
    return rail, line, trace, clock

# -----------------------------------------------------
# Animation
# -----------------------------------------------------
ani = FuncAnimation(fig, animate, frames=range(0, n + 1, STRIDE),
                     interval=STRIDE * dt * 1000, blit=True)
plt.tight_layout()

# -----------------------------------------------------
# Salida: si estamos en Colab/Jupyter, guardar GIF con pillow
# (no requiere ffmpeg) y mostrarlo; si no, usar plt.show()
# -----------------------------------------------------
try:
    get_ipython()  # existe solo en Jupyter/Colab
    ani.save("pendulo.gif", writer="pillow", fps=15)
    from IPython.display import Image, display
    display(Image(filename="pendulo.gif"))
except NameError:
    plt.show()