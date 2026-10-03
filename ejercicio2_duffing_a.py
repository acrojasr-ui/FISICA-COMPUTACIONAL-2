"""
Cuarta actividad - Mapas Estroboscopicos II  |  Ejercicio 2
Oscilador de Duffing forzado:  x'' + d x' - a x + b x^3 = g cos(om t)
Parametros: d=0.02, a=-1, b=5, g=8, om=0.5   (compara con la Fig. 1-(a))

Estructura: parametros, condiciones iniciales, parametros del metodo,
dinamica, RK4, integracion, puntos estroboscopicos y grafica.
"""
import numpy as np
from numpy import cos, pi
import matplotlib.pyplot as plt

# -----------------------------------------------------------
# Parametros del sistema
# -----------------------------------------------------------
d = 0.02      # delta (amortiguamiento)
a = -1.0      # alpha
b = 5.0       # beta
g = 8.0       # gamma (amplitud de la fuerza)
om = 0.5      # omega
T = 2*pi / om # periodo de la fuerza

# -----------------------------------------------------------
# Condiciones iniciales: N orbitas a la vez (cada una de un color)
# -----------------------------------------------------------
N = 400
rng = np.random.default_rng(1)
x0 = rng.uniform(1.0, 1.8, N)
v0 = rng.uniform(-2.5, 2.5, N)

# -----------------------------------------------------------
# Parametros del metodo
# -----------------------------------------------------------
Trans = 100          # periodos que se descartan (regimen transitorio)
Nperiods = 300      # periodos que se grafican por orbita
pasos = 300          # pasos de RK4 por periodo
h = T / pasos        # paso temporal: t = jT cae exactamente en un paso

# -----------------------------------------------------------
# Dinamica: Duffing forzado
# -----------------------------------------------------------
def dyn(t, y):
    x, v = y
    dx = v
    dv = g*cos(om*t) - d*v + a*x - b*x**3
    return np.array([dx, dv])

# -----------------------------------------------------------
# Runge-Kutta de orden 4
# -----------------------------------------------------------
def rk4(f, t, y, h):
    k1 = h * f(t, y)
    k2 = h * f(t + h/2, y + k1/2)
    k3 = h * f(t + h/2, y + k2/2)
    k4 = h * f(t + h, y + k3)
    return y + (k1 + 2*k2 + 2*k3 + k4) / 6

# -----------------------------------------------------------
# Integracion: se avanza periodo a periodo y solo se guardan
# los puntos estroboscopicos t = jT (j > Trans)
# -----------------------------------------------------------
y = np.vstack([x0, v0])          # forma (2, N)
t = 0.0
PE = []                          # puntos estroboscopicos

for j in range(Trans + Nperiods):
    for _ in range(pasos):
        y = rk4(dyn, t, y, h)
        t += h
    if j >= Trans:
        PE.append(y.copy())

PE = np.array(PE)                # forma (Nperiods, 2, N)

print(f"{PE.shape[0]*PE.shape[2]} puntos | x en [{PE[:,0].min():.2f}, {PE[:,0].max():.2f}] | "
      f"xdot en [{PE[:,1].min():.2f}, {PE[:,1].max():.2f}]")

# -----------------------------------------------------------
# Mapa estroboscopico
# -----------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 6))
colores = plt.cm.gist_rainbow(np.linspace(0, 1, N))
X = PE[:, 0, :].ravel()
V = PE[:, 1, :].ravel()
C = np.tile(colores, (Nperiods, 1))      # un color por orbita
orden = rng.permutation(X.size)          # mezcla el orden para que no se tapen
ax.scatter(X[orden], V[orden], s=0.8, c=C[orden])

ax.set_xlabel(r'$x$', fontsize=20)
ax.set_ylabel(r'$\dot{x}$', fontsize=20)
ax.tick_params(axis='both', labelsize=14)
ax.text(0.97, 0.97, r'$\delta = 0.02$', transform=ax.transAxes,
        ha='right', va='top', fontsize=14)
plt.tight_layout()
plt.savefig("ejercicio2_mapa.pdf", format="pdf", bbox_inches="tight")
plt.savefig("ejercicio2_mapa.png", dpi=250, bbox_inches="tight")
plt.show()
