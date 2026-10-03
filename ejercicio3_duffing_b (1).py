"""
Cuarta actividad - Mapas Estroboscopicos II  |  Ejercicio 3
Oscilador de Duffing forzado:  x'' + d x' - a x + b x^3 = g cos(om t)
Parametros: d=0.15, b=1, g=0.3, om=1   (compara con la Fig. 1-(b))

Se resuelve con a = -1 (valor del enunciado) y, para comparar con la
Fig. 1-(b), tambien con a = +1 (pozo doble). Ver informe.

Estructura: parametros, condiciones iniciales, parametros del metodo,
dinamica, RK4, integracion, puntos estroboscopicos y grafica.
"""
import numpy as np
from numpy import cos, pi
import matplotlib.pyplot as plt

# -----------------------------------------------------------
# Parametros del sistema
# -----------------------------------------------------------
d = 0.15      # delta (amortiguamiento)
b = 1.0       # beta
g = 0.3       # gamma (amplitud de la fuerza)
om = 1.0      # omega
T = 2*pi / om # periodo de la fuerza
alphas = [-1.0, 1.0]   # alpha del enunciado (-1) y alpha de pozo doble (+1)

# -----------------------------------------------------------
# Condiciones iniciales: N orbitas a la vez (cada una de un color)
# -----------------------------------------------------------
N = 100
rng = np.random.default_rng(1)
x0 = rng.uniform(-1.5, 1.5, N)
v0 = rng.uniform(-0.6, 1.1, N)

# -----------------------------------------------------------
# Parametros del metodo
# -----------------------------------------------------------
Trans = 100          # periodos que se descartan (regimen transitorio)
Nperiods = 500       # periodos que se grafican por orbita
pasos = 200          # pasos de RK4 por periodo
h = T / pasos        # paso temporal: t = jT cae exactamente en un paso

# -----------------------------------------------------------
# Dinamica: Duffing forzado (usa la variable global a)
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
# Integracion: para cada alpha se avanza periodo a periodo y solo
# se guardan los puntos estroboscopicos t = jT (j > Trans)
# -----------------------------------------------------------
resultados = {}
for a in alphas:
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
    resultados[a] = PE
    print(f"alpha = {a:+.0f}: x en [{PE[:,0].min():.4f}, {PE[:,0].max():.4f}] | "
          f"xdot en [{PE[:,1].min():.4f}, {PE[:,1].max():.4f}]")

# -----------------------------------------------------------
# Mapa estroboscopico: comparacion alpha = -1 vs alpha = +1
# -----------------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(11, 5))
colores = plt.cm.gist_rainbow(np.linspace(0, 1, N))

for ax, a in zip(axs, alphas):
    PE = resultados[a]
    X = PE[:, 0, :].ravel()
    V = PE[:, 1, :].ravel()
    C = np.tile(colores, (Nperiods, 1))      # un color por orbita
    orden = rng.permutation(X.size)          # mezcla el orden para que no se tapen
    ax.scatter(X[orden], V[orden], s=0.8 if a > 0 else 25, c=C[orden])
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-0.8, 1.4)
    ax.set_xlabel(r'$x$', fontsize=18)
    ax.set_ylabel(r'$\dot{x}$', fontsize=18)
    ax.tick_params(axis='both', labelsize=12)
    ax.set_title(rf'$\alpha = {a:+.0f}$', fontsize=16)
    ax.text(0.97, 0.97, r'$\delta = 0.15$', transform=ax.transAxes,
            ha='right', va='top', fontsize=13)

plt.tight_layout()
plt.savefig("ejercicio3_comparacion.pdf", format="pdf", bbox_inches="tight")
plt.savefig("ejercicio3_comparacion.png", dpi=250, bbox_inches="tight")
plt.show()
