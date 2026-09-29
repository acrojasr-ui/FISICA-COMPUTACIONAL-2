"""
Tercera actividad - Mapas Estroboscopicos
Fisica Computacional Aplicada a la Dinamica No Lineal - Semillero MECA

Pendulo simple forzado sin amortiguamiento:
    theta'' + (g/l) sin(theta) = (F0/(m l)) cos(omega t)

Mapa estroboscopico: se registra (theta, theta') cada periodo T = 2*pi/omega.
Se integra con RK4, vectorizado sobre muchas condiciones iniciales a la vez.
"""
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------
# 1. Parametros del sistema
# ---------------------------------------------------------------
m_l = 1.0            # m*l
g_l = 1.0            # g/l
F0 = 0.01            # amplitud de la fuerza
omega = 2.0 / np.pi  # frecuencia de forzamiento
T = 2.0 * np.pi / omega   # periodo de forzamiento (= pi^2)

# ---------------------------------------------------------------
# 2. Condiciones iniciales (malla de orbitas)
# ---------------------------------------------------------------
n_theta, n_omega = 25, 14
theta0 = np.linspace(-np.pi, np.pi, n_theta)
dtheta0 = np.linspace(-2.4, 2.4, n_omega)
TH0, DTH0 = np.meshgrid(theta0, dtheta0)
# estado: arreglo (2, N)  ->  fila 0: theta, fila 1: theta'
y0 = np.vstack([TH0.ravel(), DTH0.ravel()])
N = y0.shape[1]

# Condiciones extra cerca del punto fijo y de la separatriz (region caotica)
extra = np.array([[0.0, 0.0], [0.3, 0.0], [0.6, 0.0], [0.9, 0.0],
                  [2.9, 0.0], [-2.9, 0.0], [3.0, 0.3], [-3.0, -0.3],
                  [3.1, 0.0], [-3.1, 0.0]]).T
y0 = np.hstack([y0, extra])
N = y0.shape[1]

# ---------------------------------------------------------------
# 3. Parametros del metodo numerico
# ---------------------------------------------------------------
pasos_por_periodo = 200          # dt = T / 200 (muestreo exacto en t = nT)
dt = T / pasos_por_periodo
n_periodos = 1500                # puntos por orbita en el mapa

# ---------------------------------------------------------------
# 4. Funcion de dinamica
# ---------------------------------------------------------------
def dinamica(t, y):
    """y = [theta, theta']  ->  dy/dt = [theta', theta'']"""
    theta, dtheta = y
    ddtheta = -g_l * np.sin(theta) + (F0 / m_l) * np.cos(omega * t)
    return np.array([dtheta, ddtheta])

# ---------------------------------------------------------------
# 5. Metodo Runge-Kutta de orden 4 (un paso)
# ---------------------------------------------------------------
def rk4(t, y, dt):
    k1 = dinamica(t, y)
    k2 = dinamica(t + dt / 2, y + dt * k1 / 2)
    k3 = dinamica(t + dt / 2, y + dt * k2 / 2)
    k4 = dinamica(t + dt, y + dt * k3)
    return y + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

# ---------------------------------------------------------------
# 6. Integracion y toma de puntos estroboscopicos
# ---------------------------------------------------------------
def envolver(theta):
    """Lleva theta al intervalo [-pi, pi)."""
    return (theta + np.pi) % (2 * np.pi) - np.pi

puntos = np.zeros((n_periodos + 1, 2, N))
puntos[0] = y0
y = y0.copy()
t = 0.0
for n in range(n_periodos):
    for _ in range(pasos_por_periodo):
        y = rk4(t, y, dt)
        t += dt
    puntos[n + 1] = y                     # muestreo en t = (n+1) T
    puntos[n + 1, 0] = envolver(y[0])     # theta en [-pi, pi)
    y[0] = puntos[n + 1, 0]               # (la fuerza no depende de theta: es valido)

# ---------------------------------------------------------------
# 7. Grafica del mapa estroboscopico
# ---------------------------------------------------------------
rng = np.random.default_rng(1)
colores = rng.random((N, 3))

fig, ax = plt.subplots(figsize=(7, 5))
for i in range(N):
    ax.plot(puntos[:, 0, i], puntos[:, 1, i], '.', ms=1.2, color=colores[i])
ax.set_xlim(-np.pi, np.pi)
ax.set_ylim(-2.5, 2.5)
ax.set_xlabel(r'$\theta$')
ax.set_ylabel(r'$\dot{\theta}$')
ax.set_title('Mapa estroboscópico del péndulo simple forzado')
plt.tight_layout()
plt.savefig('mapa_estroboscopico.png', dpi=200)
plt.show()
