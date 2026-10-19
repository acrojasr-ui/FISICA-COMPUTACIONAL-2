import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Parámetros
# ---------------------------------------------------------
E = 1
mu = 2.8
r_max = E / (mu - 1)

# ---------------------------------------------------------
# Cálculo de las funciones
# ---------------------------------------------------------
r = np.linspace(0, r_max, 400)
dr = np.sqrt((2*E - 2*r*(mu - 1)) / (mu + 1))

# ---------------------------------------------------------
# Figura
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5), dpi=180)

# Área sombreada y curvas superior e inferior
ax.fill_between(r, -dr, dr, color='lightblue', alpha=0.5)
ax.plot(r, dr, color='blue', linewidth=2)
ax.plot(r, -dr, color='blue', linewidth=2)

# Puntos iniciales
ax.scatter([0, 0], [dr[0], -dr[0]], color='blue', s=40)

# ---------------------------------------------------------
# Formato
# ---------------------------------------------------------
# Etiquetas y título
ax.set_xlabel('r', fontsize=12)
ax.set_ylabel(r'$\dot{r}$', fontsize=12)
ax.set_title('Diagrama de fases', fontsize=14)

# Ejes coordenados y cuadrícula
ax.axhline(0, color='black', linewidth=0.8)
ax.axvline(0, color='black', linewidth=0.8)
ax.grid(True, linestyle='--', alpha=0.4)

# Límites de los ejes
ax.set_xlim(0, r_max * 1.1)
ax.set_ylim(-max(dr) * 1.2, max(dr) * 1.2)

# ---------------------------------------------------------
# Mostrar
# ---------------------------------------------------------
plt.tight_layout()
plt.show()