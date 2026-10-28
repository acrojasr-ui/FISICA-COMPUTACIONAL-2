import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# -----------------------------------
# initial point
#--------------------------------------------
p = np.array([0.5, 0])
#-------------------------------------
# iterations and arrays
#-------------------------------------------------
N = 200000
X = np.zeros(N)
Y = np.zeros(N)
#-------------------------------------------------
# Barnsley fern algorithm
#-------------------------------------------
for n in range(N):
  r = np.random.random()
  xn = p[0]
  yn = p[1]
  if r < 0.10: # 10%  -> T1
    p = np.array([0.05*xn, 0.6*yn])
  elif r < 0.20:  # 10% -> T2
    p = np.array([0.05*xn, -0.5*yn + 1])
  elif r < 0.40:  # 20% -> T3
    p = np.array([0.46*xn - 0.15*yn, 0.39*xn + 0.38*yn + 0.6])
  elif r < 0.60:  # 20% -> T4
    p = np.array([0.47*xn - 0.15*yn, 0.17*xn + 0.42*yn + 1.1])
  elif r < 0.80:  # 20% -> T5
    p = np.array([0.43*xn + 0.28*yn, -0.25*xn + 0.45*yn + 1])
  else:  # 20% -> T6
    p = np.array([0.42*xn + 0.26*yn, -0.35*xn + 0.31*yn + 0.7])
  X[n] = p[0]
  Y[n] = p[1]

#-------------------------------------------------
# Colores: tronco (y < y_tronco) y copa
#-------------------------------------------------
y_tronco = 0.6                       # altura donde empieza la copa
color_copa   = 'deeppink'
color_tronco = 'saddlebrown'
tronco = Y < y_tronco

#------------------------------------------
# Scatter plot
#-----------------------------------------------
fig, ax = plt.subplots(figsize=(6,9))
ax.scatter(X[~tronco], Y[~tronco], s=0.1, color=color_copa)
ax.scatter(X[tronco],  Y[tronco],  s=0.1, color=color_tronco)
ax.text(0.03,0.97,rf'$N = {N}$', transform=ax.transAxes,ha='left', va='top', fontsize=12)
ax.axis('off')
ax.set_aspect('equal')
plt.tight_layout()
#-----------------------------------------------------------------
# figure format
#------------------------------------------------------
plt.savefig("ferm.pdf", format="pdf", bbox_inches="tight", dpi=800)
plt.show()

#=================================================
# BOX COUNTING
#=================================================
# Cuadrado que contiene al fractal: lado L = mayor extensión, fractal centrado
xmin, ymin = X.min(), Y.min()
L = max(X.max() - xmin, Y.max() - ymin)
xc = 0.5*(X.max() + xmin)
yc = 0.5*(Y.max() + ymin)
x0 = xc - L/2
y0 = yc - L/2

k_values = np.arange(2, 17)          # epsilon = 1/2^k, n = 2^k cajas por lado
k_plot   = np.arange(2, 10)          # k para los cuales se guarda la imagen
n_boxes  = []
N_eps    = []

def box_indices(n):
  """Índices (ix, iy) de la caja que ocupa cada punto."""
  ix = np.clip(((X - x0)/L*n).astype(np.int64), 0, n-1)
  iy = np.clip(((Y - y0)/L*n).astype(np.int64), 0, n-1)
  return ix, iy

def count_boxes(n):
  """Número de cajas ocupadas N(eps) (sin construir la matriz completa)."""
  ix, iy = box_indices(n)
  return np.unique(ix*n + iy).size

def occupied_grid(n):
  """Matriz booleana (n x n) de cajas ocupadas, solo para graficar."""
  ix, iy = box_indices(n)
  grid = np.zeros((n, n), dtype=bool)
  grid[iy, ix] = True
  return grid

for k in k_values:
  n = 2**k
  n_boxes.append(n)
  N_eps.append(count_boxes(n))
  Df_k = np.log(N_eps[-1])/np.log(n)    # ln N(eps) / ln(1/eps)

  if k in k_plot:
    grid = occupied_grid(n)
    fig, ax = plt.subplots(figsize=(6,6))
    # cajas ocupadas en azul claro
    cmap = ListedColormap([(0,0,0,0), (0.76, 0.87, 0.93, 1)])
    ax.imshow(grid, origin='lower', extent=[x0, x0+L, y0, y0+L], cmap=cmap,
              interpolation='nearest', zorder=0)
    # cuadrícula
    lw = 0.6 if n <= 64 else 0.25
    edges = np.linspace(0, L, n+1)
    for e in edges:
      ax.axvline(x0 + e, color='gray', lw=lw, alpha=0.5, zorder=1)
      ax.axhline(y0 + e, color='gray', lw=lw, alpha=0.5, zorder=1)
    # fractal (mismos colores que la figura principal)
    ax.scatter(X[~tronco], Y[~tronco], s=0.1, color=color_copa,
               linewidths=0, rasterized=True, zorder=2)
    ax.scatter(X[tronco], Y[tronco], s=0.1, color=color_tronco,
               linewidths=0, rasterized=True, zorder=2)
    ax.text(0.03, 0.97,
            rf'$N = {N}$' + '\n' + rf'$n = {n},\ N(\epsilon) = {N_eps[-1]}$' + '\n'
            + rf'$D_f = {Df_k:.6f}$',
            transform=ax.transAxes, ha='left', va='top', fontsize=10,
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.8), zorder=3)
    ax.set_xlim(x0, x0+L)
    ax.set_ylim(y0, y0+L)
    ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(rf'Box counting: $\epsilon = 1/2^{{{k}}} = 1/{n}$')
    plt.tight_layout()
    plt.savefig(f"box_counting_k{k}.png", dpi=200, bbox_inches="tight")
    plt.show()

#-------------------------------------------------
# Regresión lineal  ln N(eps) vs ln(1/eps)
#-------------------------------------------------
n_boxes = np.array(n_boxes)
N_eps   = np.array(N_eps)
ln_inv_eps = np.log(n_boxes)          # ln(1/eps)
ln_N_eps   = np.log(N_eps)

# Región lineal: se descartan las escalas donde N(eps) se acerca a N
mask = N_eps <= 0.05*N
slope, intercept = np.polyfit(ln_inv_eps[mask], ln_N_eps[mask], 1)

fig, ax = plt.subplots(figsize=(6,4.5))
ax.plot(ln_inv_eps, ln_N_eps, 'o')
ax.plot(ln_inv_eps[mask], slope*ln_inv_eps[mask] + intercept, '-',
        label=rf'$D_f = {slope:.4f}$')
ax.set_xlabel(r'$\ln(1/\epsilon)$')
ax.set_ylabel(r'$\ln N(\epsilon)$')
ax.legend()
plt.tight_layout()
plt.savefig("box_counting_regresion.png", dpi=200, bbox_inches="tight")
plt.show()