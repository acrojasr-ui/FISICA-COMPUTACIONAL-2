import numpy as np
from numpy import sin, cos, pi
import matplotlib.pyplot as plt
import os
# -----------------------------------------------------
# System parameters
# -----------------------------------------------------
delta = 0.1
alpha = 2
beta = 2
om = 1.2
T = 2*pi / om
# -----------------------------------------------------
# Initial conditions
# -----------------------------------------------------
x0 = 1.0
v0 = 1.0
# -----------------------------------------------------
# Gamma values
# -----------------------------------------------------
gamma_min = 0.1
gamma_max = 7
dgamma = 0.001
gamma_values = np.arange(gamma_min, gamma_max + dgamma, dgamma)
n_orbits = len(gamma_values)
# -----------------------------------------------------
# Numerical method parameters
# -----------------------------------------------------
Trans = 250
Nkeep = 300
steps_per_T = 300
dt = T / steps_per_T

# -----------------------------------------------------
# Initial state
# -----------------------------------------------------
x = np.full(n_orbits, x0)
v = np.full(n_orbits, v0)
y = np.concatenate([x, v])

# -----------------------------------------------------
# Dynamics
# -----------------------------------------------------
def dyn(t, y):
    x = y[:n_orbits]
    v = y[n_orbits:]
    dx = v
    dv = -delta*v + alpha*x - beta*x**3 + gamma_values*cos(om*t)
    return np.concatenate([dx, dv])

# -----------------------------------------------------
# Fourth-order Runge-Kutta method
# -----------------------------------------------------
def rk4(f, t, y, h):
    k1 = h*f(t, y)
    k2 = h*f(t + h/2, y + k1/2)
    k3 = h*f(t + h/2, y + k2/2)
    k4 = h*f(t + h, y + k3)
    return y + (k1 + 2*k2 + 2*k3 + k4)/6

# -----------------------------------------------------
# Storage for stroboscopic points
# -----------------------------------------------------
x_strobe = np.empty((Nkeep, n_orbits))
v_strobe = np.empty((Nkeep, n_orbits))
save_index = 0

# -----------------------------------------------------
# Integration
# -----------------------------------------------------
total_periods = Trans + Nkeep
total_steps = total_periods * steps_per_T
for step in range(total_steps):
    current_time = step*dt
    y = rk4(dyn, current_time, y, dt)
    completed_period = (step + 1) // steps_per_T
    if (step + 1) % steps_per_T == 0:
        if completed_period > Trans:
            x_strobe[save_index] = y[:n_orbits]
            v_strobe[save_index] = y[n_orbits:]
            save_index += 1

# -----------------------------------------------------
# Bifurcation diagram of x & Figure format
# -----------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 6))
for i in range(n_orbits):
    ax.scatter(np.full(Nkeep, gamma_values[i]),
               x_strobe[:, i], s=0.1, color='blue', linewidths=0, rasterized=True)
ax.set_xlabel(r'$\gamma$', fontsize=16)
ax.set_ylabel(r'$x$', fontsize=16)
ax.tick_params(axis='both', labelsize=12)
ax.text(0.03, 0.97, rf'$\delta={delta},\ \alpha={alpha},\ \beta={beta},\ \omega={om}$',
        transform=ax.transAxes, ha='left', va='top', fontsize=12)
ax.set_xlim(gamma_min, gamma_max)
ax.set_box_aspect(0.65)
plt.tight_layout()
plt.savefig('Bifurcation_x.pdf', format='pdf', bbox_inches='tight', dpi=800)
plt.show()

# -----------------------------------------------------
# Bifurcation diagram of x_dot & Figure format
# -----------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 6))
for i in range(n_orbits):
    ax.scatter(np.full(Nkeep, gamma_values[i]),
               v_strobe[:, i], s=0.1, color='blue', linewidths=0, rasterized=True)
ax.set_xlabel(r'$\gamma$', fontsize=16)
ax.set_ylabel(r'$\dot{x}$', fontsize=16)
ax.tick_params(axis='both', labelsize=12)
ax.text(0.03, 0.97, rf'$\delta={delta},\ \alpha={alpha},\ \beta={beta},\ \omega={om}$',
        transform=ax.transAxes, ha='left', va='top', fontsize=12)
ax.set_xlim(gamma_min, gamma_max)
ax.set_box_aspect(0.65)
plt.tight_layout()
plt.savefig('Bifurcation_v.pdf', format='pdf', bbox_inches='tight', dpi=800)
plt.show()

# -----------------------------------------------------
# Comparison with Fig. 1(a) and Fig. 1(b) of the activity
# Fig1a_ref.png and Fig1b_ref.png: images of the reference panels
# (they must be in the same folder as this script)
# -----------------------------------------------------
gamma_grid = np.tile(gamma_values, (Nkeep, 1))
comparisons = [('Fig1a_ref.png', x_strobe, r'$x$', 'Comparison_x.pdf'),
               ('Fig1b_ref.png', v_strobe, r'$\dot{x}$', 'Comparison_v.pdf')]
for ref_file, data, ylabel, out_name in comparisons:
    if os.path.exists(ref_file):
        fig, axs = plt.subplots(1, 2, figsize=(14, 5))
        axs[0].imshow(plt.imread(ref_file))
        axs[0].axis('off')
        axs[0].set_title('Fig. 1 (reference)', fontsize=14)
        axs[1].scatter(gamma_grid.ravel(), data.ravel(),
                       s=0.1, color='blue', linewidths=0, rasterized=True)
        axs[1].set_title('Python (RK4)', fontsize=14)
        axs[1].set_xlabel(r'$\gamma$', fontsize=16)
        axs[1].set_ylabel(ylabel, fontsize=16)
        axs[1].set_xlim(0, 7)
        axs[1].tick_params(axis='both', labelsize=12)
        plt.tight_layout()
        plt.savefig(out_name, format='pdf', bbox_inches='tight', dpi=800)
        plt.show()