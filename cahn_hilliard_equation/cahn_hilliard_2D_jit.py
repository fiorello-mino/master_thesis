import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import time


# ============================================================
#                INTEGRAZIONE CAHN-HILLIARD
# ============================================================

@njit
def evolve_cahn_hilliard(phi_init, N_steps, dt, dx, dy, epsilon, M0, snap_steps, snap_array, mass_array, energy_array):

    N0, N1 = phi_init.shape
    phi = np.copy(phi_init)
    
    # Allocazioni
    phi_new   = np.empty_like(phi)
    lapl_phi  = np.empty_like(phi)
    w_prime   = np.empty_like(phi)
    mu        = np.empty_like(phi)
    mobility  = np.empty_like(phi)
    grad_mu_x = np.empty_like(phi)
    grad_mu_y = np.empty_like(phi)
    J_x       = np.empty_like(phi)
    J_y       = np.empty_like(phi)
    div_J     = np.empty_like(phi)
    
    snap_index = 0
    
    for n in range(N_steps):
        
        # Salva snapshots selezionati
        if snap_index < len(snap_steps) and n == snap_steps[snap_index]:
            snap_array[snap_index] = np.copy(phi)
            snap_index += 1

        # Calcolo della massa
        mass_array[n] = np.sum(phi)*dx*dy
        
        energy_array[n] = 0.0
        # -----------------------------------------------------
        #            CALCOLO DI MU
        # -----------------------------------------------------
        for i in range(N0):
            i_up = (i - 1) % N0
            i_down = (i + 1) % N0
            
            for j in range(N1):
                j_right = (j + 1) % N1
                j_left = (j - 1) % N1
                
                # laplaciano di phi
                lapl_phi[i, j] = (
                    phi[i, j_right]
                    + phi[i, j_left]
                    + phi[i_up, j]
                    + phi[i_down, j]
                    - 4 * phi[i, j] 
                ) / dx**2
                
                # calcolo mu
                w_prime[i, j] = (36.0 / epsilon) * phi[i, j] * (1 + 2 * phi[i, j]**2 - 3 * phi[i, j])
                
                mu[i, j] = - epsilon * lapl_phi[i, j] + w_prime[i, j]
                
                # calcolo energia
                grad_phi_x = (phi[i, j_right] - phi[i, j_left]) / (2.0 * dx)
                grad_phi_y = (phi[i_up, j] - phi[i_down, j])   / (2.0 * dy)
                grad2 = grad_phi_x**2 + grad_phi_y**2
                w = (18.0 / epsilon) * phi[i,j]**2 * (1 - phi[i,j])**2
                
                energy_array[n] += (0.5 * epsilon * grad2 + w) * dx * dy


        # -----------------------------------------------------
        #           STEP ESPLICITO PER PHI
        # -----------------------------------------------------
        for i in range(N0):
            i_up = (i - 1) % N0
            i_down = (i + 1) % N0
            
            for j in range(N1):
                j_right = (j + 1) % N1
                j_left = (j - 1) % N1

                # mobilità variabile
                mobility[i, j] = (36.0 / epsilon) * M0 * phi[i, j]**2 * (1 - phi[i, j])**2 + 1e-9
                
                # gradiente di mu
                grad_mu_x[i,j] = ( mu[i, j_right] - mu[i, j_left] ) / (2*dx)
                grad_mu_y[i,j] = ( mu[i_up, j] - mu[i_down, j] ) / (2*dy)
                
                # corrente J = M grad mu
                J_x[i, j] = mobility[i,j] * grad_mu_x[i, j]
                J_y[i, j] = mobility[i,j] * grad_mu_y[i, j]
                
                
        # Divergenza di J
        for i in range(N0):
            i_up = (i - 1) % N0
            i_down = (i + 1) % N0
            
            for j in range(N1):
                j_right = (j + 1) % N1
                j_left = (j - 1) % N1
                
                # divergenza di J
                div_x = ( J_x[i, j_right] - J_x[i, j_left] ) / (2*dx)
                div_y = ( J_y[i_up, j] - J_y[i_down, j] ) / (2*dy)
                div_J[i, j] = div_x + div_y

                # step esplicito per phi
                phi_new[i, j] = phi[i, j] + dt * div_J[i, j]

        # Swap tra phi e phi_new in modo che phi al passo n+1 sia phi_new
        phi, phi_new = phi_new, phi


# ============================================================
#                    PARAMETRI SIMULAZIONE
# ============================================================

M0 = 5e-5
N_steps = 400000
L = 1
N = 64
dx = L / (N - 1)
dy = L / (N - 1)
epsilon = 5 * dx
dt =  1e-4


# -----------------------------------------------------
#           CONDIZIONE INIZIALE
# -----------------------------------------------------

# 1. Profilo random
# -----------------------------------------------------
# phi_initial = np.random.rand(N, N)



# 2. Profilo rettangolo
# -----------------------------------------------------
# phi_initial = np.zeros((N, N))

# # dimensioni del rettangolo
# h = 16   # altezza in celle (direzione i / riga)
# w = 32   # larghezza in celle (direzione j / colonna)

# # coordinate del rettangolo centrato
# i0 = (N - h) // 2      # riga iniziale
# i1 = i0 + h            # riga finale (esclusa)
# j0 = (N - w) // 2      # colonna iniziale
# j1 = j0 + w            # colonna finale (esclusa)

# phi_initial[i0:i1, j0:j1] = 1.0



# 3. Profilo coseno
# -----------------------------------------------------
x = np.linspace(0, 1, N)
y = np.linspace(0, 1, N)
X, Y = np.meshgrid(x, y)
cos = (1.0 / 15) * (np.cos((2 * np.pi) * X)) + 0.5
phi_initial = (Y < cos).astype(float)
mode = np.cos(2 * np.pi * x)


# -----------------------------------------------------
#           QUANTITÀ FISICHE
# -----------------------------------------------------

mass_array = np.zeros(N_steps)  # Integrale di phi su tutto il dominio

# -----------------------------------------------------
#           CONFIGURAZIONE PLOT
# -----------------------------------------------------

snap_steps = np.array([0, N_steps // 3, 2 * N_steps // 3, N_steps - 1])
snap_array = np.zeros((len(snap_steps), N, N))


# -----------------------------------------------------
#           EVOLUZIONE CAHN_HILLIARD
# -----------------------------------------------------
t0 = time.perf_counter()

dts = [5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4]
colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5']

for dt, c in zip(dts, colors):
    
    phi0 = phi_initial.copy()
    energy_array = np.zeros(N_steps)
    
    # evolve con questo dt e questo energy_array
    evolve_cahn_hilliard(phi0, N_steps, dt, dx, dy, epsilon, M0, snap_steps, snap_array, mass_array, energy_array)

    t = np.arange(N_steps) * dt
    plt.plot(t, energy_array, color=c, label=f'dt={dt:g}')

    # print(dt, energy_array[0], energy_array[10], energy_array[100])
    
plt.xlabel('t')
plt.ylabel('E(t)')
plt.title('Energia libera vs tempo per diversi dt')
plt.legend()
plt.xlim(0.0, 2.0)
plt.tight_layout()
plt.show()
t1 = time.perf_counter()
print(f"Tempo esecuzione: {t1 - t0:.2f} s")


# -----------------------------------------------------
#           PLOT CAMPI
# -----------------------------------------------------

# fig, axes = plt.subplots(2, 2, figsize=(10, 8))
# for k, step in enumerate(snap_steps):
#     ax = axes.flat[k]
#     im = ax.imshow(snap_array[k], origin="lower", extent=[0, L, 0, L], cmap="coolwarm")
#     ax.set_title(rf'$\phi(t={step*dt:.2f})$')
#     ax.set_xlabel('x'); ax.set_ylabel('y')
#     plt.colorbar(im, ax=ax)
# plt.tight_layout()
# plt.show()


# -----------------------------------------------------
#           CONSERVAZIONE DELL'INTEGRALE DI PHI
# -----------------------------------------------------

# n_steps = np.arange(len(mass_array))
# t = n_steps * dt
# plt.figure(figsize=(6, 4))
# plt.plot(t[:10000], mass_array[:10000])
# plt.xlabel("t")
# plt.ylabel(r"Integral of $\phi$")
# plt.title(r"Mass conservation")
# plt.tight_layout()
# plt.show()


# -----------------------------------------------------
#           PLOT ENERGIA
# -----------------------------------------------------

# n_steps = np.arange(len(energy_array))
# t = n_steps * dt
# plt.figure(figsize=(6, 4))
# plt.plot(t, energy_array)
# plt.xlabel("t")
# plt.ylabel(r"E(t)")
# plt.title(r"Energy vs time")
# plt.tight_layout()
# plt.show()