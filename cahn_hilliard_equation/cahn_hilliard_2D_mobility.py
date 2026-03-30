import numpy as np
import matplotlib.pyplot as plt

# ============================================================
#                    PARAMETRI SIMULAZIONE
# ============================================================

M0 = 5e-5
T = 3000000
L = 1
N = 64
dx = L / (N - 1)
dy = L / (N - 1)
delta = dx
epsilon = 8 * delta
dt =  1e-4


# -----------------------------------------------------
#           CONDIZIONE INIZIALE
# -----------------------------------------------------

# Profilo random
phi = np.random.rand(N, N)
phi_new = np.empty_like(phi)
phi_initial = np.copy(phi)


# -----------------------------------------------------
#           QUANTITÀ FISICHE
# -----------------------------------------------------

mass = np.zeros(T)  # Integrale di phi su tutto il dominio

# -----------------------------------------------------
#           CONFIGURAZIONE PLOT
# -----------------------------------------------------

snapshots = {}
steps = [0, T // 3, 2 * T // 3, T - 1]  # Momenti chiave per snapshot


# -----------------------------------------------------
#           FUNZIONI
# -----------------------------------------------------


# Gradiente 
def gradient (phi, dx, dy):
    grad_x = ( np.roll(phi, -1, axis=1) - np.roll(phi, +1, axis=1) ) / (2*dx)
    grad_y = ( np.roll(phi, -1, axis=0) - np.roll(phi, +1, axis=0) ) / (2*dy)
    return grad_x, grad_y


# Divergenza
def divergence (grad_x, grad_y, dx, dy):
    div_x = ( np.roll(grad_x, -1, axis=1) - np.roll(grad_x, 1, axis=1) ) / (2*dx)
    div_y = ( np.roll(grad_y, -1, axis=0) - np.roll(grad_y, 1, axis=0) ) / (2*dy)
    return div_x + div_y


# Laplaciano
def laplacian (phi, dx):
    laplacian_phi = (
        np.roll(phi, 1, axis=0)
        + np.roll(phi, -1, axis=0)
        + np.roll(phi, 1, axis=1)
        + np.roll(phi, -1, axis=1)
        - 4 * phi
    ) / dx**2
    return laplacian_phi


def w_prime (phi):
    return (36.0 / epsilon) * phi * (1 + 2 * phi**2 - 3 * phi)


def mobility (phi):
    return (36.0 / epsilon) * M0 * phi**2 * (1 - phi)**2 + 1e-6




# ============================================================
#                INTEGRAZIONE CAHN-HILLIARD
# ============================================================

for n in range(T):
    # Salva snapshots selezionati
    if n in steps:
        snapshots[n] = np.copy(phi)

    # Calcola integrale di phi
    mass[n] = np.sum(phi)

    # -----------------------------------------------------
    #            CALCOLO DI MU
    # -----------------------------------------------------

    # Laplaciano di phi
    lapl_phi = laplacian(phi, dx)

    # Derivata del potenziale double well
    w_prime_phi = w_prime(phi)

    # Calcolo di mu
    mu = -epsilon * lapl_phi + w_prime_phi

    # -----------------------------------------------------
    #           STEP ESPLICITO PER PHI
    # -----------------------------------------------------

    # Mobilità M scalare
    M = mobility(phi)

    # Gradiente di mu
    grad_mu_x, grad_mu_y = gradient (mu, dx, dy)
    # Divergenza di M grad mu
    div_J = divergence(M*grad_mu_x, M*grad_mu_y, dx, dy)
    
    # Step esplicito
    phi_new = phi + dt * div_J

    # Swap tra phi e phi_new in modo che phi al passo n+1 sia phi_new
    phi, phi_new = phi_new, phi


# -----------------------------------------------------
#           PLOT CAMPI
# -----------------------------------------------------

fig, axes = plt.subplots(2, 2, figsize=(10, 8))

for ax, n in zip(axes.ravel(), steps):
    im = ax.imshow(snapshots[n], origin="lower", extent=[0, L, 0, L], cmap="coolwarm")
    t_fin = n * dt
    ax.set_title(rf"$\phi(x,y,t={t_fin:.7f})$")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(im, ax=ax)

plt.tight_layout()
plt.show()


# -----------------------------------------------------
#           EVOLUZIONE ENERGIA
# -----------------------------------------------------

# n_steps = np.arange(len(energy))
# t = n_steps * dt
# plt.figure(figsize=(6, 4))
# plt.plot(t, energy, "-o", markersize=2)
# plt.xlabel("t")
# plt.ylabel("E(t)")
# plt.title("Energy")
# plt.tight_layout()
# plt.show()


# -----------------------------------------------------
#           CONSERVAZIONE DELL'INTEGRALE DI PHI
# -----------------------------------------------------

n_steps = np.arange(len(mass))
t = n_steps * dt
plt.figure(figsize=(6, 4))
plt.plot(t, mass, "-o", markersize=1)
plt.xlabel("t")
plt.ylabel(r"Integral of $\phi$")
plt.title(r"Mass conservation")
plt.tight_layout()
plt.show()
