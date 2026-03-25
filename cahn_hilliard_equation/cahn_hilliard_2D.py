import numpy as np
import matplotlib.pyplot as plt

# ============================================================
#                    PARAMETRI SIMULAZIONE
# ============================================================

M = 10
T = 100000
L = 1
N = 100
dx = L / (N - 1)
dy = L / (N - 1)
delta = dx
epsilon = 3 * delta
dt = dx**4 / (35 * M)


# -----------------------------------------------------
#           CONDIZIONE INIZIALE
# -----------------------------------------------------

# Profilo random
phi = np.random.rand(N, N)
phi_new = np.empty_like(phi)
phi_initial = np.copy(phi)

# Profilo 2
# x = np.linspace(0, 1, 100)
# y = np.linspace(0, 1, 100)
# X, Y = np.meshgrid(x, y)
# R0 = 0.1
# phi = np.sqrt(X**2 + Y**2)
# phi_new = np.empty_like(phi)
# phi_initial = np.copy(phi)


# -----------------------------------------------------
#           QUANTITÀ FISICHE
# -----------------------------------------------------

energy = np.zeros(T)  # Evoluzione energia nel tempo
mass = np.zeros(T)  # Integrale di phi su tutto il dominio

# -----------------------------------------------------
#           CONFIGURAZIONE PLOT
# -----------------------------------------------------

snapshots = {}
steps = [0, T // 3, 2 * T // 3, T - 1]  # Momenti chiave per snapshot


# ============================================================
#                INTEGRAZIONE CAHN-HILLIARD
# ============================================================

for n in range(T):
    # Salva snapshots selezionati
    if n in steps:
        snapshots[n] = np.copy(phi)

    # Calcola energia libera
    energy[n] = (2.0 * 18.0 / epsilon) * np.sum(phi**2 * (1 - phi**2))

    # Calcola integrale di phi
    mass[n] = np.sum(phi)

    # -----------------------------------------------------
    #            CALCOLO DI MU
    # -----------------------------------------------------

    # Laplaciano discreto di phi
    laplacian_phi = (
        np.roll(phi, 1, axis=0)
        + np.roll(phi, -1, axis=0)
        + np.roll(phi, 1, axis=1)
        + np.roll(phi, -1, axis=1)
        - 4 * phi
    ) / delta**2

    # Derivata del potenziale double well
    w_prime = (36.0 / epsilon) * phi * (1 + 2 * phi**2 - 3 * phi)

    # Calcolo di mu
    mu = -epsilon * laplacian_phi + w_prime

    # -----------------------------------------------------
    #           STEP ESPLICITO PER PHI
    # -----------------------------------------------------

    # Laplaciano discreto di mu
    laplacian_mu = (
        np.roll(mu, 1, axis=0)
        + np.roll(mu, -1, axis=0)
        + np.roll(mu, 1, axis=1)
        + np.roll(mu, -1, axis=1)
        - 4 * mu
    ) / delta**2

    # Step esplicito
    phi_new = phi + dt * M * laplacian_mu

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
