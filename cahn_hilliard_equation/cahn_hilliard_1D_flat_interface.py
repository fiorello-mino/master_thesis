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
epsilon = 8 * dx
dt =  1e-4


# -----------------------------------------------------
#           CONDIZIONE INIZIALE
# -----------------------------------------------------

# Profilo random
phi = np.zeros(N)
for i in range(N//2):
    phi[i] = 1
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
def derivative_x (phi, dx):
    return ( np.roll(phi, -1) - np.roll(phi, +1) ) / (2*dx)


# Divergenza
# def divergence (grad_x, grad_y, dx, dy):
#     div_x = ( np.roll(grad_x, -1, axis=1) - np.roll(grad_x, 1, axis=1) ) / (2*dx)
#     div_y = ( np.roll(grad_y, -1, axis=0) - np.roll(grad_y, 1, axis=0) ) / (2*dy)
#     return ( np.roll(grad_x, -1, axis=1) - np.roll(grad_x, 1, axis=1) ) / (2*dx)


# Laplaciano
def laplacian (phi, dx):
    laplacian_phi = (
        np.roll(phi, 1)
        + np.roll(phi, -1)
        - 2 * phi
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
    if n % 100000 :
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
    derivative_mu_x = derivative_x (mu, dx)
    # Divergenza di M grad mu
    div_J = derivative_x(M*derivative_mu_x, dx)
    
    # Step esplicito
    phi_new = phi + dt * div_J

    # Swap tra phi e phi_new in modo che phi al passo n+1 sia phi_new
    phi, phi_new = phi_new, phi


# -----------------------------------------------------
#           PLOT CAMPI
# -----------------------------------------------------

x = np.linspace(0, L, N)
x0 = L/2
phi_an = 0.5 * (1 - np.tanh(3 * ( x - x0 ) / epsilon ))
fig, axes = plt.subplots(1, 2, figsize=(9, 5))

for ax, n in zip(axes.ravel(), steps):
    t_fin = n * dt
    ax.plot(x, snapshots[n], label='Numerical')
    ax.plot(x, phi_an, '--', label='Analytical tanh')
    ax.set_title(fr"$\phi(x,t={t_fin:.3f})$")
    ax.set_xlabel("x")
    ax.set_ylabel(r"$\phi$")
    ax.legend()

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
