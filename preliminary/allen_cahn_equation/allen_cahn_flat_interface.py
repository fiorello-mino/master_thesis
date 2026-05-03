import numpy as np
import matplotlib.pyplot as plt

# Imposto i parametri dell'equazione
beta = 10
T = 3600
L = 1
N = 100
dx = L / (N - 1)
delta = dx
epsilon = 5 * delta
dt = 0.02 * delta**2 / (beta * epsilon)

# Imposto la condizione iniziale: interfaccia piana tra le due fasi
phi = np.zeros(N)
for i in range(N//2):
    phi[i] = 1

phi_new = np.empty_like(phi)
phi_initial = np.copy(phi)

# Energia del sistema
energy = np.zeros(T)


# Snapshots per plot
snapshots = {}
steps = [0, T//3, 2*T//3, T-1]

# Integro l'equazione di Allen-Cahn
for n in range(T):
    
    if n in steps:
        snapshots[n] = np.copy(phi)
    
    energy[n] = (2.0 * 18.0 / epsilon) * np.sum(phi**2 * (1 - phi**2))
    #energy[n] = np.sum((18.0/epsilon) * phi**2 * (1.0 - phi)**2) * delta

    laplacian = (
            
        np.roll(phi, 1) 
        + np.roll(phi, -1) 
        - 2 * phi
        
    ) / delta**2
    
    w_prime = (36.0 / epsilon) * phi * (1 + 2 * phi**2 - 3 * phi)
    #w_prime = (36.0 / epsilon) * phi * (1.0 - phi) * (1.0 - 2.0 * phi)
    
    phi_new = phi + dt * beta * (epsilon * laplacian - w_prime)
    
    phi, phi_new = phi_new, phi # swap tra phi e phi_new in modo che phi al passo n+1 sia phi_new


# Plot sistema
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

#Plot energia
n_steps = np.arange(len(energy))
t = n_steps * dt
plt.figure(figsize=(6,4))
plt.plot(t, energy, '-o', markersize=2)
plt.xlabel('t')
plt.ylabel('E(t)')
plt.title('Energy')
plt.tight_layout()
plt.show()
