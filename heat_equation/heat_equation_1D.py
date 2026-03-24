import numpy as np
import matplotlib.pyplot as plt

# Imposto i parametri dell'equazione del calore
D = 1
T = 4000
L = 1
N = 100
dx = L / (N - 1)
x = np.linspace(0, L, N)
dt = 0.4 * dx**2 / D
r = D * dt / dx**2

# Imposto la condizione iniziale: gaussiana
x0 = 0.5 * L
sigma = L / 10
u = np.exp(-((x - x0) ** 2) / (2 * sigma**2))
u_new = np.empty_like(u)
u_initial = np.copy(u)

# Integro l'equazione del calore
for t in range(T):
    for i in range(N):
        i_left = (i - 1) % N
        i_right = (i + 1) % N
        u_new[i] = u[i] + r * (u[i_right] - 2 * u[i] + u[i_left])

    u = np.copy(u_new)


plt.figure()

plt.plot(x, u_initial, label="t = 0")
plt.plot(x, u, label="t = T_final")

plt.xlabel("x")
plt.ylabel("u(x,t)")
plt.title("Equazione del calore 1D: evoluzione gaussiana")
plt.legend()

plt.show()
