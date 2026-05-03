import numpy as np
import matplotlib.pyplot as plt

# Imposto i parametri dell'equazione del calore
D = 1
T = 30000
L = 1
N = 50
dx = L / (N - 1)
dy = L / (N - 1)
delta = dx
x = np.linspace(0, L, N)
y = np.linspace(0, L, N)
X, Y = np.meshgrid(x, y)
dt = 0.2 * delta**2 / D

# Imposto la condizione iniziale: gaussiana
x0 = 0.5 * L
y0 = 0.5 * L
sigma = L / 10
u = np.exp(-(((X - x0) ** 2 + (Y - y0) ** 2) / (2 * sigma**2)))
u_new = np.empty_like(u)
u_initial = np.copy(u)

# Integro l'equazione del calore
for t in range(T):
    for i in range(N):
        i_left = (i - 1) % N
        i_right = (i + 1) % N
        for j in range(N):
            j_left = (j - 1) % N
            j_right = (j + 1) % N
            u_new[i, j] = u[i, j] + (dt * D / delta**2) * (
                u[i_right, j]
                + u[i_left, j]
                + u[i, j_right]
                + u[i, j_left]
                - 4 * u[i, j]
            )

    u = np.copy(u_new)


fig, axes = plt.subplots(1, 2, figsize=(10, 4))

im0 = axes[0].imshow(u_initial, origin="lower", extent=[0, L, 0, L])
axes[0].set_title("u(x,y, t=0)")
axes[0].set_xlabel("x")
axes[0].set_ylabel("y")
fig.colorbar(im0, ax=axes[0])

im1 = axes[1].imshow(u, origin="lower", extent=[0, L, 0, L])
axes[1].set_title("u(x,y, t=T)")
axes[1].set_xlabel("x")
axes[1].set_ylabel("y")
fig.colorbar(im1, ax=axes[1])

plt.tight_layout()
plt.show()
