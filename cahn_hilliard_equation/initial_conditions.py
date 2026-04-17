# initial_conditions.py

import numpy as np


def random_profile(N: int) -> np.ndarray:
    
    return np.random.rand(N, N).astype(np.float64)


def rectangle_profile(N: int, h: int = 16, w: int = 32) -> np.ndarray:
    
    phi = np.zeros((N, N))

    # coordinate rettangolo centrato
    i0 = (N - h) // 2
    i1 = i0 + h
    j0 = (N - w) // 2
    j1 = j0 + w

    phi[i0:i1, j0:j1] = 1.0
    
    return phi


def cosine_step_profile(N: int) -> np.ndarray:
    
    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)
    X, Y = np.meshgrid(x, y)

    cos = (1.0 / 15.0) * np.cos(2.0 * np.pi * X) + 0.5
    phi = (Y < cos)
    
    return phi


def smooth_cosine_interface(N: int, dx: float, epsilon: float) -> np.ndarray:
    
    x = (np.arange(N) + 0.5) * dx
    y = (np.arange(N) + 0.5) * dx
    X, Y = np.meshgrid(x, y)

    y0 = y[N // 2]   # punto effettivo della griglia
    a = 1.0 / 100.0
    q = 2.0 * np.pi

    d = Y - (y0 + a * np.cos(q * X))
    phi = 0.5 * (1.0 - np.tanh(3.0 * d / epsilon))
    
    return phi