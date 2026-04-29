import numpy as np

phi0 = np.load("snapshots/0709.npy")
phi1 = np.load("snapshots/0710.npy")

print("0709:", phi0.min(), phi0.max(), np.isnan(phi0).any(), np.isinf(phi0).any())
print("0710:", phi1.min(), phi1.max(), np.isnan(phi1).any(), np.isinf(phi1).any())
print("max abs diff:", np.max(np.abs(phi1 - phi0)))
