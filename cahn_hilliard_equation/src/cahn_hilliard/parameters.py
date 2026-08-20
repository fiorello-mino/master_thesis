# parameters.py

N1 = 128
N2 = 64
dx = 1.0 / (N1-1)
dy = 1.0 / (N2-1)

dt = 1e-6
n_steps = 20_000_000
steps_per_save = 100_000

epsilon = 5 * dy
M0 = 5e-5

#out_dir = "snapshots"
live_plot = False
