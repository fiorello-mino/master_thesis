# params.py

N = 64
dx = 1 / N

dt = 1e-6
n_steps = 1_000_000
steps_per_save = 10_000

epsilon = 5 * dx
M0 = 5e-5

out_dir = "snapshots"
live_plot = False
