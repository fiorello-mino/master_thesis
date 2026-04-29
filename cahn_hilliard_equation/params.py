# params.py

N = 64
dx = 1.0 / N

dt = 1e-6
n_steps = 50_000_000
steps_per_save = 500_000

epsilon = 10 * dx
M0 = 5e-5

out_dir = "snapshots"
live_plot = True