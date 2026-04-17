# params.py

N = 128
dx = 1 / N

dt = 1e-6
n_steps = 10_000_000
steps_per_save = 100_000

epsilon = 10 * dx
M0 = 5e-5

out_dir = "snapshots"
live_plot = True