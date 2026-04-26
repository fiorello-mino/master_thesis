# params.py

N = 128
dx = 1.0 / N

dt = 1e-7
n_steps = 1_000_000_000
steps_per_save = 1_000_000

epsilon = 10 * dx
M0 = 5e-5

out_dir = "snapshots"
live_plot = False