import os
import sys
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
import cahn_hilliad.parameters as p

N_RUNS = 1000
BASE_DIR = "/data/fiorello/dataset"
MAX_WORKERS = 10


def save_params_txt(base_dir):
    lines = [
        f"N = {p.N}",
        f"dx = {p.dx}",
        f"dt = {p.dt}",
        f"n_steps = {p.n_steps}",
        f"steps_per_save = {p.steps_per_save}",
        f"epsilon = {p.epsilon}",
        f"M0 = {p.M0}",
        "model = cahn_hilliard_surface_mobility",
        "initial_condition = 0.5 + 0.05 * (rng.random((N, N)) - 0.5)",
        f"n_runs = {N_RUNS}",
        f"max_workers = {MAX_WORKERS}",
        "run_folders = 0000, 0001, ..., 0999",
    ]

    txt_path = os.path.join(base_dir, "params.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def run_single(i):
    run_dir = os.path.join(BASE_DIR, f"{i:04d}")
    os.makedirs(run_dir, exist_ok=True)

    final_file = os.path.join(run_dir, "phi_final.npy")
    if os.path.exists(final_file):
        return i, "SKIPPED", "Run già completato"

    cmd = [
        sys.executable,
        "main.py",
        "--seed", str(i),
        "--out_dir", run_dir,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        return i, "OK", result.stdout.strip()
    else:
        return i, "ERROR", result.stderr.strip()


def main():
    os.makedirs(BASE_DIR, exist_ok=True)
    save_params_txt(BASE_DIR)

    print(f"Lancio {N_RUNS} run in parallelo con MAX_WORKERS = {MAX_WORKERS}")

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(run_single, i) for i in range(N_RUNS)]

        for future in as_completed(futures):
            run_id, status, message = future.result()

            if status == "OK":
                print(f"[OK]    run {run_id:04d}")
            elif status == "SKIPPED":
                print(f"[SKIP]  run {run_id:04d}")
            else:
                print(f"[ERROR] run {run_id:04d}")
                print(message)


if __name__ == "__main__":
    main()