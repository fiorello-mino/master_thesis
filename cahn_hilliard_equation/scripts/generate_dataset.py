import os
import sys
import random
import subprocess
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
import cahn_hilliard.parameters as p

N_RUNS = 300
BASE_DIR = "/data/fiorello/dataset_64_2_random_fraction_phase1"
MAX_WORKERS = 6

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
        "initial_condition = phi_initial = random fraction phase 1 from 0.1 to 0.9",
        f"n_runs = {N_RUNS}",
        f"max_workers = {MAX_WORKERS}",
        "run_folders = 0000, 0001, ..., 0299",
        "seed = random 32-bit integer generated independently for each run",
    ]

    txt_path = os.path.join(base_dir, "params.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

def run_single(i):
    run_dir = os.path.join(BASE_DIR, f"{i:04d}")
    done_file = os.path.join(run_dir, "0200.npy")

    # se la run è completa, saltala
    if os.path.exists(done_file):
        return i, "SKIPPED", f"{done_file} already exists"

    # se la cartella esiste ma 0200.npy manca, ricomincia da zero
    if os.path.exists(run_dir):
        shutil.rmtree(run_dir)

    os.makedirs(run_dir, exist_ok=True)

    seed = random.randint(0, 2**32 - 1)

    cmd = [
        sys.executable,
        "scripts/main.py",
        "--seed", str(seed),
        "--out_dir", run_dir,
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = "src"

    result = subprocess.run(cmd, capture_output=True, text=True, env=env)

    if result.returncode == 0:
        return i, "OK", f"seed={seed}\n" + result.stdout.strip()
    else:
        return i, "ERROR", f"seed={seed}\n" + result.stderr.strip()

def main():
    os.makedirs(BASE_DIR, exist_ok=True)
    save_params_txt(BASE_DIR)

    print(f"Lancio {N_RUNS} run in parallelo con MAX_WORKERS = {MAX_WORKERS}")

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(run_single, i) for i in range(N_RUNS)]

        for future in as_completed(futures):
            run_id, status, message = future.result()

            if status == "OK":
                print(f"[OK] run {run_id:04d}")
            elif status == "SKIPPED":
                print(f"[SKIP] run {run_id:04d}")
            else:
                print(f"[ERROR] run {run_id:04d}")
                print(message)

if __name__ == "__main__":
    main()
