import subprocess
import sys
from pathlib import Path

N = 10
base_dir = Path("runs")
base_dir.mkdir(exist_ok=True)

for k in range(N):
    out_dir = base_dir / f"run_{k:03d}"

    cmd = [
        sys.executable,
        "main.py",
        "--out_dir", str(out_dir),
        "--seed", str(k),
    ]

    print(f"Lancio run {k} -> {out_dir}")
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"Run {k} fallita")
        break