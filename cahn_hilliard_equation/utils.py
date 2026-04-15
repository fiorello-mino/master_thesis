# utils.py

from pathlib import Path
import numpy as np

from pathlib import Path
import numpy as np

def load_snapshots(snap_dir="snapshots"):
    """
    Carica tutti gli snapshot .npy da una cartella e il file times.npy.
    """
    
    snap_dir = Path(snap_dir)

    if not snap_dir.exists():
        raise FileNotFoundError(f"Cartella non trovata: {snap_dir}")

    times_file = snap_dir / "times.npy"
    if not times_file.exists():
        raise FileNotFoundError(f"File non trovato: {times_file}")

    times = np.load(times_file, allow_pickle=False)

    files = []
    for f in snap_dir.glob("*.npy"):
        if f.name == "times.npy":
            continue
        files.append(f)

    files.sort()

    if not files:
        raise ValueError("Nessuno snapshot trovato")

    if len(times) != len(files):
        raise ValueError("Numero di tempi e snapshot diverso")

    snapshots = []
    for f in files:
        snapshots.append(np.load(f, allow_pickle=False))

    snapshots = np.array(snapshots)

    return times, snapshots