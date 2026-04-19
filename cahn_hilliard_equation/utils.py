# utils.py

from pathlib import Path
import numpy as np

import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation


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


def make_ch_gif(
    snap_dir="snapshots",
    output_gif="cahn_hilliard.gif",
    output_dir="results"
    fps=10,
    cmap="RdBu_r"
):
    """
    Legge tutti gli snapshots e crea una gif di evoluzione del sistema.
    """
    
    os.makedirs(output_dir, exist_ok=True)

    output_gif = os.path.join(output_dir, output_name)
    
    files = sorted(
        f for f in os.listdir(snap_dir)
        if f.endswith(".npy") and f != "times.npy"
    )

    if not files:
        raise RuntimeError("Nessun file .npy trovato nella cartella snapshots.")

    # Carico tempi
    times_path = os.path.join(snap_dir, "times.npy")
    times = None
    if os.path.exists(times_path):
        times = np.load(times_path)

        if len(times) != len(files):
            print("Attenzione: len(times) != numero di snapshot")
    
    first = np.load(os.path.join(snap_dir, files[0]))

    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(first, cmap=cmap, origin="lower")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("φ")
    
    # Titolo iniziale
    if times is not None:
        ax.set_title(f"t = {times[0]:.3e}")
    else:
        ax.set_title(files[0])
    
    ax.set_axis_off()
    fig.tight_layout()

    def update(frame_idx):
        fname = files[frame_idx]
        phi = np.load(os.path.join(snap_dir, fname))
        im.set_data(phi)
        
        if times is not None:
            ax.set_title(f"t = {times[frame_idx]:.3e}")
        else:
            ax.set_title(fname)
        
        return (im,)

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(files),
        interval=1000 / fps,
        blit=True,
    )

    writer = animation.PillowWriter(fps=fps)
    ani.save(output_gif, writer=writer)
    plt.close(fig)