from pathlib import Path
import sys
import numpy as np
from numba import njit
from scipy import ndimage


# ========== SCRIPT VARIABLES ==========
TEST_DIR = Path("/scratch/fiorello/test3D/square/train1/fv_clip_mse")
# --------------------------------------


# ========== PARAMETERS ===========
DX = 0.025
DY = 0.025
DZ = 0.025
EPS = 0.1
PHI_MIN_DOMAIN = 0.5
PHI_MAX_DOMAIN = 1.0
# =================================


def file_time(path: Path, prefix: str) -> float:
    """Tempo nel nome del file, usato per ordinare i frame .npy."""
    return float(path.stem.removeprefix(prefix))


def count_domains(phi: np.ndarray) -> int:
    """Componenti 3D della fase 0.5 <= phi <= 1, con adiacenza per faccia.

    Non si collegano i bordi opposti: il dominio spaziale non è periodico.
    """
    mask = (phi >= PHI_MIN_DOMAIN))
    structure = ndimage.generate_binary_structure(3, 1)  # 6 vicini
    _, count = ndimage.label(mask, structure=structure)
    return int(count)


def load_volume(path: Path) -> np.ndarray:
    phi = np.load(path)
    if phi.ndim != 3 or min(phi.shape) < 2:
        raise ValueError(f"Atteso volume 3D con almeno 2 celle per asse: {path}, shape={phi.shape}")
    return phi


@njit(fastmath=True)
def w_field_3D(phi: np.ndarray, epsilon: float, w: np.ndarray):
    nx, ny, nz = phi.shape
    factor = 18.0 / epsilon
            
    for x in range(nx):
        for y in range(ny):
            for z in range(nz):
                phi_xyz = phi[x, y, z]
                w[x, y, z] = factor * phi_xyz * phi_xyz * (1 - phi_xyz) * (1 - phi_xyz)


@njit(fastmath=True)
def grad_3D_neumann(
    phi: np.ndarray, 
    dx: float,
    dy: float,
    dz: float,
    grad_x: np.ndarray, 
    grad_y: np.ndarray,
    grad_z: np.ndarray
):
    nx, ny, nz = phi.shape
    
    dx2_inv = 1.0 / (2.0 * dx)
    dy2_inv = 1.0 / (2.0 * dy)
    dz2_inv = 1.0 / (2.0 * dz)
    
    # --- gradiente in x ---
    for y in range(ny):
        for z in range(nz):
            for x in range(1, nx - 1):
                grad_x[x, y, z] = (phi[x + 1, y, z] - phi[x - 1, y, z]) * dx2_inv
            
            grad_x[0, y, z] = (phi[1, y, z] - phi[0, y, z]) * dx2_inv
            grad_x[nx - 1, y, z] = (phi[nx - 1, y, z] - phi[nx - 2, y, z]) * dx2_inv

    # --- gradiente in y ---
    for x in range(nx):
        for z in range(nz):
            for y in range(1, ny - 1):
                grad_y[x, y, z] = (phi[x, y + 1, z] - phi[x, y - 1, z]) * dy2_inv
            
            grad_y[x, 0, z] = (phi[x, 1, z] - phi[x, 0, z]) * dy2_inv
            grad_y[x, ny - 1, z] = (phi[x, ny - 1, z] - phi[x, ny - 2, z]) * dy2_inv

    # --- gradiente in z ---
    for x in range(nx):
        for y in range(ny):
            for z in range(1, nz - 1):
                grad_z[x, y, z] = (phi[x, y, z + 1] - phi[x, y, z - 1]) * dz2_inv
            
            grad_z[x, y, 0] = (phi[x, y, 1] - phi[x, y, 0]) * dz2_inv
            grad_z[x, y, nz - 1] = (phi[x, y, nz - 1] - phi[x, y, nz - 2]) * dz2_inv
            
            
@njit(fastmath=True)
def compute_e_3D(phi: np.ndarray, epsilon: float, dx: float, dy: float, dz: float) -> float:
    nx, ny, nz = phi.shape
    eps2 = 0.5 * epsilon
    
    w_local = np.empty_like(phi)
    gx = np.empty_like(phi)
    gy = np.empty_like(phi)
    gz = np.empty_like(phi)
    
    w_field_3D(phi, epsilon, w_local)
    grad_3D_neumann(phi, dx, dy, dz, gx, gy, gz)
    
    total_E = 0.0
    for x in range(nx):
        for y in range(ny):
            for z in range(nz):
                grad2 = (
                    gx[x, y, z] * gx[x, y, z] +
                    gy[x, y, z] * gy[x, y, z] +
                    gz[x, y, z] * gz[x, y, z]
                )
                f_xyz = w_local[x, y, z] + eps2 * grad2
                total_E += f_xyz
    
    return total_E * dx * dy * dz
    


def main() -> None:
    
    # ----- prendo le directory delle simulazioni -----
    if not TEST_DIR.is_dir():
        sys.exit(f"{TEST_DIR} directory non trovata.")
        
    sim_folders = sorted(d for d in TEST_DIR.iterdir() if d.is_dir())
    if len(sim_folders) == 0:
        sys.exit("0 simulazioni trovate.")
    print(f"Trovate {len(sim_folders)} simulazioni.")
    
    # ----- per ogni simulazione calcolo energia, massa ed estremi -----
    for sim in sim_folders:
        evo_file = sim / "evo.txt"
        if not evo_file.is_file():
            sys.exit(f"File evo non trovato in {sim}")

        with evo_file.open() as f:
            stats = f.readlines()
        if not stats or not stats[0].startswith("#"):
            raise ValueError(f"Intestazione evo.txt mancante in {evo_file}")
        # Un eventuale riepilogo di un'esecuzione precedente viene rigenerato.
        data_lines = [line for line in stats[1:] if line.strip() and not line.lstrip().startswith("#")]
        if not data_lines:
            raise ValueError(f"Nessuna riga dati in {evo_file}")

        rows = {}
        for line in data_lines:
            parts = line.split()
            if len(parts) < 10 or len(parts) > 16:
                raise ValueError(f"Numero colonne inatteso in {evo_file}: {line.strip()}")
            time = int(parts[0])
            if float(parts[0]) != time or time in rows:
                raise ValueError(f"Tempo non intero o duplicato in {evo_file}: {parts[0]}")
            rows[time] = parts[:10]  # le colonne 11-16 vengono ricalcolate

        # ----- individua i frame predetti (possono essere salvati a passo > 1) -----
        pred_dir = sim / "pred_npy"
        if not pred_dir.is_dir():
            sys.exit(f"Cartella pred_npy non trovata in {sim}")

        frames_pred = sorted(pred_dir.glob("*.npy"), key=lambda p: file_time(p, "phi_"))
        if not frames_pred:
            sys.exit(f"0 frames pred nella cartella {pred_dir}")

        pred_by_time = {}
        for path in frames_pred:
            time_float = file_time(path, "phi_")
            time = int(time_float)
            if time_float != time or time in pred_by_time or time not in rows:
                raise ValueError(f"Tempo pred non valido o assente da evo.txt: {path}")
            pred_by_time[time] = path

        # ----- ordina i frame veri; evo.txt usa l'indice temporale 0,1,... -----
        true_dir = sim / "true_npy"
        if not true_dir.is_dir():
            sys.exit(f"Cartella true_npy non trovata in {sim}")

        frames_true = sorted(true_dir.glob("*.npy"), key=lambda p: file_time(p, "surf_"))
        if not frames_true:
            sys.exit(f"0 frames true nella cartella {true_dir}")

        times = sorted(rows)
        if times != list(range(len(times))) or len(frames_true) < len(times):
            raise ValueError(f"I tempi evo.txt devono essere 0..N-1 e avere i corrispondenti true_npy: {sim}")

        extrema = {
            "true": {"min": (float("inf"), None), "max": (-float("inf"), None)},
            "pred": {"min": (float("inf"), None), "max": (-float("inf"), None)},
        }
        output_rows = []
        for time in times:
            true = load_volume(frames_true[time])
            pred = load_volume(pred_by_time[time]) if time in pred_by_time else None
            if pred is not None and pred.shape != true.shape:
                raise ValueError(f"Shape pred/true diversa a t={time} in {sim}")

            true_min, true_max = float(np.min(true)), float(np.max(true))
            pred_min = float(np.min(pred)) if pred is not None else float("nan")
            pred_max = float(np.max(pred)) if pred is not None else float("nan")

            for name, low, high in (("true", true_min, true_max), ("pred", pred_min, pred_max)):
                if np.isfinite(low) and low < extrema[name]["min"][0]:
                    extrema[name]["min"] = (low, time)
                if np.isfinite(high) and high > extrema[name]["max"][0]:
                    extrema[name]["max"] = (high, time)

            parts = rows[time]
            parts[5:9] = [f"{true_min:.17g}", f"{pred_min:.17g}",
                          f"{true_max:.17g}", f"{pred_max:.17g}"]

            e_true = compute_e_3D(true, EPS, DX, DY, DZ)
            m_true = np.sum(true) * DX * DY * DZ
            if pred is None:
                e_pred = m_pred = float("nan")
                n_pred = "NaN"
            else:
                e_pred = compute_e_3D(pred, EPS, DX, DY, DZ)
                m_pred = np.sum(pred) * DX * DY * DZ
                n_pred = str(count_domains(pred))
            n_true = count_domains(true)

            parts.extend((f"{e_true:.6e}", f"{e_pred:.6e}",
                          f"{m_true:.6e}", f"{m_pred:.6e}",
                          str(n_true), n_pred))
            output_rows.append("\t".join(parts) + "\n")

        # Conserva le prime dieci colonne già usate dagli altri script.
        header = stats[0].split("\t11:", 1)[0].rstrip("\n")
        header += ("\t11: E_true\t12: E_pred\t13: mass_true\t14: mass_pred"
                   "\t15: domains_true\t16: domains_pred\n")
        summary = "# sequence_extrema (solo frame .npy disponibili):"
        for name in ("true", "pred"):
            for kind in ("min", "max"):
                value, time = extrema[name][kind]
                summary += f" {kind}_{name}={value:.17g} (t={time})"
        summary += "\n"

        with evo_file.open("w") as f:
            f.write(header)
            f.writelines(output_rows)
            f.write(summary)
        print(f"Analizzata {sim.name}: {len(times)} tempi, {len(pred_by_time)} predizioni.")


if __name__ == "__main__":
    main()
