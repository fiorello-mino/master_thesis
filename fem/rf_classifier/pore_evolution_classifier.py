from pathlib import Path
import meshio
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

PHASE_FIELD_NAME = "phi"
PHI_THRESHOLD = 0.5


def read_vtu(vtu_path: Path):
    if not vtu_path.is_file():
        raise FileNotFoundError(f"File {vtu_path} non trovato.")
    mesh = meshio.read(vtu_path)
    points = mesh.points
    phi = np.asarray(mesh.point_data[PHASE_FIELD_NAME]).reshape(-1)
    return mesh, points, phi


def phi_mask_blue(phi: np.ndarray, threshold: float = PHI_THRESHOLD) -> np.ndarray:
    return phi < threshold


def build_graph(points, cells):
    rows = []
    cols = []

    for cb in cells:
        cell = cb.data
        for conn_cell in cell:
            i, j, k = int(conn_cell[0]), int(conn_cell[1]), int(conn_cell[2])
            rows += [i, j, j, k, i, k]
            cols += [j, i, k, j, k, i]

    n_points = points.shape[0]
    data = np.ones(len(rows), dtype=bool)
    coo = coo_matrix((data, (rows, cols)), shape=(n_points, n_points))
    return coo.tocsr()


def build_comp_blue(graph_full, phi_mask_blue):
    idx_blue = np.where(phi_mask_blue)[0]
    graph_blue = graph_full[idx_blue[:, None], idx_blue[None, :]]

    n_comp, labels_sub = connected_components(graph_blue, directed=False)

    labels_full = np.full(graph_full.shape[0], -1, dtype=int)
    labels_full[idx_blue] = labels_sub
    return n_comp, labels_full


def pore_masks_from_labels(points: np.ndarray, labels_full: np.ndarray, y_touch_tol: float = 1e-3):
    y_all = points[:, 1]
    y_max_all = float(y_all.max())

    pore_masks = []

    for comp_id in np.unique(labels_full):
        if comp_id < 0:
            continue

        m = labels_full == comp_id
        if m.sum() == 0:
            continue

        y_max_comp = float(y_all[m].max())

        if abs(y_max_comp - y_max_all) < y_touch_tol:
            continue

        pore_masks.append(m)

    return pore_masks


def count_blue_domains(vtu_path: Path, y_touch_tol: float = 1e-3) -> int:
    mesh, points, phi = read_vtu(vtu_path)
    adj = build_graph(points, mesh.cells)
    phi_mask = phi_mask_blue(phi, PHI_THRESHOLD)
    _, labels_full = build_comp_blue(adj, phi_mask)
    pore_masks = pore_masks_from_labels(points, labels_full, y_touch_tol=y_touch_tol)
    return len(pore_masks)


def main():
    vtu_path = Path("surf_20.0.vtu")
    n_bubbles = count_blue_domains(vtu_path)
    print(n_bubbles)


if __name__ == "__main__":
    main()