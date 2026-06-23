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
    return points, phi


def phi_mask_blue(phi: np.ndarray, threshold: float = PHI_THRESHOLD) -> np.ndarray:
    return phi < threshold


def build_graph(points, cells):
    
    rows = []
    cols = []
    
    for cb in cells:
        cell = cb.data              # celle di un tipo
        for conn_cell in cell:      # elenco di celle di quel tipo
            i, j, k = int(conn_cell[0]), int(conn_cell[1]), int(conn_cell[2])
            rows += [i, j, j, k, i, k]
            cols += [j, i, k, j, k, i]
    
    n_points = points.shape[0]
    data = np.ones(len(rows), dtype=bool)
    coo = coo_matrix((data, (rows, cols)), shape=(n_points, n_points))
    
    return coo.tocsr()
    
    
def build_comp_blue(graph_full, phi_mask_blue):
    
    idx_blue = np.where(phi_mask_blue)[0]    # indici dei punti in cui phi < 0.5
    graph_blue = graph_full[idx_blue[:, None], idx_blue[None, :]] # grafo solo dei punti in cui phi < 0.5
    
    n_comp, labels_sub = connected_components(graph_blue, directed=False)
    
    labels_full = np.full(graph_full.shape[0], -1, dtype=int)
    labels_full[idx_blue] = labels_sub  # vale -1 sugli indici dei punti in cui phi > 0.5, vale k = numero della componente connessa dove phi < 0.5

    return n_comp, labels_full


def pore_masks_from_labels(points: np.ndarray, labels_full: np.ndarray, y_touch_tol: float = 1e-3):
    y_all = points[:, 1]
    y_max_all = float(y_all.max())
    
    print("y_max_all: ", y_max_all)

    pore_masks = []

    for comp_id in np.unique(labels_full):
        if comp_id < 0:
            continue  # salta i non-blu

        m = labels_full == comp_id # m ha shape (n_points) e vale true nei punti della componente connessa comp_id
        if m.sum() == 0:
            continue

        y_max_comp = float(y_all[m].max())
        
        print("y_max_comp: ", y_max_comp)
        # se tocca il top -> scarta
        if abs(y_max_comp - y_max_all) < y_touch_tol:
            continue

        pore_masks.append(m)

    return pore_masks

def main():
    vtu_path = Path("surf_20.00000.vtu")

    mesh = meshio.read(vtu_path)
    points = mesh.points
    phi = np.asarray(mesh.point_data[PHASE_FIELD_NAME]).reshape(-1)

    cb = mesh.cells[0]
    print("Tipo celle:", cb.type)
    print("conn shape:", cb.data.shape)
    print("prime 5 celle:\n", cb.data[:5])
    print(points.shape[0])

    adj = build_graph(points, mesh.cells)

    print("Numero di nodi:", adj.shape)
    print("Numero di archi (non orientati):", adj.nnz // 2)

    phi_mask = phi_mask_blue(phi, PHI_THRESHOLD)
    n_comp_blue, labels_full = build_comp_blue(adj, phi_mask)

    print("Numero di componenti connesse blu:", n_comp_blue)
    
    pore_masks = pore_masks_from_labels(points, labels_full)
    
    print("pore_masks:", pore_masks)




if __name__ == "__main__":
    main()