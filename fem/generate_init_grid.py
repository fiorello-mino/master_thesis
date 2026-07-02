from pathlib import Path
import csv
import itertools
import numpy as np

factor = 1.9
OUT_DIR = Path("/home/fiorello/init_files/test/grid")
OUT_DIR.mkdir(parents=True, exist_ok=True)
TEMPLATE = Path("/home/fiorello/master_thesis/fem/init_template.dat")

OUT_DIR2 = Path("/home/fiorello/master_thesis/fem")
META_CSV = OUT_DIR / "summary.csv"

epsilon = 0.0130208 * factor
width_max = 0.055 * factor

N_PORES_LIST = range(1, 8)
K_VALUES = np.linspace(2.0, 5.0, 12)
DEPTH_VALUES = [8, 9, 10, 11, 12, 13]


def load_pore2d(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"File {path} non trovato.")
    return path.read_text()


def get_phi_block(path: Path):
    text = load_pore2d(path)
    start_marker = "#       PHI"
    start_idx = text.find(start_marker)
    if start_idx == -1:
        raise RuntimeError(f"Sezione {start_marker} non trovata.")
    end_marker = "#############################################################################################"
    end_idx = text.find(end_marker, start_idx + 1)
    if end_idx == -1:
        raise RuntimeError(f"Stringa {end_marker} non trovata.")
    return text[:start_idx], text[start_idx:end_idx], text[end_idx:]


def build_shape_line(n_pores: int) -> str:
    names = ["rectangle"] + [f"rectangle{i}" for i in range(1, n_pores + 1)]
    return f"surf->phi->shape:                               {' + '.join(names)}"


def generate_base_rectangle(base_sides, base_center):
    return "\n".join([
        f"rectangle->sides length: [{base_sides[0]},{base_sides[1]}]",
        f"rectangle->center:       [{base_center[0]},{base_center[1]}]",
        " ",
    ])


def generate_pore_rectangle(i: int, rectangle_sides, x_center, y_center):
    return "\n".join([
        f"rectangle{i}->sides length:  [{rectangle_sides[0]},{rectangle_sides[1]}]",
        f"rectangle{i}->center:    [{x_center},{y_center}]",
        " ",
    ])


def generate_all_pores(
    n_pores: int,
    epsilon: float,
    x_min: float,
    x_max: float,
    y_center: float,
    height_min: float,
    height_max: float,
    width_max: float,
    k_spacing: float,
    ratio: float,
):
    L = x_max - x_min

    while n_pores >= 1:
        if n_pores <= 2:
            w_target = L / 2.0
        elif n_pores <= 8:
            denom = (n_pores - 1) * k_spacing + 1
            w_target = L / denom
        else:
            denom = (n_pores - 1) * k_spacing + 1
            w_target = 0.8 * L / denom

        w_max_final = min(w_target, width_max)
        w = max(4 * epsilon, w_max_final)

        success = False
        for _ in range(10000):
            d_c = (k_spacing + 1) * w
            span = (n_pores - 1) * d_c + w

            if span <= L - (10 * epsilon):
                success = True
                break

            new_w = max(4 * epsilon, 0.9 * w)
            if new_w == w:
                break
            w = new_w

        if success:
            d_c = (k_spacing + 1) * w
            span = (n_pores - 1) * d_c + w

            x_mid = 0.5 * (x_min + x_max)
            first_center = x_mid - 0.5 * span + 0.5 * w
            x_centers = [first_center + i * d_c for i in range(n_pores)]

            Ly = 2 * ratio * w
            Ly = min(max(Ly, height_min), height_max)

            blocks = []
            for i, x_c in enumerate(x_centers, start=1):
                blocks.append(generate_pore_rectangle(i, (w, Ly), x_c, y_center))

            return {
                "pores_block": "\n".join(blocks),
                "n_pores_effective": n_pores,
                "width": w,
                "height": 0.5*Ly - 0.1*factor, #profondità del poro = Ly/2 - altezza del ceiling
            }

        n_pores -= 1

    raise RuntimeError("Impossibile piazzare pori.")


def replace_output_directory(text: str, new_output_dir: str) -> str:
    target_key = "output->directory:"
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.strip().startswith(target_key):
            lines[i] = f"{target_key:<55}{new_output_dir}"
            return "\n".join(lines) + "\n"
    raise RuntimeError("Linea 'output->directory:' non trovata nel template.")


def generate_phi_block_grid(n_pores_requested: int, k_spacing: float, depth_ratio: float):
    lines = []
    lines += [
        "#       PHI",
        " ",
        "surf->phi->mode:                                shape % external file , constant",
        "surf->phi->external file:             init/test.arh",
        "surf->phi->constant:                            1.",
        " ",
    ]

    x_min = -0.5 * factor
    x_max = 0.5 * factor
    L_base_x = x_max - x_min
    L_base_y = 0.2 * factor
    base_sides = (L_base_x, L_base_y)
    base_center = (0.0, 0.5 * factor)
    y_center = 0.5 * factor
    height_min = L_base_y + 0.2 * factor
    height_max = 2.0 * factor - 10 * epsilon

    base_block = generate_base_rectangle(base_sides, base_center)

    geom = generate_all_pores(
        n_pores=n_pores_requested,
        epsilon=epsilon,
        x_min=x_min,
        x_max=x_max,
        y_center=y_center,
        height_min=height_min,
        height_max=height_max,
        width_max=width_max,
        k_spacing=k_spacing,
        ratio=depth_ratio,
    )

    lines.append(build_shape_line(geom["n_pores_effective"]))
    lines.append("surf->phi->shape->inner value:                  0")
    lines.append("surf->phi->shape->outer value:                  1")
    lines.append("surf->phi->shape->center:             [ 0. , 0. ]")
    lines.append(" ")
    lines.append("surf->phi->shape->eps:                          ${surf->eps}")
    lines.append(" ")
    lines.append(base_block)
    lines.append(geom["pores_block"])
    lines.append(" ")

    meta = {
        "n_pores_requested": n_pores_requested,
        "n_pores_effective": geom["n_pores_effective"],
        "k_spacing": float(k_spacing),
        "depth_ratio": float(depth_ratio),
        "width": geom["width"],
        "height": geom["height"],
    }
    return "\n".join(lines), meta


def main():
    before, _, after = get_phi_block(TEMPLATE)

    rows = []
    tag_counter = 0

    for n_pores, k_spacing, depth in itertools.product(N_PORES_LIST, K_VALUES, DEPTH_VALUES):
        tag = f"{tag_counter:03d}"

        new_phi_block, meta = generate_phi_block_grid(
            n_pores_requested=n_pores,
            k_spacing=float(k_spacing),
            depth_ratio=float(depth),
        )

        new_text = before + new_phi_block + after
        new_output_dir = f"/scratch/fiorello/mesoEvo_install_seq/dataset_pores_grid/{tag}"
        new_text = replace_output_directory(new_text, new_output_dir)

        out_path = OUT_DIR / f"{tag}.dat"
        out_path.write_text(new_text)

        rows.append({
            "tag": tag,
            "init_file": str(out_path),
            "output_dir": new_output_dir,
            "case_region": "intermediate",
            "n_pores_requested": meta["n_pores_requested"],
            "n_pores_effective": meta["n_pores_effective"],
            "k_spacing": meta["k_spacing"],
            "depth_ratio": meta["depth_ratio"],
            "width": meta["width"],
            "height": meta["height"],
        })

        tag_counter += 1

    fieldnames = [
        "tag",
        "init_file",
        "output_dir",
        "case_region",
        "n_pores_requested",
        "n_pores_effective",
        "k_spacing",
        "depth_ratio",
        "width",
        "height",
    ]

    with open(META_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Creati {len(rows)} init file.")
    print(f"Metadata salvato in: {META_CSV}")


if __name__ == "__main__":
    main()
