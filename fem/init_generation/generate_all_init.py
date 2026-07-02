from pathlib import Path
import random
import csv


factor = 1.9
N_FILES = 50


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

    before = text[:start_idx]
    phi_block = text[start_idx:end_idx]
    after = text[end_idx:]

    return before, phi_block, after


def build_shape_line(n_pores: int) -> str:
    names = ["rectangle"]
    for i in range(1, n_pores + 1):
        names.append(f"rectangle{i}")
    shape_exp = " + ".join(names)
    return f"surf->phi->shape:                               {shape_exp}"


def generate_base_rectangle(base_sides, base_center):
    lines = []
    lines.append(f"rectangle->sides length: [{base_sides[0]},{base_sides[1]}]")
    lines.append(f"rectangle->center:       [{base_center[0]},{base_center[1]}]")
    lines.append(" ")
    return "\n".join(lines)


def generate_pore_rectangle(i: int, rectangle_sides, x_center, y_center):
    lines = []
    lines.append(f"rectangle{i}->sides length:  [{rectangle_sides[0]},{rectangle_sides[1]}]")
    lines.append(f"rectangle{i}->center:    [{x_center},{y_center}]")
    lines.append(" ")
    return "\n".join(lines)


def sample_params_from_case(distance: str, depth: str):
    if distance == "near":
        k_spacing = random.uniform(1.5, 2.3)
    elif distance == "far":
        k_spacing = random.uniform(4.0, 8.0)
    else:
        raise ValueError(f"Distanza non valida: {distance}")

    if depth == "deep":
        ratio = random.uniform(12.0, 20.0)
    elif depth == "shallow":
        ratio = random.uniform(4.0, 9.0)
    else:
        raise ValueError(f"Profondità non valida: {depth}")

    return k_spacing, ratio


def generate_all_pores(
    n_pores: int,
    epsilon: float,
    x_min: float,
    x_max: float,
    y_center: float,
    height_min: float,
    height_max: float,
    width_max: float,
    distance: str,
    depth: str,
):
    L = x_max - x_min
    k_spacing, ratio = sample_params_from_case(distance, depth)

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
                rectangle_sides = (w, Ly)
                blocks.append(generate_pore_rectangle(i, rectangle_sides, x_c, y_center))

            return "\n".join(blocks), n_pores, k_spacing, ratio, w, Ly

        n_pores -= 1

    raise RuntimeError("Impossibile piazzare denti: nemmeno con n_pores=1 si trova w valido.")


def generate_phi_block(
    epsilon: float,
    width_max: float,
    distance: str,
    depth: str,
):
    lines = []

    lines.append("#       PHI")
    lines.append(" ")
    lines.append("surf->phi->mode:                                shape % external file , constant")
    lines.append("surf->phi->external file:             init/test.arh")
    lines.append("surf->phi->constant:                            1.")
    lines.append(" ")

    x_min = -0.5 * factor
    x_max = 0.5 * factor

    L_base_x = x_max - x_min
    L_base_y = 0.2 * factor
    base_sides = (L_base_x, L_base_y)
    base_center = (0.0 * factor, 0.5 * factor)

    n_pores = random.randint(1, 7)
    y_center = 0.5 * factor
    height_min = L_base_y + 0.2 * factor
    height_max = 2.0 * factor - 10 * epsilon

    base_block = generate_base_rectangle(base_sides, base_center)

    pores_block, n_pores_final, k_spacing, ratio, w, Ly = generate_all_pores(
        n_pores=n_pores,
        epsilon=epsilon,
        x_min=x_min,
        x_max=x_max,
        y_center=y_center,
        height_min=height_min,
        height_max=height_max,
        width_max=width_max,
        distance=distance,
        depth=depth,
    )

    lines.append(build_shape_line(n_pores_final))
    lines.append("surf->phi->shape->inner value:                  0")
    lines.append("surf->phi->shape->outer value:                  1")
    lines.append("surf->phi->shape->center:             [ 0. , 0. ]")
    lines.append(" ")
    lines.append("surf->phi->shape->eps:                          ${surf->eps}")
    lines.append(" ")

    lines.append(base_block)
    lines.append(pores_block)
    lines.append(" ")

    meta = {
        "distance": distance,
        "depth": depth,
        "n_pores": n_pores_final,
        "k_spacing": k_spacing,
        "ratio": ratio,
        "pore_width": w,
        "pore_height": 0.5*Ly - 0.1*factor,
    }

    return "\n".join(lines), meta


def random_case():
    distance = random.choice(["near", "far"])
    depth = random.choice(["shallow", "deep"])
    return distance, depth


def replace_output_directory(text: str, new_output_dir: str) -> str:
    target_key = "output->directory:"
    lines = text.splitlines()

    replaced = False
    for i, line in enumerate(lines):
        if line.strip().startswith(target_key):
            lines[i] = f"{target_key:<55}{new_output_dir}"
            replaced = True
            break

    if not replaced:
        raise RuntimeError("Linea 'output->directory:' non trovata nel template.")

    return "\n".join(lines) + "\n"


def main():
    path = Path("init_template.dat")
    before, _, after = get_phi_block(path)

    out_dir = Path("/home/fiorello/init_files/test/pores")
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "summary.csv"

    epsilon = 0.0130208 * factor
    width_max = 0.055 * factor

    fieldnames = [
        "tag",
        "distance",
        "depth",
        "n_pores",
        "k_spacing",
        "ratio",
        "pore_width",
        "pore_height",
        "output_dir",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=fieldnames)
        writer.writeheader()

        for idx in range(N_FILES):
            tag = f"{idx:03d}"
            distance, depth = random_case()

            new_phi_block, meta = generate_phi_block(
                epsilon=epsilon,
                width_max=width_max,
                distance=distance,
                depth=depth,
            )

            new_text = before + new_phi_block + after

            new_output_dir = f"/scratch/fiorello/data_test/pores/{tag}"
            new_text = replace_output_directory(new_text, new_output_dir)

            filename = f"{tag}.dat"
            out_path = out_dir / filename
            out_path.write_text(new_text)

            writer.writerow({
                "tag": tag,
                "distance": meta["distance"],
                "depth": meta["depth"],
                "n_pores": meta["n_pores"],
                "k_spacing": meta["k_spacing"],
                "ratio": meta["ratio"],
                "pore_width": meta["pore_width"],
                "pore_height": meta["pore_height"],
                "output_dir": new_output_dir,
            })

            print(
                f"Creato {out_path} | distance={meta['distance']}, depth={meta['depth']}, "
                f"n_pores={meta['n_pores']} | output={new_output_dir}"
            )

    print(f"\nCSV scritto in: {csv_path}")


if __name__ == "__main__":
    main()
