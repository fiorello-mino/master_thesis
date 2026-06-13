from pathlib import Path
import random


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


def build_shape_line(n_teeth: int) -> str:
    names = ["rectangle"]
    for i in range(1, n_teeth + 1):
        names.append(f"rectangle{i}")
    shape_exp = " + ".join(names)
    return f"surf->phi->shape:                               {shape_exp}"


def generate_base_rectangle(base_sides, base_center):
    lines = []
    lines.append(f"rectangle->sides length: [{base_sides[0]},{base_sides[1]}]")
    lines.append(f"rectangle->center:       [{base_center[0]},{base_center[1]}]")
    lines.append(" ")
    return "\n".join(lines)


def generate_tooth_rectangle(i: int, rectangle_sides, x_center, y_center):
    lines = []
    lines.append(f"rectangle{i}->sides length:  [{rectangle_sides[0]},{rectangle_sides[1]}]")
    lines.append(f"rectangle{i}->center:    [{x_center},{y_center}]")
    lines.append(" ")
    return "\n".join(lines)


def sample_candidate_params(case_hint=None):
    """
    case_hint guida il sampling iniziale.
    La classificazione finale deep/shallow viene comunque fatta a posteriori.
    near/far dipende solo da k_spacing:
      near <=> k_spacing <= 3
      far  <=> k_spacing > 3
    """
    if case_hint is None:
        case_hint = random.choice([
            "near_deep",
            "near_shallow",
            "far_deep",
            "far_shallow",
        ])

    if case_hint == "near_deep":
        k_spacing = random.randint(1, 3)
        depth_ratio = random.uniform(3.0, 8.0)

    elif case_hint == "near_shallow":
        k_spacing = random.randint(1, 3)
        depth_ratio = random.uniform(0.3, 2.0)

    elif case_hint == "far_deep":
        k_spacing = random.randint(4, 6)
        depth_ratio = random.uniform(3.0, 8.0)

    elif case_hint == "far_shallow":
        k_spacing = random.randint(4, 6)
        depth_ratio = random.uniform(0.3, 2.0)

    else:
        raise ValueError(f"Caso non valido: {case_hint}")

    n_teeth = random.randint(1, 8)
    return case_hint, k_spacing, depth_ratio, n_teeth


def generate_all_teeth(
    n_teeth: int,
    epsilon: float,
    x_min: float,
    x_max: float,
    y_center: float,
    height_min: float,
    height_max: float,
    width_max: float,
    k_spacing: int,
    depth_ratio: float,
):
    L = x_max - x_min

    while n_teeth >= 1:
        if n_teeth <= 2:
            w_target = L / 2.0
        elif n_teeth <= 8:
            denom = (n_teeth - 1) * k_spacing + 1
            w_target = L / denom
        else:
            denom = (n_teeth - 1) * k_spacing + 1
            w_target = 0.8 * L / denom

        w_max_final = min(w_target, width_max)
        w = max(2 * epsilon, w_max_final)

        success = False

        for _ in range(10000):
            d_c = (k_spacing + 1) * w
            span = (n_teeth - 1) * d_c + w

            if span <= L - (10 * epsilon):
                success = True
                break

            new_w = max(2 * epsilon, 0.9 * w)
            if new_w == w:
                break

            w = new_w

        if success:
            x_mid = 0.5 * (x_min + x_max)
            first_center = x_mid - 0.5 * span + 0.5 * w
            x_centers = [first_center + i * d_c for i in range(n_teeth)]

            gap = k_spacing * w
            Ly = depth_ratio * gap
            Ly = min(max(Ly, height_min), height_max)

            ratio_eff = Ly / gap if gap > 0 else float("inf")

            blocks = []
            for i, x_c in enumerate(x_centers, start=1):
                rectangle_sides = (w, Ly)
                block_i = generate_tooth_rectangle(i, rectangle_sides, x_c, y_center)
                blocks.append(block_i)

            info = {
                "n_teeth_final": n_teeth,
                "k_spacing": k_spacing,
                "w": w,
                "gap": gap,
                "Ly": Ly,
                "depth_ratio_input": depth_ratio,
                "depth_ratio_eff": ratio_eff,
            }

            return "\n".join(blocks), info

        n_teeth -= 1

    raise RuntimeError(
        "Impossibile piazzare denti: nemmeno con n_teeth=1 si trova w valido."
    )


def classify_case(k_spacing, Ly, w, ratio_threshold):

    near_far = "near" if k_spacing <= 3 else "far"

    gap = k_spacing * w
    ratio_eff = Ly / gap
    deep_shallow = "deep" if ratio_eff >= ratio_threshold else "shallow"

    case_eff = f"{near_far}_{deep_shallow}"
    return case_eff, ratio_eff, gap


def generate_phi_block(
    epsilon: float,
    width_max: float,
    case_target=None,
    ratio_threshold=4.0,
    max_tries=300,
) -> str:
    x_min = -0.5
    x_max = 0.5

    L_base_x = x_max - x_min
    L_base_y = 0.2
    base_sides = (L_base_x, L_base_y)
    base_center = (0.0, 0.5)

    y_center = 0.5
    height_min = L_base_y + 0.8
    height_max = 2.0 - 2 * epsilon

    valid_cases = {None, "near_deep", "near_shallow", "far_deep", "far_shallow"}
    if case_target not in valid_cases:
        raise ValueError(f"case_target non valido: {case_target}")

    for _ in range(max_tries):
        case_hint, k_spacing, depth_ratio, n_teeth_initial = sample_candidate_params(case_hint=case_target)

        teeth_block, info = generate_all_teeth(
            n_teeth=n_teeth_initial,
            epsilon=epsilon,
            x_min=x_min,
            x_max=x_max,
            y_center=y_center,
            height_min=height_min,
            height_max=height_max,
            width_max=width_max,
            k_spacing=k_spacing,
            depth_ratio=depth_ratio,
        )

        case_eff, ratio_eff, gap = classify_case(
            k_spacing=info["k_spacing"],
            Ly=info["Ly"],
            w=info["w"],
            ratio_threshold=ratio_threshold,
        )

        if case_target is None or case_eff == case_target:
            lines = []

            lines.append("#       PHI")
            lines.append(" ")
            lines.append("surf->phi->mode:                                shape % external file , constant")
            lines.append("surf->phi->external file:             init/test.arh")
            lines.append("surf->phi->constant:                            1.")
            lines.append(" ")

            lines.append(build_shape_line(info["n_teeth_final"]))
            lines.append("surf->phi->shape->inner value:                  0")
            lines.append("surf->phi->shape->outer value:                  1")
            lines.append("surf->phi->shape->center:             [ 0. , 0. ]")
            lines.append(" ")
            lines.append("surf->phi->shape->eps:                          ${surf->eps}")
            lines.append(" ")

            base_block = generate_base_rectangle(base_sides, base_center)
            lines.append(base_block)
            lines.append(teeth_block)
            lines.append(" ")

            return "\n".join(lines)

    raise RuntimeError(
        f"Non sono riuscito a generare un campione del tipo {case_target} in {max_tries} tentativi."
    )


def main():
    path = Path("pore2D.dat")
    before, phi_block, after = get_phi_block(path)

    out_dir = Path("/home/fiorello/mesoEvo/install_seq/init")

    new_phi_block = generate_phi_block(
        epsilon=0.01953125,
        width_max=0.08,
        case_target="near_deep",
        ratio_threshold=4.0,
        max_tries=300,
    )

    new_text = before + new_phi_block + after

    filename = "prova.dat"
    out_path = out_dir / filename
    out_path.write_text(new_text)


if __name__ == "__main__":
    main()