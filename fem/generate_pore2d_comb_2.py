from pathlib import Path
import numpy as np
import random



def load_pore2d(path: Path) -> str:
    
    if not path.is_file():
        raise FileNotFoundError(f"File {path} non trovato.")
    
    text = path.read_text()
    return text



def get_phi_block(path: Path):
    
    text = load_pore2d(path)
    
    start_marker: str = "#       PHI"
    start_idx = text.find(start_marker)
    if start_idx == -1:
        raise RuntimeError(f"Sezione {start_marker} non trovata.")
        
    end_marker: str = "#############################################################################################"
    end_idx = text.find(end_marker, start_idx + 1)
    if end_idx == -1:
        raise RuntimeError(f"Stringa {end_marker} non trovata.")
    
    before = text[:start_idx]
    phi_block = text[start_idx:end_idx]
    after = text[end_idx:]
    
    return before, phi_block, after
    
    

def build_shape_line(n_teeth: int) -> str:
    
    names = ["rectangle"]
    
    for i in range(1, n_teeth+1):
        names.append(f"rectangle{i}")
    
    shape_exp: str = " + ".join(names)
    line: str = f"surf->phi->shape:                               {shape_exp}"
    
    return line



def generate_base_rectangle(base_sides, base_center):
    
    lines = []
    
    lines.append(f"rectangle->sides length:	[{base_sides[0]},{base_sides[1]}]")
    lines.append(f"rectangle->center:		[{base_center[0]},{base_center[1]}]")
    lines.append(" ")
    
    block = "\n".join(lines)
    return block



def generate_tooth_rectangle(i: int, rectangle_sides, x_center, y_center):
    
    lines = []
    
    lines.append(f"rectangle{i}->sides length:	[{rectangle_sides[0]},{rectangle_sides[1]}]")
    lines.append(f"rectangle{i}->center:	[{x_center},{y_center}]")
    lines.append(" ")
    
    block = "\n".join(lines)
    return block



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
):
    blocks = []

    L = x_max - x_min

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
    
    for _ in range(10000): 
        d_c = (k_spacing+1) * w
        span = (n_teeth - 1) * d_c + w
        if span <= L-(4*epsilon):
            break
       
        w = max(2 * epsilon, 0.9 * w)
    else:
        raise RuntimeError(
            f"Impossibile piazzare {n_teeth} denti: span={span:.4f} > L={L:.4f} anche dopo riduzione di w."
        )

    x_mid = 0.5 * (x_min + x_max)
    first_center = x_mid - 0.5 * span + 0.5 * w
    x_centers = [first_center + i * d_c for i in range(n_teeth)]

    for i, x_c in enumerate(x_centers, start=1):
        #Ly = random.uniform(height_min, height_max)
        Ly = min(40*w, height_max)
        rectangle_sides = (w, Ly)
        block_i = generate_tooth_rectangle(i, rectangle_sides, x_c, y_center)
        blocks.append(block_i)

    return "\n".join(blocks)

    
    
def generate_phi_block(
    epsilon: float,
    width_max: float,
    k_spacing: int,
) -> str:
    lines = []

    # header PHI
    lines.append("#       PHI")
    lines.append(" ")
    lines.append("surf->phi->mode:                                shape % external file , constant")
    lines.append("surf->phi->external file:             init/test.arh")
    lines.append("surf->phi->constant:                            1.")
    lines.append(" ")

    # dominio in x
    x_min = -0.5
    x_max = 0.5

    # base rossa
    L_base_x = x_max - x_min          # coerente con il dominio
    L_base_y = 0.2
    base_sides = (L_base_x, L_base_y)
    base_center = (0.0, 0.5)

    # parametri denti blu
    n_teeth = 6
    y_center = 0.5
    height_min = L_base_y + 0.8
    height_max = 2.0 - 2*epsilon

    # riga shape (ora che conosci n_teeth)
    lines.append(build_shape_line(n_teeth))
    # inner/outer a seconda di come hai definito fase blu/rossa
    lines.append("surf->phi->shape->inner value:                  0")
    lines.append("surf->phi->shape->outer value:                  1")
    lines.append("surf->phi->shape->center:             [ 0. , 0. ]")
    lines.append(" ")
    lines.append("surf->phi->shape->eps:                          ${surf->eps}")
    lines.append(" ")

    # blocchi base + denti
    base_block = generate_base_rectangle(base_sides, base_center)
    teeth_block = generate_all_teeth(
        n_teeth=n_teeth,
        epsilon=epsilon,
        x_min=x_min,
        x_max=x_max,
        y_center=y_center,
        height_min=height_min,
        height_max=height_max,
        width_max=width_max,
        k_spacing=k_spacing,
    )

    lines.append(base_block)
    lines.append(teeth_block)
    lines.append(" ")

    return "\n".join(lines)
    
        
        
def main():
    path = Path("pore2D.dat")
    before, phi_block, after = get_phi_block(path)

    epsilon = 0.01953125
    width_max = 0.1

    out_dir = Path("/home/fiorello/mesoEvo/install_seq/init")

    new_phi_block = generate_phi_block(
    epsilon=0.01953125,
    width_max=0.05,
    k_spacing=3,
    )
    new_text = before + new_phi_block + after

    filename = f"prova.dat"
    out_path = out_dir / filename
    out_path.write_text(new_text)
    

if __name__ == "__main__":
    main()
