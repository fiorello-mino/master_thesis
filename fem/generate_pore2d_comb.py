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
    gap_min: float                    
):
    blocks = []

    L = x_max - x_min

    max_tries = 10000
    for _ in range(max_tries):
        widths = [random.uniform(2*epsilon, width_max) for _ in range(n_teeth)]
        total_width = sum(widths) + (n_teeth - 1) * gap_min
        if total_width <= L:
            break
    else:
        raise RuntimeError("Impossibile trovare una configurazione di larghezze compatibile dopo 10000 tentativi.")
    
    cur_x = x_min + gap_min
    
    for i in range(n_teeth):
        
        if i > 0:
            cur_x += gap_min
            
        Lx = widths[i]
        Ly = random.uniform(height_min, height_max)

        rectangle_sides = (Lx, Ly)
        x_center = cur_x + 0.5*Lx
        block_i = generate_tooth_rectangle(i+1, rectangle_sides, x_center, y_center)
        blocks.append(block_i)
        
        cur_x += Lx

    return "\n".join(blocks)

    
    
def generate_phi_block(
    epsilon: float,
    width_max: float
) -> str:
    
    
    lines = []
    
    lines.append("#       PHI")
    lines.append(" ")
    lines.append("surf->phi->mode:                                shape % external file , constant")
    lines.append("surf->phi->external file:				init/test.arh")
    lines.append("surf->phi->constant:                            1.")
    lines.append(" ")
    
    x_max = 0.5
    x_min = -0.5
    L_base_x = 2.0
    L_base_y = random.uniform(0.2, 0.5)
    base_sides = (L_base_x, L_base_y)
    center_base_x = 0.0
    center_base_y = 0.5
    base_center = (center_base_x, center_base_y)
    
    n_teeth = random.randint(2, 8)
    height_min = L_base_y + 0.7
    height_max = 2.0 - L_base_y
    y_center = 0.5
    gap_min = 2*epsilon
    
    lines.append(build_shape_line(n_teeth))
    lines.append("surf->phi->shape->inner value:                  0")
    lines.append("surf->phi->shape->outer value:                  1")
    lines.append("surf->phi->shape->center:		        [ 0. , 0. ]")
    lines.append(" ")
    lines.append("surf->phi->shape->eps:                          ${surf->eps}")
    lines.append(" ")
    
    base_block = generate_base_rectangle(base_sides, base_center)
    teeth_block = generate_all_teeth(
        n_teeth, epsilon, x_min, x_max, y_center,
        height_min, height_max, width_max, gap_min
    )
    
    lines.append(base_block)
    lines.append(teeth_block)
    lines.append(" ")

    block = "\n".join(lines)
    return block
    
        
        
def main():
    
    path = Path("pore2D.dat")
    before, phi_block, after = get_phi_block(path)
    
    new_phi_block = generate_phi_block(
        epsilon = 0.01953125,
        width_max = 0.3
    )
    
    new_text = before + new_phi_block + after
    
    out_path = Path("prova.dat")
    out_path.write_text(new_text)
    

if __name__ == "__main__":
    main()
