from pathlib import Path
import numpy as np

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


def generate_all_teeth(n_teeth: int, rectangle_sides, x_min, x_max, y_center):
    
    blocks = []
    
    # rettangoli pettine
    centers_distance = (x_max - x_min) / n_teeth
    x_centers = np.arange(x_min, x_max, centers_distance) + 0.5*centers_distance
    
    for i in range(1, n_teeth+1):
        blocks.append(generate_tooth_rectangle(i, rectangle_sides, x_centers[i-1], y_center))
        
    return "\n".join(blocks)

    
def generate_phi_block(
    n_teeth: int, 
    base_sides, 
    base_center, 
    rectangle_sides, 
    x_min, 
    x_max, 
    y_center
) -> str:
    
    lines = []
    
    lines.append("#       PHI")
    lines.append(" ")
    lines.append("surf->phi->mode:                                shape % external file , constant")
    lines.append("surf->phi->external file:				init/test.arh")
    lines.append("surf->phi->constant:                            1.")
    lines.append(" ")
    lines.append(build_shape_line(n_teeth))
    lines.append("surf->phi->shape->inner value:                  1")
    lines.append("surf->phi->shape->outer value:                  0")
    lines.append("surf->phi->shape->center:		        [ 0. , 0. ]")
    lines.append(" ")
    lines.append("surf->phi->shape->eps:                          ${surf->eps}")
    lines.append(" ")
    
    base_block = generate_base_rectangle(base_sides, base_center)
    teeth_block = generate_all_teeth(n_teeth, rectangle_sides, x_min, x_max, y_center)
    lines.append(base_block)
    lines.append(teeth_block)
    lines.append(" ")

    block = "\n".join(lines)
    return block
    
    

        
def main():
    
    path = Path("pore2D.dat")
    before, phi_block, after = get_phi_block(path)
    
    new_phi_block = generate_phi_block(
        n_teeth = 4,
        base_sides = (1.2, 0.4),
        base_center = (0, 0.5),
        rectangle_sides = (0.1, 0.4),
        x_min = -0.5,
        x_max = 0.5,
        y_center = 0.4
    )
    
    new_text = before + new_phi_block + after
    
    out_path = Path("prova.dat")
    out_path.write_text(new_text)
    

if __name__ == "__main__":
    main()