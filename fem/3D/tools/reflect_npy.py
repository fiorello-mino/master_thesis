from pathlib import Path
import shutil

import numpy as np


# =====================================================================
# CONFIGURAZIONE
# =====================================================================

INPUT_DIR = Path(
    '/data/fiorello/pores3D/dataset_ext_test/square'
)

OUTPUT_DIR = Path(
    '/data/fiorello/pores3D/dataset_ext_test/square_reflected'
)

# False:
#   interrompe se la cartella output esiste già.
#
# True:
#   cancella completamente square_reflected e la ricrea.
OVERWRITE = False
# =====================================================================


def prepare_output_directory(output_dir: Path, overwrite: bool) -> None:
    """
    Crea la cartella output.

    Se esiste già:
    - OVERWRITE=False -> errore, per evitare sovrascritture accidentali;
    - OVERWRITE=True  -> elimina la cartella interamente e la ricrea.
    """
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f'La cartella output esiste già:\n'
                f'  {output_dir}\n\n'
                f'Imposta OVERWRITE = True se vuoi rigenerarla.'
            )

        print(f'Cancello la cartella output esistente: {output_dir}')
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=False)


def make_reflected_supercell(phi: np.ndarray) -> np.ndarray:
    """
    Costruisce una supercella 2x2x1 tramite duplicazione con riflessioni.

    Convenzione input:
        phi.shape == (nx, ny, nz)

    Output:
        phi_big.shape == (2*nx, 2*ny, nz)

    Disposizione dei blocchi:

        phi_big[0:nx, 0:ny, :]       = flip x + flip y
        phi_big[0:nx, ny:2*ny, :]    = flip x
        phi_big[nx:2*nx, 0:ny, :]    = flip y
        phi_big[nx:2*nx, ny:2*ny, :] = originale

    Il volume non viene modificato lungo z.
    """
    phi = np.asarray(phi)

    if phi.ndim != 3:
        raise ValueError(
            'Ogni file deve contenere un volume 3D con shape '
            f'(nx, ny, nz), ma è stata trovata shape {phi.shape}.'
        )

    nx, ny, nz = phi.shape

    phi_flip_x = np.flip(phi, axis=0)
    phi_flip_y = np.flip(phi, axis=1)
    phi_flip_xy = np.flip(phi, axis=(0, 1))

    phi_big = np.empty(
        (2 * nx, 2 * ny, nz),
        dtype=phi.dtype,
    )

    # Quadrante inferiore-destro: volume originale.
    phi_big[nx:, ny:, :] = phi

    # Quadrante superiore-destro: riflessione lungo x.
    phi_big[:nx, ny:, :] = phi_flip_x

    # Quadrante inferiore-sinistro: riflessione lungo y.
    phi_big[nx:, :ny, :] = phi_flip_y

    # Quadrante superiore-sinistro: riflessione lungo x e y.
    phi_big[:nx, :ny, :] = phi_flip_xy

    return phi_big


def main() -> None:
    """
    Legge ricorsivamente tutti i file .npy da INPUT_DIR e salva, mantenendo
    la struttura delle sottocartelle, una supercella 2x2x1 riflessa in
    OUTPUT_DIR.
    """
    if not INPUT_DIR.is_dir():
        raise FileNotFoundError(
            f'La directory input non esiste:\n  {INPUT_DIR}'
        )

    npy_files = sorted(INPUT_DIR.rglob('*.npy'))

    if not npy_files:
        raise FileNotFoundError(
            f'Nessun file .npy trovato in:\n  {INPUT_DIR}'
        )

    prepare_output_directory(
        output_dir=OUTPUT_DIR,
        overwrite=OVERWRITE,
    )

    print(f'Input directory:  {INPUT_DIR}')
    print(f'Output directory: {OUTPUT_DIR}')
    print(f'File .npy trovati: {len(npy_files)}')
    print('Operazione: supercella 2x2x1 con reflection lungo x e y.')
    print()

    for index, input_file in enumerate(npy_files, start=1):
        relative_path = input_file.relative_to(INPUT_DIR)
        output_file = OUTPUT_DIR / relative_path

        output_file.parent.mkdir(parents=True, exist_ok=True)

        phi = np.load(input_file)
        phi_big = make_reflected_supercell(phi)

        np.save(output_file, phi_big)

        print(
            f'[{index:>5}/{len(npy_files)}] '
            f'{relative_path} | '
            f'{tuple(phi.shape)} -> {tuple(phi_big.shape)} | '
            f'dtype={phi.dtype}'
        )

    print()
    print('=' * 72)
    print(f'DONE: generated {len(npy_files)} reflected 2x2x1 supercells.')
    print(f'Output: {OUTPUT_DIR}')
    print('=' * 72)


if __name__ == '__main__':
    main()