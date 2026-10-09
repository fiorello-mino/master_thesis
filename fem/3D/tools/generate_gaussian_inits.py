#!/usr/bin/env python3
"""Generate AMDiS init files with random, distinct Gaussian parameters."""

from decimal import Decimal, InvalidOperation
from pathlib import Path, PurePosixPath
import random
import re

# ================= PARAMETRI DA MODIFICARE =================
BASE_DIR = Path("/home/fiorello/mesoEvo/install_seq/init/")
TEMPLATE = BASE_DIR / 'gaussian_template_P07.dat'
INIT_DIR = BASE_DIR / 'init_gaussian/'

SIGMA_MIN = 1.5  # Esempio: modificare secondo l'intervallo desiderato.
SIGMA_MAX = 3.0
AMPLITUDE_MIN = -2.0
AMPLITUDE_MAX = -1.0
STEP = 0.1  # Passo fisso della griglia, per sigma e ampiezza.
NUM_FILES = 10
SEED = 4  # None per una nuova estrazione casuale a ogni esecuzione.
SHAPE = '-gaussian + -plane'
MACRO_FILE_NAME = './macro/trenchB_16_16_60.3d'

# Directory madre sul computer che esegue AMDiS.
# Il suo nome (qui P07) diventa anche il prefisso delle simulazioni.
# Esempio di output: .../P07/P07_gaussian_S1.0_A2.0
SIMULATION_OUTPUT_BASE = '/scratch/fiorello/data3D/gaussian/P08'
# ==========================================================


def tenths(value):
    try:
        scaled = Decimal(str(value)) * 10
        if not scaled.is_finite() or scaled != scaled.to_integral_value():
            raise ValueError
        return int(scaled)
    except (InvalidOperation, ValueError, OverflowError):
        raise ValueError('Usare un valore multiplo di 0.1.')


def fmt(value):
    return f'{Decimal(value) / 10:.1f}'


def replace_key(text, key, value):
    pattern = rf'(?m)^[ \t]*{re.escape(key)}:[^\n]*$'
    text, count = re.subn(pattern, lambda _: f'{key}: {value}', text)
    if count != 1:
        raise ValueError(f'Attesa una sola definizione di {key}; trovate {count}.')
    return text


def render(template, sigma, amplitude, output, shape, macro_file_name):
    text = replace_key(template, 'surf->phi->shape', shape)
    # Anchored keys avoid matching substrings such as energy->corner.
    text = re.sub(r'(?m)^[ \t]*cylinder->[^\n]*\n?', '', text)
    text = re.sub(
        r'(?m)^[ \t]*gaussian->(?:sigma \+ amplitude|sigma|amplitude):[^\n]*\n?',
        '', text,
    )
    parameters = f'gaussian->sigma + amplitude: [{fmt(sigma)}, {fmt(sigma)}, {fmt(amplitude)}]'
    text = text.replace(f'surf->phi->shape: {shape}',
                        f'surf->phi->shape: {shape}\n{parameters}', 1)
    mesh_match = re.search(r'(?m)^[ \t]*surf->space->mesh:[ \t]*([^\n%#]+)', text)
    if mesh_match is None:
        raise ValueError('Manca surf->space->mesh nel template.')
    text = replace_key(text, f'{mesh_match[1].strip()}->macro file name', macro_file_name)
    return replace_key(text, 'output->directory', str(output))


def main():
    try:
        if STEP != 0.1:
            raise ValueError('Questo script usa il passo fisso STEP = 0.1.')
        sigma_min, sigma_max = tenths(SIGMA_MIN), tenths(SIGMA_MAX)
        amplitude_min, amplitude_max = tenths(AMPLITUDE_MIN), tenths(AMPLITUDE_MAX)
        if not 0 < sigma_min <= sigma_max:
            raise ValueError('Richiesto 0 < SIGMA_MIN <= SIGMA_MAX.')
        if not amplitude_min <= amplitude_max < 0:
            raise ValueError('Richiesto AMPLITUDE_MIN <= AMPLITUDE_MAX < 0.')
        choices = [(s, a) for s in range(sigma_min, sigma_max + 1)
                   for a in range(amplitude_min, amplitude_max + 1)]
        if not isinstance(NUM_FILES, int) or not 1 <= NUM_FILES <= len(choices):
            raise ValueError(f'NUM_FILES deve essere tra 1 e {len(choices)} (senza duplicati).')
        template = Path(TEMPLATE).read_text()
        output_base = PurePosixPath(SIMULATION_OUTPUT_BASE)
        if not output_base.name or output_base.name in {'.', '..'}:
            raise ValueError('SIMULATION_OUTPUT_BASE deve indicare una directory madre con un nome.')
        pairs = random.Random(SEED).sample(choices, NUM_FILES)
        init_dir = Path(INIT_DIR)
        files = []
        for sigma, amplitude in pairs:
            # L'ampiezza resta negativa nell'init; nel nome compare il modulo.
            name = f'gaussian_S{fmt(sigma)}_A{fmt(abs(amplitude))}_{output_base.name}'
            path = init_dir / f'{name}.dat'
            if path.exists():
                raise ValueError(f'File gia esistente: {path}. Cambiare INIT_DIR nel codice.')
            files.append((path, render(template, sigma, amplitude, output_base / name,
                                       SHAPE, MACRO_FILE_NAME)))
        init_dir.mkdir(parents=True, exist_ok=True)
        for path, content in files:
            with path.open('x') as handle:
                handle.write(content)
            print(path)
    except (OSError, ValueError) as error:
        raise SystemExit(f'Errore: {error}')


if __name__ == '__main__':
    main()
