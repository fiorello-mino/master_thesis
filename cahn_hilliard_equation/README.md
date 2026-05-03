# Cahn-Hilliard Equation

This folder contains the code developed for the numerical simulation of the Cahn-Hilliard equation as part of the master's thesis project. The implementation is organized as a small Python package, with separate folders for source code, executable scripts, and generated results.

## Project structure

```text
cahn_hilliard_equation/
├── results/
├── scripts/
└── src/
    └── cahn_hilliard/
        ├── __init__.py
        ├── free_energy.py
        ├── initial_conditions.py
        ├── operators.py
        ├── parameters.py
        ├── solver.py
        └── utils.py
```

The `src/cahn_hilliard/` folder contains the core implementation of the model and is structured as a Python package. The `scripts/` folder is intended for runnable scripts, while `results/` stores generated outputs such as simulation data, figures, or animations.

## Modules

The package currently includes the following modules:

- `solver.py`: numerical routines for time evolution and simulation workflow.
- `parameters.py`: definition and handling of simulation parameters.
- `free_energy.py`: free-energy terms and chemical potential definitions.
- `operators.py`: spatial differential operators used by the solver.
- `initial_conditions.py`: generation of initial fields for the simulations.
- `utils.py`: helper functions used across the project.

## Scripts

The `scripts/` directory contains executable entry points for running simulations, generating datasets, or post-processing outputs. These scripts import the package modules from `src/cahn_hilliard/`.

## Results

The `results/` directory is used to store generated outputs from the simulations. Depending on the workflow, this may include saved fields, processed datasets, plots, and GIF animations.

## Usage

From the `cahn_hilliard_equation` directory, scripts can be executed after making the package visible to Python. Typical usage is through the scripts stored in the `scripts/` folder, which rely on imports from the `cahn_hilliard` package.

Example import inside a script:

```python
from cahn_hilliard.solver import *
from cahn_hilliard.parameters import *
```

Example command:

```bash
python scripts/main.py
```

## Notes

This code is part of the `master_thesis` repository and is currently under active development. The folder organization is intended to keep the numerical implementation modular, reusable, and easier to maintain during thesis work.