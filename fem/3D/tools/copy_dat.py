from pathlib import Path
import shutil

CARTELLA_A = r"/archive/roberto/poresAMDIS/rectPattern"
CARTELLA_B = r"/data/fiorello/pores3D/data_train/rectangle/init"

origine = Path(CARTELLA_A).resolve()
destinazione = Path(CARTELLA_B).resolve()

if not origine.is_dir():
    raise SystemExit(f"Cartella di origine non trovata: {origine}")

if origine == destinazione:
    raise SystemExit("Le cartelle A e B devono essere diverse.")

# Cerca tutti i .dat nelle sottocartelle, escludendo B.
file_da_copiare = [
    file
    for file in origine.rglob("iso*.dat")
    if file.is_file() and destinazione not in file.parents
]

destinazione.mkdir(parents=True, exist_ok=True)

for file in file_da_copiare:
    nuovo_file = destinazione / file.name

    # Evita di sovrascrivere file con lo stesso nome.
    numero = 1
    while nuovo_file.exists():
        nuovo_file = destinazione / f"{file.stem}_{numero}{file.suffix}"
        numero += 1

    shutil.copy2(file, nuovo_file)

print(f"Copiati {len(file_da_copiare)} file in {destinazione}")
