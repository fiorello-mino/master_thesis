#!/usr/bin/env python3

from pathlib import Path

# Numero di box in x, y e z
NX = 10
NY = 10
NZ = 70

# Dimensione dei box lungo x, y e z
BOX_SIZE_X = 0.04
BOX_SIZE_Y = 0.04
BOX_SIZE_Z = 0.04

# Valore assegnato a ogni box
BOX_VALUE = 1

OUTPUT_FILE = Path("poro_04_04_28.bt3")


def generate_bt3():
    row = " ".join([str(BOX_VALUE)] * NX)

    with OUTPUT_FILE.open("w") as file:
        file.write("dim: 3\n")
        file.write(f"boxes: {NX} {NY} {NZ}\n")
        file.write(
            f"box-size: {BOX_SIZE_X} {BOX_SIZE_Y} {BOX_SIZE_Z}\n"
        )

        for k in range(NZ):
            file.write(f"z = {k}:\n")
            for _ in range(NY):
                file.write(row + "\n")
            file.write("\n")


if __name__ == "__main__":
    generate_bt3()

    lx = NX * BOX_SIZE_X
    ly = NY * BOX_SIZE_Y
    lz = NZ * BOX_SIZE_Z

    print(f"File generato: {OUTPUT_FILE}")
    print(f"Layer z: {NZ}")
    print(f"Matrice per layer: {NX} x {NY}")
    print(f"Dimensione dominio: {lx} x {ly} x {lz}")
