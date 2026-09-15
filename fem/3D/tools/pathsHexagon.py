from pathlib import Path

BASE = Path("/data/fiorello/pores3D/data/hexagon_full/iso_P08")
OUTPUT = Path("/scratch/fiorello/test3D/hexagon_full/ext_test.txt")


def frame_index(path: Path) -> int:
    value = path.stem.removeprefix("surf_")
    return round(float(value) * 1_000_000)


def main():
    if not BASE.is_dir():
        raise FileNotFoundError(f"Directory non trovata:\n{BASE}")

    lines = []

    for sim_folder in sorted(BASE.iterdir()):
        if not sim_folder.is_dir():
            continue

        surf_files = sorted(
            sim_folder.glob("surf_*.npy"),
            key=frame_index,
        )

        if not surf_files:
            continue

        line = " ".join(str(path.resolve()) for path in surf_files)
        lines.append(line)

    OUTPUT.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print(f"Folder di simulazione: {len(lines)}")
    print(f"Output: {OUTPUT.resolve()}")

    if lines:
        print(f"Numero di path nella prima riga: {len(lines[0].split())}")


if __name__ == "__main__":
    main()
