from __future__ import annotations

import argparse
import ast
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Iterator

import numpy as np
from numba import njit
import torch
from PIL import Image
from torchvision import transforms

PROJECT_ROOT = Path("/home/fiorello/CRANE_bc")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TRAIN_LOGS_ROOT = Path("/home/machine_learning/train_pores/train_logs")

from src.classes import ConvGRU
from src.utils import seq2npy, seq2png_treaded, seq2vtk


LOGGER = logging.getLogger(__name__)


def best_model_path(log_dir_name: str) -> tuple[Path, int, float, Path]:
    log_dir_path = TRAIN_LOGS_ROOT / log_dir_name
    valid_loss_file = log_dir_path / "valid_loss.txt"

    if not log_dir_path.is_dir():
        raise FileNotFoundError(f"La cartella di log non esiste: {log_dir_path}")

    if not valid_loss_file.is_file():
        raise FileNotFoundError(f"File valid_loss.txt non trovato: {valid_loss_file}")

    min_loss = None
    best_epoch = None

    with valid_loss_file.open("r") as f:
        for epoch, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            try:
                value = float(line)
            except ValueError as e:
                raise ValueError(
                    f"Valore non valido in {valid_loss_file} alla riga {epoch + 1}: {line!r}"
                ) from e

            if min_loss is None or value < min_loss:
                min_loss = value
                best_epoch = epoch

    if best_epoch is None:
        raise ValueError(f"Il file {valid_loss_file} è vuoto o contiene solo righe vuote")

    model_path = log_dir_path / "model" / f"epoch_{best_epoch}.pt"

    if not model_path.is_file():
        raise FileNotFoundError(f"Il miglior modello atteso non esiste: {model_path}")

    LOGGER.info(
        "Best model trovato in %s: epoch=%d, valid_loss=%.8e",
        log_dir_name,
        best_epoch,
        min_loss
    )

    return model_path, best_epoch, min_loss, log_dir_path


def parse_value(raw: str):
    raw = raw.strip()

    if raw == "":
        return raw
    if raw == "True":
        return True
    if raw == "False":
        return False
    if raw == "None":
        return None

    try:
        return ast.literal_eval(raw)
    except Exception:
        pass

    try:
        return int(raw)
    except ValueError:
        pass

    try:
        return float(raw)
    except ValueError:
        pass

    return raw


def load_train_args(args_txt_path: Path) -> dict:
    parsed = {}

    with args_txt_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if "\t : \t" in line:
                key, value = line.split("\t : \t", 1)
            elif ": \t" in line:
                key, value = line.split(": \t", 1)
            elif ":" in line:
                key, value = line.split(":", 1)
            else:
                continue

            parsed[key.strip()] = parse_value(value.strip())

    return parsed


@dataclass
class Config:
    model_path: Path
    sequence_table: Path
    output_folder: Path
    args_txt_path: Path

    img_size: int = 128
    use_cuda: bool = True
    delta_png: int = 1

    min_seq: int = 1
    hidden_units: int = 2
    input_channels: int = 1
    output_channels: int | None = None
    hidden_channels: int = 16
    kernel_size: int = 5
    padding_mode: tuple[str, str] | str = ("circular", "reflect")
    separable: bool = False
    bias: bool = True
    divergence: bool = True
    conservative: bool = False
    num_params: int = 0
    dropout: bool = False
    dropout_prob: float | None = None
    noise_reg: float = 0.0

    epsilon: float = 5.0 / 64
    dx: float = 1.0 / 64
    dt: float = 1e-6
    steps_per_save: int = 100_000
    starting_frame: int = 10

    num_png: int = 100
    num_vtk: int = 0
    num_npy: int = 0
    num_evo: int = 25000
    overwrite: str = "abort"


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Run ConvGRU inference on tabulated sequences."
    )
    parser.add_argument("--log-dir-name", type=str, required=True)
    parser.add_argument("--sequence-table", type=Path, required=True)
    parser.add_argument("--output-folder", type=Path, required=True)
    parser.add_argument("--img-size", type=int, default=None)
    parser.add_argument("--use-cuda", action="store_true")
    parser.add_argument("--delta-png", type=int, default=1)

    parser.add_argument("--min-seq", type=int, default=None)

    parser.add_argument("--dt", type=float, default=1e-6)
    parser.add_argument("--steps_per_save", type=int, default=100_000)
    parser.add_argument("--starting_frame", type=int, default=10)

    parser.add_argument("--num-png", type=int, default=100)
    parser.add_argument("--num-vtk", type=int, default=0)
    parser.add_argument("--num-npy", type=int, default=0)
    parser.add_argument("--num-evo", type=int, default=25000)
    parser.add_argument(
        "--overwrite",
        choices=["abort", "delete", "continue"],
        default="abort",
        help="Comportamento se la cartella di output esiste già."
    )
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='[%(asctime)s] %(levelname)s - %(message)s'
    )

    if args.delta_png < 1:
        raise ValueError("--delta-png deve essere >= 1")

    model_path, best_epoch, min_loss, log_dir_path = best_model_path(args.log_dir_name)
    args_txt_path = log_dir_path / "args.txt"

    if not args_txt_path.is_file():
        raise FileNotFoundError(f"args.txt non trovato: {args_txt_path}")

    train_args = load_train_args(args_txt_path)

    img_size = int(train_args["size"]) if args.img_size is None else args.img_size
    min_seq = int(train_args["subseq_min"]) if args.min_seq is None else args.min_seq

    if min_seq < 1:
        raise ValueError("--min-seq deve essere >= 1")

    padding = train_args["padding"]
    if isinstance(padding, list):
        padding_mode = tuple(padding)
    else:
        padding_mode = padding

    return Config(
        model_path=model_path,
        sequence_table=args.sequence_table,
        output_folder=args.output_folder,
        args_txt_path=args_txt_path,
        img_size=img_size,
        use_cuda=args.use_cuda,
        delta_png=args.delta_png,
        min_seq=min_seq,
        hidden_units=int(train_args["hidden"]),
        input_channels=1,
        output_channels=None,
        hidden_channels=int(train_args["channels"]),
        kernel_size=int(train_args["kernel_size"]),
        padding_mode=padding_mode,
        separable=False,
        bias=bool(train_args["bias"]),
        divergence=bool(train_args["divergence"]),
        conservative=bool(train_args.get("conservative", False)),
        num_params=int(train_args["num_params"]),
        dropout=bool(train_args["dropout"]),
        dropout_prob=train_args.get("dropout_prob"),
        noise_reg=float(train_args.get("noise_reg", 0.0)),
        epsilon=float(train_args["eps"]),
        dx=float(train_args["dx"]),
        dt=args.dt,
        steps_per_save=args.steps_per_save,
        starting_frame=args.starting_frame,
        num_png=args.num_png,
        num_vtk=args.num_vtk,
        num_npy=args.num_npy,
        num_evo=args.num_evo,
        overwrite=args.overwrite,
    )


def build_transform(img_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((img_size, img_size)),
    ])


def load_state(file_path: Path) -> np.ndarray:
    if file_path.suffix.lower() == ".png":
        with Image.open(file_path) as img:
            return np.array(img.convert("L"))
    if file_path.suffix.lower() == ".npy":
        return np.load(file_path)
    raise ValueError(f"Unsupported extension: {file_path.suffix}")


def ensure_4d(state: torch.Tensor) -> torch.Tensor:
    while state.ndim < 4:
        state = state.unsqueeze(0)
    return state


def load_sequences(
    table_path: Path,
    num_params: int,
    transform: transforms.Compose,
) -> Iterator[tuple[torch.Tensor, torch.Tensor | None, list[str]]]:
    with table_path.open("r") as sequence_file:
        for raw_line in sequence_file:
            parts = raw_line.split()
            if not parts:
                continue

            if num_params > 0:
                snap_paths = parts[:-num_params]
                params = torch.tensor(
                    [[float(val) for val in parts[-num_params:]]],
                    dtype=torch.float32
                )
            else:
                snap_paths = parts
                params = None

            sequence = []
            for snap_path in snap_paths:
                file_path = Path(snap_path)
                state_np = load_state(file_path)
                state = ensure_4d(torch.from_numpy(state_np).float())

                if file_path.suffix.lower() == ".png":
                    state = transform(state)
                elif file_path.suffix.lower() == ".npy":
                    pass
                else:
                    raise ValueError(f"Unsupported extension: {file_path.suffix}")

                sequence.append(state)

            yield torch.stack(sequence, dim=1), params, snap_paths


def prepare_output_folder(output_folder: Path, overwrite: str) -> None:
    if not output_folder.exists():
        output_folder.mkdir(parents=True, exist_ok=True)
        return

    if overwrite == "delete":
        shutil.rmtree(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
    elif overwrite == "continue":
        output_folder.mkdir(parents=True, exist_ok=True)
    else:
        raise FileExistsError(
            f"Output folder {output_folder} already esiste. "
            "Usa --overwrite delete oppure --overwrite continue."
        )


def get_device(use_cuda: bool) -> torch.device:
    if use_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    if use_cuda:
        LOGGER.warning("CUDA richiesta ma non disponibile, uso CPU.")
    return torch.device("cpu")


def build_model(cfg: Config, device: torch.device) -> ConvGRU:
    model = ConvGRU(
        hidden_units=cfg.hidden_units,
        input_channels=cfg.input_channels,
        output_channels=cfg.output_channels,
        hidden_channels=cfg.hidden_channels,
        kernel_size=cfg.kernel_size,
        padding_mode=cfg.padding_mode,
        separable=cfg.separable,
        bias=cfg.bias,
        divergence=cfg.divergence,
        conservative=cfg.conservative,
        num_params=cfg.num_params,
        dropout=cfg.dropout,
        dropout_prob=cfg.dropout_prob,
    )

    model.load_state_dict(torch.load(cfg.model_path, map_location=device))
    model.to(device)
    model.eval()

    if cfg.divergence:
        model.make_div_filters(torch.zeros(1, device=device))

    return model


@njit(fastmath=True)
def grad_2D(phi: np.ndarray, dx: float, grad_x: np.ndarray, grad_y: np.ndarray):
    ny, nx = phi.shape
    dx2_inv = 1.0 / (2.0 * dx)

    for y in range(ny):
        y_up = (y + 1) % ny
        y_down = (y - 1) % ny

        for x in range(nx):
            x_left = (x - 1) % nx
            x_right = (x + 1) % nx

            grad_x[y, x] = (phi[y, x_right] - phi[y, x_left]) * dx2_inv
            grad_y[y, x] = (phi[y_up, x] - phi[y_down, x]) * dx2_inv
            

@njit(fastmath=True)
def grad_2D_neumann_along_y(phi: np.ndarray, dx, grad_x: np.ndarray, grad_y: np.ndarray):
    """
    Calcola il gradiente del campo scalare 2D su griglia uniforme con BC di Neumann lungo y
    e periodicità in y usando schema delle differenze centrate.
    """
    
    ny, nx = phi.shape
    dx2_inv = 1.0 / (2.0 * dx)
    
    for y in range(ny):
        for x in range(nx):
            xl = (x - 1) % nx
            xr = (x + 1) % nx
            
            grad_x[y, x] = (phi[y, xr] - phi[y, xl]) * dx2_inv
            
            # Bordi superiore e inferiore
            if y == 0 or y == ny - 1:
                grad_y[y, x] = 0.0
            # Punti interni
            else:
                grad_y[y, x] = (phi[y+1, x] - phi[y-1, x]) * dx2_inv


@njit(fastmath=True)
def w_field(phi: np.ndarray, epsilon: float, w: np.ndarray):
    ny, nx = phi.shape
    factor = 18.0 / epsilon

    for y in range(ny):
        for x in range(nx):
            phi_ij = phi[y, x]
            w[y, x] = factor * phi_ij * phi_ij * (1.0 - phi_ij) * (1.0 - phi_ij)


@njit(fastmath=True)
def total_free_energy(phi: np.ndarray, epsilon: float, dx: float) -> float:
    ny, nx = phi.shape
    eps2 = 0.5 * epsilon
    dx2 = dx * dx

    w_local = np.empty_like(phi)
    gx = np.empty_like(phi)
    gy = np.empty_like(phi)

    w_field(phi, epsilon, w_local)
    grad_2D_neumann_along_y(phi, dx, gx, gy)

    total_E = 0.0
    for y in range(ny):
        for x in range(nx):
            grad2 = gx[y, x] * gx[y, x] + gy[y, x] * gy[y, x]
            f_ij = w_local[y, x] + eps2 * grad2
            total_E += f_ij

    return total_E * dx2


def compute_timestep_metrics(
    pred_sequence: torch.Tensor,
    true_sequence: torch.Tensor,
    jump: int
) -> tuple[np.ndarray, np.ndarray]:
    timesteps = pred_sequence.shape[1]
    mae = np.full(timesteps, np.nan, dtype=np.float64)
    mse = np.full(timesteps, np.nan, dtype=np.float64)

    pred_np = pred_sequence.detach().cpu().numpy()
    true_np = true_sequence.detach().cpu().numpy()

    for t in range(jump, timesteps):
        diff = pred_np[:, t, ...] - true_np[:, t, ...]
        mae[t] = np.abs(diff).mean()
        mse[t] = np.square(diff).mean()

    return mae, mse


def write_evo_file(
    kk_path: Path,
    sequence: torch.Tensor,
    pred_sequence: torch.Tensor,
    jump: int,
    epsilon: float,
    dx: float,
    dt: float,
    steps_per_save: int,
    starting_frame: int
) -> None:
    sequence_np = sequence.detach().cpu().numpy()
    pred_np = pred_sequence.detach().cpu().numpy()

    evo_path = kk_path / "evo.txt"
    with evo_path.open("w") as file_evo:
        file_evo.write(
            "# 1: time | 2: MAE | 3: MSE | 4: avg_True | 5: avg_Pred | "
            "6: min_True | 7: min_Pred | 8: max_True | 9: max_Pred | "
            "10: E_True | 11: E_Pred "
            "12: mass_True | 13: mass_Pred\n"
        )

        for t in range(jump):
            true = sequence_np[:, t, ...]
            true_2d = true[0, 0, :, :]

            e_true = total_free_energy(true_2d, epsilon, dx)
            mass_true = true_2d.sum(axis=(-1,-2))* dx**2
            time = (t + starting_frame) * dt * steps_per_save

            file_evo.write(
                f"{time}\tnan\tnan\t{true.mean()}\tnan\t"
                f"{true.min()}\tnan\t{true.max()}\tnan\t"
                f"{e_true}\tnan"
                f"{mass_true}\tnan\n"
            )

        for t in range(jump, pred_sequence.shape[1]):
            true = sequence_np[:, t, ...]
            pred = pred_np[:, t, ...]
            diff = pred - true

            true_2d = true[0, 0, :, :]
            pred_2d = pred[0, 0, :, :]

            e_true = total_free_energy(true_2d, epsilon, dx)
            e_pred = total_free_energy(pred_2d, epsilon, dx)
            mass_true = true_2d.sum(axis=(-1,-2))* dx**2
            mass_pred = pred_2d.sum(axis=(-1,-2))* dx**2
            time = (t + starting_frame) * dt * steps_per_save

            file_evo.write(
                f"{time}\t{np.abs(diff).mean()}\t{np.square(diff).mean()}\t"
                f"{true.mean()}\t{pred.mean()}\t"
                f"{true.min()}\t{pred.min()}\t{true.max()}\t{pred.max()}\t"
                f"{e_true}\t{e_pred}\t"
                f"{mass_true}\t{mass_pred}\n"
            )


def save_png_outputs(
    kk_path: Path,
    pred_sequence: torch.Tensor,
    true_sequence: torch.Tensor,
    delta_png: int
) -> None:
    true_dir = kk_path / "true_png"
    pred_dir = kk_path / "pred_png"
    diff_dir = kk_path / "diff_png"

    true_dir.mkdir(exist_ok=True)
    pred_dir.mkdir(exist_ok=True)
    diff_dir.mkdir(exist_ok=True)

    args_true = SimpleNamespace(
        nproc=4,
        cmap="RdBu_r",
        paths={"png": str(true_dir)},
        vmin=0.0,
        vmax=1.0,
    )
    seq2png_treaded(
        true_sequence[:, ::delta_png, ...].cpu(),
        name="snap",
        args=args_true,
        delta=1
    )

    args_pred = SimpleNamespace(
        nproc=4,
        cmap="RdBu_r",
        paths={"png": str(pred_dir)},
        vmin=0.0,
        vmax=1.0,
    )
    seq2png_treaded(
        pred_sequence[:, ::delta_png, ...].cpu(),
        name="snap",
        args=args_pred,
        delta=1
    )

    args_diff = SimpleNamespace(
        nproc=4,
        cmap="bwr",
        clim=[-1.0, 1.0],
        vmin=-1.0,
        vmax=1.0,
        paths={"png": str(diff_dir)},
    )
    seq2png_treaded(
        pred_sequence[:, ::delta_png, ...].cpu() - true_sequence[:, ::delta_png, ...].cpu(),
        name="snap",
        args=args_diff,
        delta=1
    )


def save_optional_outputs(
    cfg: Config,
    kk_path: Path,
    sequence: torch.Tensor,
    pred_sequence: torch.Tensor,
    counters: dict[str, int],
) -> None:
    if counters["npy"] < cfg.num_npy:
        pred_npy_dir = kk_path / "pred_npy"
        pred_npy_dir.mkdir(exist_ok=True)
        seq2npy(pred_sequence.cpu(), path=str(pred_npy_dir), start_snap=cfg.starting_frame)
        counters["npy"] += 1

    if counters["png"] < cfg.num_png:
        save_png_outputs(kk_path, pred_sequence, sequence, cfg.delta_png)
        counters["png"] += 1

    if counters["vtk"] < cfg.num_vtk:
        true_vtk_dir = kk_path / "true_vtk"
        pred_vtk_dir = kk_path / "pred_vtk"
        true_vtk_dir.mkdir(exist_ok=True)
        pred_vtk_dir.mkdir(exist_ok=True)
        seq2vtk(sequence.cpu(), path=str(true_vtk_dir))
        seq2vtk(pred_sequence.cpu(), path=str(pred_vtk_dir))
        counters["vtk"] += 1


def infer_sequence(
    model: ConvGRU,
    sequence: torch.Tensor,
    params: torch.Tensor | None,
    jump: int,
    noise_reg: float,
) -> torch.Tensor:
    input_data = sequence[:, :jump, ...]
    target_data = sequence[:, 1:, ...]
    future = sequence.shape[1] - jump - 1

    pred_target = model(
        input_data,
        future=future,
        params=params,
        noise_reg=noise_reg,
        approx_inference=False,
    )

    return torch.cat([sequence[:, :1, ...], pred_target], dim=1)


def main() -> None:
    cfg = parse_args()
    prepare_output_folder(cfg.output_folder, cfg.overwrite)
    device = get_device(cfg.use_cuda)
    transform = build_transform(cfg.img_size)
    model = build_model(cfg, device)

    counters = {"npy": 0, "evo": 0, "png": 0, "vtk": 0}
    errors_path = cfg.output_folder / "errors.txt"
    model_path_log = cfg.output_folder / "model_path.txt"

    with errors_path.open("w") as file_mae:
        file_mae.write(
            "# 1: id | 2: maxMAE | 3: maxMSE | 4: overallMAE | 5: overallMSE\n"
        )

        with torch.no_grad():
            for kk, (sequence, params, snap_paths) in enumerate(
                load_sequences(cfg.sequence_table, cfg.num_params, transform)
            ):
                seq_name = f"{kk:03d}"
                kk_path = cfg.output_folder / seq_name
                kk_path.mkdir(exist_ok=True)

                true_sequence_dir = Path(snap_paths[0]).parent if snap_paths else None
                symlink_path = kk_path / "true_npy"
                if true_sequence_dir and not symlink_path.exists():
                    symlink_path.symlink_to(true_sequence_dir, target_is_directory=True)

                LOGGER.info("Predicting sequence %s", seq_name)

                sequence = sequence.to(device)
                if params is not None:
                    params = params.to(device)

                pred_sequence = infer_sequence(
                    model,
                    sequence,
                    params,
                    cfg.min_seq,
                    cfg.noise_reg,
                )

                mae, mse = compute_timestep_metrics(
                    pred_sequence, sequence, cfg.min_seq
                )

                file_mae.write(
                    f"{seq_name} {np.nanmax(mae)} {np.nanmax(mse)} "
                    f"{np.nanmean(mae)} {np.nanmean(mse)}\n"
                )

                if counters["evo"] < cfg.num_evo:
                    write_evo_file(
                        kk_path,
                        sequence,
                        pred_sequence,
                        cfg.min_seq,
                        cfg.epsilon,
                        cfg.dx,
                        cfg.dt,
                        cfg.steps_per_save,
                        cfg.starting_frame,
                    )
                    counters["evo"] += 1

                save_optional_outputs(cfg, kk_path, sequence, pred_sequence, counters)

                del pred_sequence
                del sequence

    with model_path_log.open("a") as file_path:
        print(cfg.model_path, file=file_path)

    LOGGER.info("Done.")


if __name__ == "__main__":
    main()