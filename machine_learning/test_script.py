from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path("/home/fiorello/CRANE_bc")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import ast
import logging
import shutil
from dataclasses import dataclass
from typing import Iterator

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from src.classes import ConvGRU


TRAIN_LOGS_ROOT = Path("/home/machine_learning/train_pores/train_logs")
LOGGER = logging.getLogger("test_script")


@dataclass
class Config:
    log_dir_name: str
    sequence_table: Path
    output_folder: Path

    train_logs_root: Path = TRAIN_LOGS_ROOT
    model_path: Path | None = None
    valid_loss_path: Path | None = None
    args_txt_path: Path | None = None

    img_size: int = 128
    use_cuda: bool = True

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
    noise_reg: float = 0.0125
    num_params: int = 0
    dropout: bool = False
    dropout_prob: float | None = 0.25

    epsilon: float = 0.024739583333333336
    dx: float = 0.014960629921259842
    dt: float = 1.0e-6
    steps_per_save: int = 100_000
    starting_frame: int = 10

    num_png: int = 100
    num_npy: int = 0
    num_evo: int = 25_000

    overwrite: str = "abort"


def positive_int(value: str) -> int:
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"{value} is not a positive integer")
    return ivalue


def non_negative_int(value: str) -> int:
    ivalue = int(value)
    if ivalue < 0:
        raise argparse.ArgumentTypeError(f"{value} is not a non-negative integer")
    return ivalue


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Test a trained ConvGRU model by automatically selecting the best checkpoint from train_logs."
    )

    parser.add_argument("--log-dir-name", required=True, type=str)
    parser.add_argument("--sequence-table", required=True, type=Path)
    parser.add_argument("--output-folder", required=True, type=Path)

    parser.add_argument("--img-size", type=positive_int, default=None)
    parser.add_argument("--cpu", dest="use_cuda", action="store_false")
    parser.set_defaults(use_cuda=True)

    parser.add_argument("--min-seq", type=positive_int, default=None)

    parser.add_argument("--num-png", type=non_negative_int, default=100)
    parser.add_argument("--num-npy", type=non_negative_int, default=0)
    parser.add_argument("--num-evo", type=non_negative_int, default=25_000)

    parser.add_argument("--dt", type=float, default=1.0e-6)
    parser.add_argument("--steps-per-save", type=positive_int, default=100_000)
    parser.add_argument("--starting-frame", type=non_negative_int, default=10)

    parser.add_argument("--overwrite", choices=("abort", "replace"), default="abort")

    args = parser.parse_args()

    sequence_table = args.sequence_table.expanduser().resolve()
    output_folder = args.output_folder.expanduser().resolve()

    if not sequence_table.is_file():
        raise FileNotFoundError(f"Sequence table not found: {sequence_table}")

    cfg = Config(
        log_dir_name=args.log_dir_name,
        sequence_table=sequence_table,
        output_folder=output_folder,
        use_cuda=args.use_cuda,
        num_png=args.num_png,
        num_npy=args.num_npy,
        num_evo=args.num_evo,
        dt=args.dt,
        steps_per_save=args.steps_per_save,
        starting_frame=args.starting_frame,
        overwrite=args.overwrite,
    )

    if args.img_size is not None:
        cfg.img_size = args.img_size
    if args.min_seq is not None:
        cfg.min_seq = args.min_seq

    return cfg


def prepare_output_dir(output_folder: Path, overwrite: str) -> None:
    if output_folder.exists():
        if overwrite == "replace":
            shutil.rmtree(output_folder)
        else:
            raise FileExistsError(
                f"Output folder already exists: {output_folder}. "
                f"Use --overwrite replace to overwrite it."
            )

    output_folder.mkdir(parents=True, exist_ok=True)


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


def find_best_epoch(valid_loss_path: Path) -> tuple[int, float]:
    losses = []

    with valid_loss_path.open("r") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            losses.append((idx, float(line)))

    if not losses:
        raise RuntimeError(f"No valid losses found in {valid_loss_path}")

    best_epoch, best_loss = min(losses, key=lambda x: x[1])
    return best_epoch, best_loss


def hydrate_config_from_train_logs(cfg: Config) -> Config:
    log_dir = cfg.train_logs_root / cfg.log_dir_name
    if not log_dir.is_dir():
        raise FileNotFoundError(f"Training log dir not found: {log_dir}")

    valid_loss_path = log_dir / "valid_loss.txt"
    args_txt_path = log_dir / "args.txt"
    model_dir = log_dir / "model"

    if not valid_loss_path.is_file():
        raise FileNotFoundError(f"valid_loss.txt not found: {valid_loss_path}")
    if not args_txt_path.is_file():
        raise FileNotFoundError(f"args.txt not found: {args_txt_path}")
    if not model_dir.is_dir():
        raise FileNotFoundError(f"model dir not found: {model_dir}")

    train_args = load_train_args(args_txt_path)
    best_epoch, best_loss = find_best_epoch(valid_loss_path)
    model_path = model_dir / f"epoch_{best_epoch}.pt"

    if not model_path.is_file():
        raise FileNotFoundError(f"Best checkpoint not found: {model_path}")

    cfg.valid_loss_path = valid_loss_path
    cfg.args_txt_path = args_txt_path
    cfg.model_path = model_path

    cfg.img_size = int(train_args["size"]) if cfg.img_size == 128 else cfg.img_size
    cfg.min_seq = int(train_args["subseq_min"]) if cfg.min_seq == 1 else cfg.min_seq

    cfg.hidden_units = int(train_args["hidden"])
    cfg.input_channels = 1
    cfg.output_channels = None
    cfg.hidden_channels = int(train_args["channels"])
    cfg.kernel_size = int(train_args["kernel_size"])

    padding = train_args["padding"]
    if isinstance(padding, list):
        cfg.padding_mode = tuple(padding)
    else:
        cfg.padding_mode = padding

    cfg.separable = False
    cfg.bias = bool(train_args["bias"])
    cfg.divergence = bool(train_args["divergence"])
    cfg.conservative = bool(train_args["conservative"])
    cfg.noise_reg = float(train_args["noise_reg"])
    cfg.num_params = int(train_args["num_params"])
    cfg.dropout = bool(train_args["dropout"])
    cfg.dropout_prob = train_args["dropout_prob"]
    cfg.epsilon = float(train_args["eps"])
    cfg.dx = float(train_args["dx"])

    LOGGER.info("Training log dir: %s", log_dir)
    LOGGER.info("Selected best epoch %d with valid loss %.8e", best_epoch, best_loss)
    LOGGER.info("Resolved checkpoint: %s", model_path)

    return cfg


def get_device(use_cuda: bool) -> torch.device:
    if use_cuda and torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def build_transform(img_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((img_size, img_size)),
        ]
    )


def load_png(path: Path) -> np.ndarray:
    return np.array(Image.open(path))


def load_npy(path: Path) -> np.ndarray:
    return np.load(path)


def load_state(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".png":
        return load_png(path)
    if suffix == ".npy":
        return load_npy(path)
    raise ValueError(f"Unsupported file format: {path.suffix}")


def ensure_4d(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 2:
        return tensor.unsqueeze(0).unsqueeze(0)
    if tensor.ndim == 3:
        return tensor.unsqueeze(0)
    if tensor.ndim == 4:
        return tensor
    raise ValueError(f"Unsupported tensor shape: {tuple(tensor.shape)}")


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
                    dtype=torch.float32,
                )
            else:
                snap_paths = parts
                params = None

            sequence = []

            for snap_path in snap_paths:
                path = Path(snap_path)
                state_np = load_state(path)
                state = ensure_4d(torch.from_numpy(state_np).float())

                if path.suffix.lower() == ".png":
                    state = transform(state)
                elif path.suffix.lower() == ".npy":
                    pass
                else:
                    raise ValueError(f"Unsupported extension: {path.suffix}")

                sequence.append(state)

            yield torch.stack(sequence, dim=1), params, snap_paths


def build_model(cfg: Config, device: torch.device) -> ConvGRU:
    if cfg.model_path is None:
        raise RuntimeError("model_path is not resolved")

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

    state_dict = torch.load(cfg.model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    if cfg.divergence:
        model.make_div_filters(torch.zeros(1, device=device))

    return model


def infer_sequence(
    model: ConvGRU,
    sequence: torch.Tensor,
    params: torch.Tensor | None,
    jump: int,
    noise_reg: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    input_data = sequence[:, :jump, ...]
    target_data = sequence[:, 1:, ...]
    future = sequence.shape[1] - jump - 1

    if future < 0:
        raise ValueError(
            f"Sequence too short for jump={jump}. Sequence length is {sequence.shape[1]}"
        )

    with torch.no_grad():
        y_pred = model(
            input_data,
            future=future,
            params=params,
            noise_reg=noise_reg,
            approx_inference=False,
        )

    pred_full = torch.cat([sequence[:, :1, ...], y_pred], dim=1)
    return y_pred, target_data, pred_full


def compute_timestep_metrics(
    pred_sequence: torch.Tensor,
    true_sequence: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    timesteps = pred_sequence.shape[1]
    mae = np.empty(timesteps, dtype=np.float64)
    mse = np.empty(timesteps, dtype=np.float64)

    pred_np = pred_sequence.detach().cpu().numpy()
    true_np = true_sequence.detach().cpu().numpy()

    for t in range(timesteps):
        diff = pred_np[:, t, ...] - true_np[:, t, ...]
        mae[t] = np.abs(diff).mean()
        mse[t] = np.square(diff).mean()

    return mae, mse


def compute_e(c: torch.Tensor, eps: float, dx: float) -> torch.Tensor:
    w = (18.0 / eps) * c**2 * (1.0 - c)**2
    gx = (c.roll(-1, dims=-1) - c.roll(1, dims=-1)) / (2.0 * dx)
    gy = (c.roll(-1, dims=-2) - c.roll(1, dims=-2)) / (2.0 * dx)
    grad_sq = gx**2 + gy**2
    f = 0.5 * eps * grad_sq + w
    return f.sum(dim=(-1, -2)) * dx**2


def write_evo_file(
    output_path: Path,
    pred_full: torch.Tensor,
    true_full: torch.Tensor,
    epsilon: float,
    dx: float,
    dt: float,
    steps_per_save: int,
    starting_frame: int,
) -> None:
    pred_cpu = pred_full.detach().cpu()
    true_cpu = true_full.detach().cpu()

    with output_path.open("w") as file:
        file.write("frame\ttime\tmse\tmae\tE_pred\tE_true\n")

        for t in range(pred_cpu.shape[1]):
            pred_t = pred_cpu[:, t, ...]
            true_t = true_cpu[:, t, ...]

            diff = pred_t - true_t
            mse = torch.mean(diff**2).item()
            mae = torch.mean(torch.abs(diff)).item()

            e_pred = torch.mean(compute_e(pred_t.squeeze(2), epsilon, dx)).item()
            e_true = torch.mean(compute_e(true_t.squeeze(2), epsilon, dx)).item()

            physical_frame = starting_frame + t
            physical_time = physical_frame * steps_per_save * dt

            file.write(
                f"{physical_frame}\t{physical_time:.8e}\t{mse:.8e}\t{mae:.8e}\t"
                f"{e_pred:.8e}\t{e_true:.8e}\n"
            )


def save_frame_png(tensor_2d: torch.Tensor, path: Path) -> None:
    array = tensor_2d.detach().cpu().numpy()
    array = np.clip(array, 0.0, 1.0)
    image = (255.0 * array).astype(np.uint8)
    Image.fromarray(image).save(path)


def save_png_samples(
    pred_full: torch.Tensor,
    true_full: torch.Tensor,
    output_dir: Path,
    max_frames: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    pred_cpu = pred_full.detach().cpu()
    true_cpu = true_full.detach().cpu()

    total_frames = pred_cpu.shape[1]
    num_frames = min(total_frames, max_frames)

    for t in range(num_frames):
        pred_frame = pred_cpu[0, t, 0, :, :]
        true_frame = true_cpu[0, t, 0, :, :]

        save_frame_png(pred_frame, output_dir / f"pred_{t:04d}.png")
        save_frame_png(true_frame, output_dir / f"true_{t:04d}.png")


def save_npy_samples(
    pred_full: torch.Tensor,
    true_full: torch.Tensor,
    output_dir: Path,
    max_frames: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    pred_cpu = pred_full.detach().cpu().numpy()
    true_cpu = true_full.detach().cpu().numpy()

    total_frames = pred_cpu.shape[1]
    num_frames = min(total_frames, max_frames)

    for t in range(num_frames):
        np.save(output_dir / f"pred_{t:04d}.npy", pred_cpu[0, t, 0, :, :])
        np.save(output_dir / f"true_{t:04d}.npy", true_cpu[0, t, 0, :, :])


def main() -> None:
    configure_logging()
    cfg = parse_args()
    cfg = hydrate_config_from_train_logs(cfg)
    prepare_output_dir(cfg.output_folder, cfg.overwrite)

    device = get_device(cfg.use_cuda)
    LOGGER.info("Using device: %s", device)

    transform = build_transform(cfg.img_size)
    model = build_model(cfg, device)

    summary_rows = []

    for seq_idx, (sequence, params, snap_paths) in enumerate(
        load_sequences(cfg.sequence_table, cfg.num_params, transform)
    ):
        LOGGER.info("Processing sequence %d", seq_idx)

        sequence = sequence.to(device)
        if params is not None:
            params = params.to(device)

        LOGGER.info("Sequence shape: %s", tuple(sequence.shape))

        y_pred, target_data, pred_full = infer_sequence(
            model=model,
            sequence=sequence,
            params=params,
            jump=cfg.min_seq,
            noise_reg=cfg.noise_reg,
        )

        mae, mse = compute_timestep_metrics(y_pred, target_data)

        seq_dir = cfg.output_folder / f"sequence_{seq_idx:04d}"
        png_dir = seq_dir / "png"
        npy_dir = seq_dir / "npy"
        txt_dir = seq_dir / "txt"
        txt_dir.mkdir(parents=True, exist_ok=True)

        metrics_path = txt_dir / "pred_vs_true_metrics.txt"
        with metrics_path.open("w") as f:
            f.write("relative_timestep\tmae\tmse\n")
            for t in range(len(mae)):
                f.write(f"{t+1}\t{mae[t]:.8e}\t{mse[t]:.8e}\n")

        write_evo_file(
            output_path=txt_dir / "evolution.txt",
            pred_full=pred_full,
            true_full=sequence,
            epsilon=cfg.epsilon,
            dx=cfg.dx,
            dt=cfg.dt,
            steps_per_save=cfg.steps_per_save,
            starting_frame=cfg.starting_frame,
        )

        if cfg.num_png > 0:
            save_png_samples(
                pred_full=pred_full,
                true_full=sequence,
                output_dir=png_dir,
                max_frames=cfg.num_png,
            )

        if cfg.num_npy > 0:
            save_npy_samples(
                pred_full=pred_full,
                true_full=sequence,
                output_dir=npy_dir,
                max_frames=cfg.num_npy,
            )

        summary_rows.append(
            (
                seq_idx,
                float(np.mean(mae)),
                float(np.mean(mse)),
                len(snap_paths),
            )
        )

    summary_path = cfg.output_folder / "summary.txt"
    with summary_path.open("w") as f:
        f.write("sequence_idx\tmean_mae\tmean_mse\tlength\n")
        for seq_idx, mean_mae, mean_mse, seq_len in summary_rows:
            f.write(f"{seq_idx}\t{mean_mae:.8e}\t{mean_mse:.8e}\t{seq_len}\n")

    LOGGER.info("Done. Results saved in %s", cfg.output_folder)


if __name__ == "__main__":
    main()