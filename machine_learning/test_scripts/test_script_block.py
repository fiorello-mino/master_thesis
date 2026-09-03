# <<< import external modules <<<
from sys import path
path.append('/home/fiorello/CRANE/')

import os
import shutil
from pathlib import Path
from typing import Union

import numpy as np
import torch
import matplotlib.pyplot as plt

from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
# === import external modules ===


# <<< import CRANE modules <<<
from PersistentModel import PersistentModel
from src.utils import *
# === import CRANE modules ===


# <<< OUTPUT VARIABLES <<<
NUM_PNG     : int = 100
DELTA_PNG   : int = 1
# Do not set DELTA_PNG < 1 if PNG output is enabled.

NUM_NPY     : int = 100
DELTA_NPY   : int = 10
# === OUTPUT VARIABLES ===


# <<< PREDICTION VARIABLES <<<
# PRED_FRAMES == 0:
#   The sequence table contains the complete ground truth.
#   The model predicts from t=MIN_SEQ up to t=len(trueSeq)-1,
#   then computes error metrics at every predicted timestep.
#
# PRED_FRAMES > 0:
#   The sequence table contains exactly MIN_SEQ input frames per sequence.
#   The model generates PRED_FRAMES future frames autoregressively.
#   No ground-truth metrics can be computed.
PRED_FRAMES : int = 0
# === PREDICTION VARIABLES ===


# <<< SCRIPT VARIABLES <<<
LOG_DIR = Path(
    '/home/fiorello/master_thesis/machine_learning/train/train_logs/'
    'lr5e-5_hl3_2_tr10'
)

SEQUENCE_TABLE  : str = '/data/fiorello/testtest/test_set_random.txt'
OUTPUT_FOLDER   : str = '/data/fiorello/test_del_test2'
CUDA            : bool = True
# === SCRIPT VARIABLES ===


# <<< MODEL VARIABLES <<<
MIN_SEQ         : int = 1

HIDDEN_UNITS    : int = 3
INPUT_CHANNELS  : int = 1
OUTPUT_CHANNELS : int = 1
HIDDEN_CHANNELS : int = 16
KERNEL_SIZE     : int = 5
PADDING_MODE    : str = 'circular'

SEPARABLE       : bool = False
BIAS            : bool = True
DIVERGENCE      : bool = True
NUM_PARAMS      : int = 0
DROPOUT         : bool = False
DROPOUT_PROB    : Union[float, None] = None
CONSERVATIVE    : bool = False
# === MODEL VARIABLES ===


class CustomNameSpace:
    """
    Dummy class retained for compatibility with CRANE utilities,
    if they require this interface.
    """
    pass


class OutputMan:
    """
    Handles outputs for one sequence.

    `writeEVO()` supports:
    - Initial frames: `true` is available, `pred=None`.
    - Inference-only prediction: `true=None`, `pred` is available.
    - Evaluation prediction: both `true` and `pred` are available.

    Only frames that have both `true` and `pred` contribute to
    aggregate error statistics.
    """

    def __init__(
        self,
        path: Union[str, Path],
        deltaNPY: int = -1,
        deltaPNG: int = -1,
    ) -> None:
        self.path = Path(path)
        self.deltaNPY = deltaNPY
        self.deltaPNG = deltaPNG

        if self.deltaNPY > 0:
            (self.path / 'pred_npy').mkdir()

        if self.deltaPNG > 0:
            (self.path / 'pred_png').mkdir()
            (self.path / 'diff_png').mkdir()

            self.cmap = LinearSegmentedColormap.from_list(
                'bwr',
                ['blue', 'white', 'red'],
            )

        self.fileEVO = open(self.path / 'evo.txt', 'w')

        self.fileEVO.write(
            '# 1: time | '
            '2: MAE | '
            '3: MSE | '
            '4: mean_True | '
            '5: mean_Pred | '
            '6: min_True | '
            '7: min_Pred | '
            '8: max_True | '
            '9: max_Pred | '
            '10: symdiff\n'
        )

        self.maxMae = 0.0
        self.maxMse = 0.0
        self.sumMae = 0.0
        self.sumMse = 0.0

        self.maxSymDiff = 0.0
        self.sumSymDiff = 0.0

        self.niter_eval = 0

    def close(self) -> None:
        """Explicitly closes evo.txt."""
        if not self.fileEVO.closed:
            self.fileEVO.close()

    def __del__(self) -> None:
        self.close()

    @staticmethod
    def savePNG(fname: Union[str, Path], phi: np.ndarray) -> None:
        """
        Saves a phase field as an 8-bit grayscale PNG after clipping
        its values to the interval [0, 1].
        """
        phiclip = np.clip(phi, 0.0, 1.0)
        pix = (phiclip * 255.0).astype(np.uint8)

        img = Image.fromarray(pix, mode='L')
        img.save(fname)

    def _save_prediction(self, time: int, pred: np.ndarray) -> None:
        """
        Saves prediction outputs according to DELTA_NPY and DELTA_PNG.
        """
        if self.deltaNPY > 0 and time % self.deltaNPY == 0:
            np.save(
                self.path / 'pred_npy' / f'{time:03d}.npy',
                pred,
            )

        if self.deltaPNG > 0 and time % self.deltaPNG == 0:
            self.savePNG(
                self.path / 'pred_png' / f'{time:03d}.png',
                pred,
            )

    def _save_difference(
        self,
        time: int,
        true: np.ndarray,
        pred: np.ndarray,
    ) -> None:
        """
        Saves the clipped signed difference pred - true as a PNG.
        """
        if self.deltaPNG <= 0:
            return

        if time % self.deltaPNG != 0:
            return

        diff = np.clip(pred - true, -1.0, 1.0)

        plt.imsave(
            self.path / 'diff_png' / f'{time:03d}.png',
            diff,
            cmap=self.cmap,
            vmin=-1.0,
            vmax=1.0,
        )

    def writeEVO(
        self,
        time: int,
        true: Union[np.ndarray, None] = None,
        pred: Union[np.ndarray, None] = None,
    ) -> None:
        """
        Writes a single frame entry to evo.txt.

        A prediction is saved only when `pred` exists. Thus, initial
        frames for which `pred=None` can never trigger `np.clip(None, ...)`.
        """
        if true is None and pred is None:
            raise ValueError(
                'writeEVO: both true and pred cannot be None.'
            )

        # Initial/input frame: true exists, pred does not.
        if pred is None:
            self.fileEVO.write(
                f'{time}\t'
                f'NaN\t'
                f'NaN\t'
                f'{true.mean()}\t'
                f'NaN\t'
                f'{true.min()}\t'
                f'NaN\t'
                f'{true.max()}\t'
                f'NaN\t'
                f'NaN\n'
            )
            self.fileEVO.flush()
            return

        # A real prediction exists: save it according to output cadence.
        self._save_prediction(time, pred)

        # Prediction-only mode: no corresponding ground truth.
        if true is None:
            self.fileEVO.write(
                f'{time}\t'
                f'NaN\t'
                f'NaN\t'
                f'NaN\t'
                f'{pred.mean()}\t'
                f'NaN\t'
                f'{pred.min()}\t'
                f'NaN\t'
                f'{pred.max()}\t'
                f'NaN\n'
            )
            self.fileEVO.flush()
            return

        # Ground truth and prediction both exist: compute statistics.
        difference = pred - true

        mae = np.abs(difference).mean()
        mse = (difference ** 2.0).mean()
        symDiff = np.abs(pred.round() - true.round()).mean()

        self.maxMae = max(self.maxMae, mae)
        self.maxMse = max(self.maxMse, mse)
        self.sumMae += mae
        self.sumMse += mse

        self.maxSymDiff = max(self.maxSymDiff, symDiff)
        self.sumSymDiff += symDiff

        self.niter_eval += 1

        self.fileEVO.write(
            f'{time}\t'
            f'{mae}\t'
            f'{mse}\t'
            f'{true.mean()}\t'
            f'{pred.mean()}\t'
            f'{true.min()}\t'
            f'{pred.min()}\t'
            f'{true.max()}\t'
            f'{pred.max()}\t'
            f'{symDiff}\n'
        )

        self._save_difference(time, true, pred)
        self.fileEVO.flush()

    def writeSTAT(self, fileSTAT, seq_name: str) -> None:
        """
        Writes sequence-level aggregate statistics.

        If no ground-truth/prediction pair exists, all metrics are NaN.
        """
        if self.niter_eval == 0:
            fileSTAT.write(
                f'{seq_name} NaN NaN NaN NaN NaN NaN\n'
            )
            fileSTAT.flush()
            return

        avgMae = self.sumMae / self.niter_eval
        avgMse = self.sumMse / self.niter_eval
        avgSymDiff = self.sumSymDiff / self.niter_eval

        fileSTAT.write(
            f'{seq_name} '
            f'{self.maxMae} '
            f'{self.maxMse} '
            f'{avgMae} '
            f'{avgMse} '
            f'{self.maxSymDiff} '
            f'{avgSymDiff}\n'
        )
        fileSTAT.flush()


def best_model_path(log_dir_path: Union[str, Path]) -> Path:
    """
    Returns the checkpoint path associated with the lowest validation loss.
    """
    log_dir_path = Path(log_dir_path)
    valid_loss_file = log_dir_path / 'valid_loss.txt'

    if not log_dir_path.is_dir():
        raise FileNotFoundError(
            f'La cartella di log non esiste: {log_dir_path}'
        )

    if not valid_loss_file.is_file():
        raise FileNotFoundError(
            f'File valid_loss.txt non trovato: {valid_loss_file}'
        )

    min_loss = None
    best_epoch = None

    with valid_loss_file.open('r') as file:
        for epoch, line in enumerate(file):
            line = line.strip()

            if not line:
                continue

            try:
                loss = float(line)
            except ValueError as exc:
                raise ValueError(
                    f'Valore non valido in {valid_loss_file}, '
                    f'riga {epoch + 1}: {line!r}'
                ) from exc

            if min_loss is None or loss < min_loss:
                min_loss = loss
                best_epoch = epoch

    if best_epoch is None:
        raise ValueError(
            f'Il file {valid_loss_file} è vuoto '
            f'o contiene solo righe vuote.'
        )

    checkpoint_path = log_dir_path / 'model' / f'epoch_{best_epoch}.pt'

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f'Checkpoint del best model non trovato: {checkpoint_path}'
        )

    print(
        f'Best model trovato in {log_dir_path}:\n'
        f'epoch={best_epoch}, valid_loss={min_loss}'
    )

    return checkpoint_path


def prepare_output_directory(output_folder: Union[str, Path]) -> None:
    """
    Creates the output directory. If it exists, asks the user whether
    to delete it or abort the program.
    """
    output_folder = Path(output_folder)

    if output_folder.exists():
        overwrite_key = input(
            f'Output folder "{output_folder}" already exists. '
            f'"d" delete, "a" abort - '
            f'"ENTER or other key to continue).\n'
        ).lower()

        if overwrite_key == 'd':
            shutil.rmtree(output_folder)

        elif overwrite_key == 'a':
            raise SystemExit(0)

        else:
            raise FileExistsError(
                f'Output folder già esistente e non eliminata: '
                f'{output_folder}'
            )

    output_folder.mkdir(parents=True, exist_ok=False)


def get_device(use_cuda: bool) -> str:
    """Uses CUDA when requested and available; otherwise returns CPU."""
    if use_cuda and torch.cuda.is_available():
        return 'cuda'

    if use_cuda:
        print('CUDA non disponibile, continuo usando CPU.')

    return 'cpu'


def load_sequence(
    sequence_line: str,
    output_manager: OutputMan,
) -> tuple[torch.Tensor, list[np.ndarray]]:
    """
    Loads a sequence-table row.

    Returns:
    - iniSeq: first MIN_SEQ frames with shape [B, T, C, H, W];
    - trueSeq: every frame listed on the row as NumPy arrays.

    The initial frames are logged with pred=None and do not contribute
    to MAE/MSE/symDiff.
    """
    frame_paths = sequence_line.strip().split()

    if len(frame_paths) < MIN_SEQ:
        raise ValueError(
            f'La sequenza ha {len(frame_paths)} frame, '
            f'ma MIN_SEQ={MIN_SEQ}.'
        )

    iniSeq = []
    trueSeq = []

    for frame_index, frame_path in enumerate(frame_paths):
        phi = np.load(frame_path)
        trueSeq.append(phi)

        if frame_index < MIN_SEQ:
            output_manager.writeEVO(
                time=frame_index,
                true=phi,
                pred=None,
            )

            phi_tensor = torch.from_numpy(phi).float()

            # Example: [H, W] -> [1, 1, H, W].
            while phi_tensor.ndim < 4:
                phi_tensor = phi_tensor.unsqueeze(0)

            iniSeq.append(phi_tensor)

    # [B, C, H, W] frames -> [B, T, C, H, W].
    iniSeq_tensor = torch.stack(iniSeq, dim=1)

    return iniSeq_tensor, trueSeq


def get_last_prediction_time(
    true_seq_length: int,
    has_ground_truth: bool,
) -> int:
    """
    Returns the final time index to predict.

    Evaluation mode:
        PRED_FRAMES == 0.
        The table contains t=0, ..., t=len(trueSeq)-1.
        First MIN_SEQ frames are inputs, hence prediction ends at
        t=len(trueSeq)-1.

    Inference-only mode:
        PRED_FRAMES > 0.
        Generates exactly PRED_FRAMES future frames after the input.

        With MIN_SEQ=1 and PRED_FRAMES=200:
        - Input: t=0.
        - Predictions: t=1, ..., 200.
    """
    if has_ground_truth:
        return true_seq_length - 1

    return MIN_SEQ + PRED_FRAMES - 1


def main() -> None:
    """
    Autoregressive one-frame-at-a-time rollout.

    Modes:
    - PRED_FRAMES == 0: full ground truth; predict and evaluate all
      timesteps after the initial MIN_SEQ input frames.
    - PRED_FRAMES > 0: input-only table; generate exactly PRED_FRAMES
      future frames without evaluation metrics.
    """
    if PRED_FRAMES < 0:
        raise ValueError(
            'PRED_FRAMES deve essere 0 oppure un intero positivo.'
        )

    has_ground_truth = (PRED_FRAMES == 0)

    prepare_output_directory(OUTPUT_FOLDER)
    device = get_device(CUDA)

    model = PersistentModel(
        hidden_units=HIDDEN_UNITS,
        input_channels=INPUT_CHANNELS,
        output_channels=OUTPUT_CHANNELS,
        hidden_channels=HIDDEN_CHANNELS,
        kernel_size=KERNEL_SIZE,
        padding_mode=PADDING_MODE,
        separable=SEPARABLE,
        bias=BIAS,
        divergence=DIVERGENCE,
        num_params=NUM_PARAMS,
        dropout=DROPOUT,
        dropout_prob=DROPOUT_PROB,
        conservative=CONSERVATIVE,
    )

    model_path = best_model_path(LOG_DIR)

    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(checkpoint)
    model.eval()
    model.to(device)

    # This preserves the method name from your original code.
    model.make_div_filters(torch.zeros(1, device=device))

    with open(SEQUENCE_TABLE, 'r') as intable:
        listseq = [
            line.strip()
            for line in intable.readlines()
            if line.strip()
        ]

    if not listseq:
        raise ValueError(
            f'La sequence table è vuota: {SEQUENCE_TABLE}'
        )

    first_num_frames = len(listseq[0].split())

    if has_ground_truth:
        if first_num_frames <= MIN_SEQ:
            raise ValueError(
                'PRED_FRAMES=0 richiede il ground truth completo, ma la '
                f'prima sequenza contiene {first_num_frames} frame con '
                f'MIN_SEQ={MIN_SEQ}.'
            )

        print(
            'Modalità evaluation: PRED_FRAMES=0. '
            'Predizione fino alla lunghezza completa di trueSeq.'
        )

    else:
        if first_num_frames != MIN_SEQ:
            raise ValueError(
                f'PRED_FRAMES={PRED_FRAMES} richiede esattamente '
                f'MIN_SEQ={MIN_SEQ} frame per riga, ma la prima sequenza '
                f'ne contiene {first_num_frames}.'
            )

        print(
            f'Modalità inference-only: {PRED_FRAMES} frame futuri '
            'autoregressivi senza ground truth.'
        )

    countNPYout = 0
    countPNGout = 0

    with open(f'{OUTPUT_FOLDER}/errors.txt', 'w') as fileSTAT:
        fileSTAT.write(
            '# 1: id | '
            '2: maxMAE | '
            '3: maxMSE | '
            '4: overallMAE | '
            '5: overallMSE | '
            '6: max(symDiff) | '
            '7: avg(symDiff)\n'
        )

        with torch.no_grad():
            for seq_index, seq in enumerate(listseq):
                frame_paths = seq.split()

                if has_ground_truth and len(frame_paths) <= MIN_SEQ:
                    raise ValueError(
                        f'Sequenza {seq_index}: ground truth insufficiente. '
                        f'Frame trovati={len(frame_paths)}, '
                        f'MIN_SEQ={MIN_SEQ}.'
                    )

                if not has_ground_truth and len(frame_paths) != MIN_SEQ:
                    raise ValueError(
                        f'Sequenza {seq_index}: in inference-only mode sono '
                        f'attesi esattamente MIN_SEQ={MIN_SEQ} frame, ma ne '
                        f'sono presenti {len(frame_paths)}.'
                    )

                seq_name = Path(frame_paths[0]).parent.name
                seq_path = Path(OUTPUT_FOLDER) / seq_name

                if seq_path.exists():
                    raise FileExistsError(
                        f'La cartella della sequenza esiste già: {seq_path}'
                    )

                seq_path.mkdir()

                original_data_dir = Path(frame_paths[0]).parent

                os.symlink(
                    original_data_dir,
                    seq_path / 'true_npy',
                    target_is_directory=True,
                )

                model.zero_grad(set_to_none=True)

                dNPY = DELTA_NPY if countNPYout < NUM_NPY else -1
                dPNG = DELTA_PNG if countPNGout < NUM_PNG else -1

                countNPYout += 1
                countPNGout += 1

                out = OutputMan(
                    path=seq_path,
                    deltaNPY=dNPY,
                    deltaPNG=dPNG,
                )

                iniSeq, trueSeq = load_sequence(
                    sequence_line=seq,
                    output_manager=out,
                )

                iniSeq = iniSeq.to(device)

                # Initializes the persistent state from the first input.
                model.set_hidden(iniSeq[:, 0:1, ...])

                params = None
                if params is not None:
                    params = params.to(device)

                last_time = get_last_prediction_time(
                    true_seq_length=len(trueSeq),
                    has_ground_truth=has_ground_truth,
                )

                print(
                    f'Predicting sequence {seq_name} '
                    f'(t={MIN_SEQ} ... t={last_time})...'
                )

                # This should only happen for malformed or degenerate input.
                if last_time < MIN_SEQ:
                    out.writeSTAT(fileSTAT, seq_name)
                    out.close()
                    continue

                # First model-generated future frame: t=MIN_SEQ.
                predSeq = model(
                    iniSeq,
                    future=0,
                    params=params,
                    approx_inference=False,
                )

                predSeq = predSeq[:, MIN_SEQ - 1:MIN_SEQ, ...]

                time = MIN_SEQ
                pred_frame = predSeq[0, 0, 0, ...].cpu().numpy()

                if has_ground_truth:
                    out.writeEVO(
                        time=time,
                        true=trueSeq[time],
                        pred=pred_frame,
                    )
                else:
                    out.writeEVO(
                        time=time,
                        true=None,
                        pred=pred_frame,
                    )

                # Every iteration predicts exactly one new frame.
                while time < last_time:
                    if time % 50 == 0:
                        print(time, end='...', flush=True)

                    predSeq = model(
                        predSeq,
                        future=0,
                        params=params,
                        approx_inference=False,
                    )

                    time += 1
                    pred_frame = predSeq[0, 0, 0, ...].cpu().numpy()

                    if has_ground_truth:
                        out.writeEVO(
                            time=time,
                            true=trueSeq[time],
                            pred=pred_frame,
                        )
                    else:
                        out.writeEVO(
                            time=time,
                            true=None,
                            pred=pred_frame,
                        )

                out.writeSTAT(fileSTAT, seq_name)
                out.close()

                print('DONE!')
                print()
                print('=' * 30)

                del iniSeq
                del trueSeq
                del predSeq

                if device == 'cuda':
                    torch.cuda.empty_cache()

    with open(f'{OUTPUT_FOLDER}/model_path.txt', 'w') as file_path:
        file_path.write(f'{model_path}\n')


if __name__ == '__main__':
    main()