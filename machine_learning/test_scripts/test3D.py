# <<< import external modules <<<
from sys import path
path.append('/home/fiorello/CRANE/')

import os
import shutil
from pathlib import Path
from typing import Union

import numpy as np
import torch
# === import external modules ===


# <<< import CRANE modules <<<
from PersistentModel import PersistentModel
from src.utils import *
# === import CRANE modules ===


# <<< OUTPUT VARIABLES <<<
# Salva NPY solo per le prime NUM_NPY sequenze.
NUM_NPY: int = 100
DELTA_NPY: int = 10

# Salva VTK solo per le prime NUM_VTK sequenze.
NUM_VTK: int = 100
DELTA_VTK: int = 1

# Nome del campo scalare visibile in ParaView.
VTK_FIELD_NAME: str = 'phi'
# === OUTPUT VARIABLES ===


# <<< PREDICTION VARIABLES <<<
# PRED_FRAMES == 0:
#   sequence_table con ground truth completo.
#   Il modello predice da t=MIN_SEQ a t=len(trueSeq)-1 e calcola
#   le metriche confrontando predizione e target a ogni tempo.
#
# PRED_FRAMES > 0:
#   sequence_table con esattamente MIN_SEQ frame iniziali per riga.
#   Il modello genera PRED_FRAMES frame futuri autoregressivamente.
#   Non vengono calcolate metriche, perché non esiste ground truth.
PRED_FRAMES: int = 1
# === PREDICTION VARIABLES ===


# <<< SCRIPT VARIABLES <<<
LOG_DIR = Path(
    '/home/fiorello/master_thesis/machine_learning/train3D/'
    'train_logs/prova3D_1'
)

SEQUENCE_TABLE: str = '/data/fiorello/pores3D/ext_test/conformal/test_set.txt'
OUTPUT_FOLDER: str = '/data/fiorello/pores3D/ext_test/conformal/mse'
CUDA: bool = True
# === SCRIPT VARIABLES ===


# <<< MODEL VARIABLES <<<
MIN_SEQ: int = 1

HIDDEN_UNITS: int = 3
INPUT_CHANNELS: int = 1
OUTPUT_CHANNELS: int = 1
HIDDEN_CHANNELS: int = 16
KERNEL_SIZE: int = 5
PADDING_MODE: str = 'reflect'

SEPARABLE: bool = False
BIAS: bool = True
DIVERGENCE: bool = True
NUM_PARAMS: int = 0
DROPOUT: bool = False
DROPOUT_PROB: Union[float, None] = None
CONSERVATIVE: bool = False
# === MODEL VARIABLES ===


class CustomNameSpace:
    """
    Dummy class maintained for compatibility with CRANE utilities.
    """
    pass


class OutputMan:
    """
    Gestisce gli output per una singola sequenza 3D.

    Convenzione di ogni volume:
        phi.shape == (nx, ny, nz)

    Output possibili:
    - evo.txt: statistiche frame per frame;
    - pred_npy/: predizioni in formato NumPy;
    - pred_vtk/: predizioni 3D in legacy ASCII VTK.

    writeEVO gestisce:
    - true disponibile e pred=None: frame iniziale, non predetto;
    - true=None e pred disponibile: rollout senza ground truth;
    - true e pred disponibili: rollout valutato con metriche.
    """

    def __init__(
        self,
        path: Union[str, Path],
        deltaNPY: int = -1,
        deltaVTK: int = -1,
        vtk_field_name: str = VTK_FIELD_NAME,
    ) -> None:
        self.path = Path(path)
        self.deltaNPY = deltaNPY
        self.deltaVTK = deltaVTK
        self.vtk_field_name = vtk_field_name

        if self.deltaNPY > 0:
            (self.path / 'pred_npy').mkdir()

        if self.deltaVTK > 0:
            (self.path / 'pred_vtk').mkdir()

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

        # Numero di frame su cui sono state effettivamente calcolate metriche.
        self.niter_eval = 0

    def close(self) -> None:
        """Chiude esplicitamente evo.txt."""
        if not self.fileEVO.closed:
            self.fileEVO.close()

    def __del__(self) -> None:
        self.close()

    @staticmethod
    def _as_3d_array(phi: np.ndarray) -> np.ndarray:
        """
        Converte phi in un ndarray 3D e controlla la convenzione attesa.

        Input atteso:
            phi.shape == (nx, ny, nz)

        Eventuali dimensioni singleton vengono eliminate con squeeze.
        """
        phi = np.asarray(phi)
        phi = np.squeeze(phi)

        if phi.ndim != 3:
            raise ValueError(
                'L output VTK richiede un volume 3D con shape '
                f'(nx, ny, nz), ma è stata ricevuta shape {phi.shape}.'
            )

        return phi

    @staticmethod
    def saveVTK(
        fname: Union[str, Path],
        phi: np.ndarray,
        field_name: str = 'phi',
    ) -> None:
        """
        Scrive un volume 3D in legacy ASCII VTK.

        Convenzione dell'array di ingresso:
            phi[ix, iy, iz]
            phi.shape == (nx, ny, nz)

        Il file VTK dichiara:
            DIMENSIONS nx ny nz

        VTK richiede nel buffer lineare che x vari più rapidamente, poi y,
        poi z. Perciò l'array viene trasformato:

            (nx, ny, nz) -> (nz, ny, nx)

        e poi appiattito in C-order.
        """
        phi = OutputMan._as_3d_array(phi)
        nx, ny, nz = phi.shape

        # phi_vtk[iz, iy, ix] = phi[ix, iy, iz]
        # In C-order, ix è l'indice che varia più rapidamente.
        phi_vtk = np.transpose(phi, (2, 1, 0)).astype(
            np.float32,
            copy=False,
        )

        with open(fname, 'w') as vtk_file:
            vtk_file.write('# vtk DataFile Version 3.0\n')
            vtk_file.write('CRANE predicted scalar field\n')
            vtk_file.write('ASCII\n')
            vtk_file.write('DATASET STRUCTURED_POINTS\n')
            vtk_file.write(f'DIMENSIONS {nx} {ny} {nz}\n')
            vtk_file.write('ORIGIN 0.0 0.0 0.0\n')
            vtk_file.write('SPACING 1.0 1.0 1.0\n')
            vtk_file.write(f'POINT_DATA {nx * ny * nz}\n')
            vtk_file.write(f'SCALARS {field_name} float 1\n')
            vtk_file.write('LOOKUP_TABLE default\n')

            np.savetxt(
                vtk_file,
                phi_vtk.ravel(order='C'),
                fmt='%.8e',
            )

    def _save_prediction(self, time: int, pred: np.ndarray) -> None:
        """
        Salva le predizioni nei formati richiesti.

        I file vengono creati solo ai tempi compatibili con i rispettivi
        delta di output.
        """
        if self.deltaNPY > 0 and time % self.deltaNPY == 0:
            np.save(
                self.path / 'pred_npy' / f'{time:03d}.npy',
                pred,
            )

        if self.deltaVTK > 0 and time % self.deltaVTK == 0:
            self.saveVTK(
                self.path / 'pred_vtk' / f'{time:03d}.vtk',
                pred,
                field_name=self.vtk_field_name,
            )

    def writeEVO(
        self,
        time: int,
        true: Union[np.ndarray, None] = None,
        pred: Union[np.ndarray, None] = None,
    ) -> None:
        """
        Scrive una riga in evo.txt e salva una predizione, quando presente.

        - I frame iniziali sono loggati con pred=None e non generano file VTK.
        - Le predizioni senza truth hanno metriche NaN.
        - Le predizioni con truth contribuiscono alle statistiche aggregate.
        """
        if true is None and pred is None:
            raise ValueError(
                'writeEVO: true e pred non possono essere entrambi None.'
            )

        # Frame iniziale noto, ma non generato dalla rete.
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

        # Predizione valida: salva NPY/VTK indipendentemente dalla truth.
        self._save_prediction(time, pred)

        # Rollout puro: nessun frame vero da usare per la valutazione.
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

        # Ground truth e predizione disponibili: calcola le metriche.
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

        self.fileEVO.flush()

    def writeSTAT(self, fileSTAT, seq_name: str) -> None:
        """
        Scrive le metriche aggregate di una sequenza in errors.txt.

        Se la simulazione viene fatta in inference-only mode, non esistono
        coppie true/pred e vengono scritti NaN.
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
    Restituisce il checkpoint con la validation loss minima.
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
    Crea la cartella di output. Se esiste, chiede se eliminarla.
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
    """Restituisce CUDA se richiesta e disponibile; altrimenti CPU."""
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
    Carica tutti i frame elencati in una riga della sequence table.

    Ritorna:
    - iniSeq: i primi MIN_SEQ frame con shape [B, T, C, nx, ny, nz];
    - trueSeq: tutti i volumi presenti nella riga, come array NumPy.

    Convenzione dei volumi:
        phi.shape == (nx, ny, nz)
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

            # Da [nx, ny, nz] a [B=1, C=1, nx, ny, nz].
            while phi_tensor.ndim < 5:
                phi_tensor = phi_tensor.unsqueeze(0)

            iniSeq.append(phi_tensor)

    # Lista di [B, C, nx, ny, nz] -> [B, T, C, nx, ny, nz].
    iniSeq_tensor = torch.stack(iniSeq, dim=1)

    return iniSeq_tensor, trueSeq


def get_last_prediction_time(
    true_seq_length: int,
    has_ground_truth: bool,
) -> int:
    """
    Restituisce l'ultimo indice temporale da predire.

    Evaluation mode, PRED_FRAMES=0:
        frame disponibili: 0, ..., len(trueSeq)-1
        frame input: 0, ..., MIN_SEQ-1
        ultimo frame predetto: len(trueSeq)-1

    Inference-only mode, PRED_FRAMES>0:
        vengono generati esattamente PRED_FRAMES frame dopo i MIN_SEQ input.

    Esempio con MIN_SEQ=1 e PRED_FRAMES=200:
        input: t=0
        predizioni: t=1, ..., 200
    """
    if has_ground_truth:
        return true_seq_length - 1

    return MIN_SEQ + PRED_FRAMES - 1


def main() -> None:
    """
    Esegue un rollout autoregressivo 3D, un frame per chiamata al modello.

    PRED_FRAMES == 0:
        la table contiene il ground truth completo e vengono calcolate
        metriche su tutti i frame da t=MIN_SEQ a t=len(trueSeq)-1.

    PRED_FRAMES > 0:
        la table contiene solo MIN_SEQ frame iniziali per sequenza e vengono
        generati esattamente PRED_FRAMES frame futuri senza metriche.
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

    # Mantiene il nome della funzione del tuo codice CRANE originale.
    model.make_div_filters(torch.zeros(1, device=device))

    with open(SEQUENCE_TABLE, 'r') as intable:
        listseq = [
            line.strip()
            for line in intable
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
    countVTKout = 0

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
                dVTK = DELTA_VTK if countVTKout < NUM_VTK else -1

                countNPYout += 1
                countVTKout += 1

                out = OutputMan(
                    path=seq_path,
                    deltaNPY=dNPY,
                    deltaVTK=dVTK,
                    vtk_field_name=VTK_FIELD_NAME,
                )

                iniSeq, trueSeq = load_sequence(
                    sequence_line=seq,
                    output_manager=out,
                )

                iniSeq = iniSeq.to(device)

                # Inizializza lo stato persistente usando il primo frame input.
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

                if last_time < MIN_SEQ:
                    out.writeSTAT(fileSTAT, seq_name)
                    out.close()
                    continue

                # Prima predizione dopo i MIN_SEQ frame iniziali.
                predSeq = model(
                    iniSeq,
                    future=0,
                    params=params,
                    approx_inference=False,
                )

                # Conserva unicamente il frame da usare come input
                # al prossimo passo autoregressivo.
                predSeq = predSeq[:, MIN_SEQ - 1:MIN_SEQ, ...]

                time = MIN_SEQ
                pred_frame = predSeq[0, 0, 0, ...].cpu().numpy()

                out.writeEVO(
                    time=time,
                    true=trueSeq[time] if has_ground_truth else None,
                    pred=pred_frame,
                )

                # Ogni iterazione genera esattamente un nuovo volume 3D.
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

                    out.writeEVO(
                        time=time,
                        true=trueSeq[time] if has_ground_truth else None,
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