# <<< import external modules <<<
from sys import path
path.append('/home/fiorello/CRANE/')


import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from typing import Union
from PIL import Image
# === import external modules ===


# <<< import CRANE modules <<<
from PersistentModel import PersistentModel
from src.utils import *
# === import CRANE modules ===


NUM_PNG     : int = 100
DELTA_PNG   : int = 1  # output frequency for png. DO NOT SET TO < 1


NUM_NPY     : int = 100
DELTA_NPY   : int = 10


PRED_FRAMES : int = 200    # 0 for predicting same frames of true sequence


# <<< SCRIPT VARIABLES <<<
LOG_DIR         = Path('/home/fiorello/master_thesis/machine_learning/train/train_logs/lr5e-5_hl3_2_tr10')
SEQUENCE_TABLE  : str = '/data/fiorello/testtest/test_set_random.txt'
OUTPUT_FOLDER   : str = '/data/fiorello/test_del_test2'
CUDA            : bool = True
FRAME_BLOCK     : int = 1
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


class CustomNameSpace():
    '''
    Dummy class to have the correct interface with PF.utils
    '''
    pass


class OutputMan():
    def __init__(self, path, deltaNPY=-1, deltaPNG=-1, has_ground_truth=None):
        self.path = path
        self.has_ground_truth = has_ground_truth  # None = da determinare, True/False = deciso
    
        self.deltaNPY = deltaNPY
        if deltaNPY > 0:
            os.mkdir(f'{path}/pred_npy')

        self.deltaPNG = deltaPNG
        if deltaPNG > 0:
            os.mkdir(f'{path}/pred_png')
            os.mkdir(f'{path}/diff_png')
            
            self.cmap = LinearSegmentedColormap.from_list(
                        "cwr",
                        ["cyan", "white", "red"]
                        )
        
        self.fileEVO = open(f"{self.path}/evo.txt", "w")
        self.fileEVO.write("# 1: time | 2: MAE | 3: MSE | 4: mean_True | 5: mean_Pred | 6: min_True | 7: min_Pred | 8: max_True | 9: max_Pred | 10: symdiff\n")

        self.maxMae = 0.0
        self.maxMse = 0.0
        self.sumMae = 0.0
        self.sumMse = 0.0
        self.maxSymDiff = 0.0
        self.sumSymDiff = 0.0

        self.niter = 0
        self.niter_eval = 0  # conta solo i frame con ground truth valido

    def __del__(self):
        self.fileEVO.close()

    def savePNG(self, fname, phi):
        phiclip = np.clip(phi, 0, 1.0)
        pix = (phiclip * 255).astype(np.uint8)
        img = Image.fromarray(pix, mode='L')
        img.save(fname)

    def writeEVO(self, time, true, pred):
        """
        Scrive una riga in evo.txt e salva PNG/NPY se necessario.
        
        Casi:
        - has_ground_truth=True:  true e pred sono entrambi validi, calcola MAE/MSE
        - has_ground_truth=False: true è None, scrive solo predizione senza metriche
        - has_ground_truth=None:  decide al primo frame in base alla disponibilità di true
        """
        # Determina se abbiamo ground truth al primo frame
        if self.has_ground_truth is None:
            if true is not None:
                self.has_ground_truth = True
            else:
                self.has_ground_truth = False

        # Salva sempre PNG/NPY se nei delta corretti
        if self.deltaNPY > 0 and time % self.deltaNPY == 0:
            np.save(f'{self.path}/pred_npy/{str(time).zfill(3)}.npy', pred)

        if self.deltaPNG > 0 and time % self.deltaPNG == 0:
            self.savePNG(f'{self.path}/pred_png/{str(time).zfill(3)}.png', pred)

        # Se abbiamo ground truth, calcola metriche e salva diff
        if self.has_ground_truth and true is not None:
            mae = np.abs(pred - true).mean()
            mse = ((pred - true) ** 2.0).mean()

            self.maxMae = max(self.maxMae, mae)
            self.maxMse = max(self.maxMse, mse)
            self.sumMae += mae
            self.sumMse += mse

            symDiff = np.abs(pred.round() - true.round()).mean()

            self.maxSymDiff = max(self.maxSymDiff, symDiff)
            self.sumSymDiff += symDiff

            self.niter_eval += 1

            # Scrivi su evo.txt con metriche
            self.fileEVO.write(
                f"{time}\t{mae}\t{mse}\t{true.mean()}\t{pred.mean()}\t"
                f"{true.min()}\t{pred.min()}\t{true.max()}\t{pred.max()}\t{symDiff}\n"
            )

            # Salva diff PNG
            if self.deltaPNG > 0 and time % self.deltaPNG == 0:
                diff = np.clip(pred - true, -1.0, 1.0)
                plt.imsave(
                    f'{self.path}/diff_png/{str(time).zfill(3)}.png',
                    diff,
                    cmap=self.cmap,
                    vmin=-1, vmax=1
                )
        else:
            # Nessun ground truth: scrivi solo predizione senza metriche
            self.fileEVO.write(
                f"{time}\tNaN\tNaN\tNaN\t{pred.mean()}\t"
                f"NaN\t{pred.min()}\tNaN\t{pred.max()}\tNaN\n"
            )

    def writeSTAT(self, fileSTAT, seq_name):
        """
        Scrive le statistiche aggregate per la sequenza.
        Se non c'è ground truth, scrive NaN per le metriche.
        """
        if self.niter_eval == 0:
            # Nessun ground truth disponibile
            fileSTAT.write(
                f"{seq_name} NaN NaN NaN NaN NaN NaN\n"
            )
        else:
            avgMae = self.sumMae / self.niter_eval
            avgMse = self.sumMse / self.niter_eval
            avgSymDiff = self.sumSymDiff / self.niter_eval

            fileSTAT.write(
                f"{seq_name} {self.maxMae} {self.maxMse} {avgMae} {avgMse} "
                f"{self.maxSymDiff} {avgSymDiff}\n"
            )


def best_model_path(log_dir_path: str) -> Path:
    valid_loss_file = LOG_DIR / "valid_loss.txt"

    if not LOG_DIR.is_dir():
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

    model_path = LOG_DIR / "model" / f"epoch_{best_epoch}.pt"

    if not model_path.is_file():
        raise FileNotFoundError(f"Il miglior modello atteso non esiste: {model_path}")

    print(f"Best model trovato in {LOG_DIR}:\nepoch={best_epoch}, valid_loss={min_loss}")

    return model_path


def main() -> None:
    '''
    Main function entrypoint
    '''
    overwrite_key = None
    if os.path.isdir(OUTPUT_FOLDER):
        overwrite_key = input(f'Output folder "{OUTPUT_FOLDER}" already exists. "d" delete, "a" abort - "ENTER or other key to continue).\n').lower()
    if overwrite_key == 'd':
        os.system(f'rm -r {OUTPUT_FOLDER}')
    elif overwrite_key == 'a':
        exit()
    if overwrite_key is None or overwrite_key == 'd':
        os.mkdir(OUTPUT_FOLDER)

    if CUDA:
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            print('cuda seems not to be available, continuing using cpu')
            device = 'cpu'
    else:
        device = 'cpu'

    model = PersistentModel(
            hidden_units        = HIDDEN_UNITS,
            input_channels      = INPUT_CHANNELS,
            output_channels     = OUTPUT_CHANNELS,
            hidden_channels     = HIDDEN_CHANNELS,
            kernel_size         = KERNEL_SIZE,
            padding_mode        = PADDING_MODE,
            separable           = SEPARABLE,
            bias                = BIAS,
            divergence          = DIVERGENCE,
            num_params          = NUM_PARAMS,
            dropout             = DROPOUT,
            dropout_prob        = DROPOUT_PROB,
            conservative        = CONSERVATIVE
        )

    # reload model
    model_path = best_model_path(LOG_DIR)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    model.to(device)
    model.make_divFilters(torch.zeros(1, device=device))

    countNPYout = 0
    countPNGout = 0

    fileSTAT = open(f"{OUTPUT_FOLDER}/errors.txt", "w")
    fileSTAT.write("# 1: id | 2: maxMAE | 3: maxMSE | 4: overallMAE | 5: overallMSE | 6: max(symDiff) | 7: avg(symDiff)\n")

    with torch.no_grad():
        with open(SEQUENCE_TABLE, "r") as intable:
            listseq = intable.readlines()

        # Determina se questa sequence_table ha ground truth completo
        first_seq_frames = listseq[0].strip().split()
        has_full_truth = (len(first_seq_frames) >= PRED_FRAMES)
        
        print(f"Sequence table has {'full ground truth' if has_full_truth else 'only MIN_SEQ frames'}")

        for seq in listseq:
            seq_name = seq.split()[0].split("/")[-2]
            seq_path = f'{OUTPUT_FOLDER}/{seq_name}'
            
            if os.path.isdir(seq_path):
                print(seq_path + " already exists!")
                continue
            
            os.mkdir(seq_path)
            os.symlink(
                seq.split()[0].replace(f'{seq.split()[0].split("/")[-1]}', ""),
                f'{seq_path}/true_npy',
                target_is_directory=True
            )

            model.zero_grad()

            if countNPYout < NUM_NPY:
                dNPY = DELTA_NPY
            else:
                dNPY = -1
            countNPYout += 1
            
            if countPNGout < NUM_PNG:
                dPNG = DELTA_PNG
            else:
                dPNG = -1
            countPNGout += 1

            out = OutputMan(seq_path, dNPY, dPNG, has_ground_truth=has_full_truth)
            
            # Load frames
            iniSeq = []
            trueSeq = []
            
            for frame_path in seq.split():
                phi = np.load(frame_path)
                trueSeq.append(phi)
                
                if len(iniSeq) < MIN_SEQ:
                    out.writeEVO(len(iniSeq), true=phi, pred=None)
                    
                    phi_tensor = torch.from_numpy(phi).float()
                    while len(phi_tensor.shape) < 4:
                        phi_tensor = phi_tensor.unsqueeze(0)
                    iniSeq.append(phi_tensor)
            
            iniSeq = torch.stack(iniSeq, dim=1)
            iniSeq = iniSeq.to(device)
            model.set_hidden(iniSeq[:, 0:1, ...])

            params = None
            if params is not None:
                params = params.to(device)

            print(f'Predicting sequence {seq_name}...')
            
            # Prima predizione dopo MIN_SEQ
            predSeq = model(
                iniSeq,
                future=0,
                params=params,
                approx_inference=False
            )
            time = MIN_SEQ

            predSeq = predSeq[:, MIN_SEQ-1:MIN_SEQ, ...]
            pred_frame = predSeq[0, 0, 0, ...].cpu().numpy()
            
            # Scrivi primo frame predetto
            if has_full_truth and time < len(trueSeq):
                out.writeEVO(time, true=trueSeq[time], pred=pred_frame)
            else:
                out.writeEVO(time, true=None, pred=pred_frame)

            # Loop autoregressivo
            while time < PRED_FRAMES:
                if time % 50 == 0:
                    print(time, end="...", flush=True)
                
                predSeq = model(
                    predSeq,
                    future=0,
                    params=params,
                    approx_inference=False
                )
                time += 1
                
                pred_frame = predSeq[0, 0, 0, ...].cpu().numpy()
                
                # Scrivi frame
                if has_full_truth and time < len(trueSeq):
                    out.writeEVO(time, true=trueSeq[time], pred=pred_frame)
                else:
                    out.writeEVO(time, true=None, pred=pred_frame)

            out.writeSTAT(fileSTAT, seq_name)

            print('DONE!')
            print()
            print('=' * 30)

            del iniSeq
            del trueSeq
            del predSeq

    file_path = open(OUTPUT_FOLDER + '/model_path.txt', 'a')
    print(str(best_model_path(LOG_DIR)), file=file_path)
    file_path.close()

    fileSTAT.close()


if __name__ == '__main__':
    main()