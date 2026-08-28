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
DELTA_PNG   : int   = 10  # output frequency for png. DO NOT SET TO < 1

NUM_NPY     : int = 100
DELTA_NPY   : int = 10

PRED_FRAMES : int = 200    # 0 for predicting same frames of true sequence

# <<< SCRIPT VARIABLES <<<
LOG_DIR         = Path('/home/fiorello/master_thesis/machine_learning/train_pores/train_logs/coeffE1e-3_coeffG3e-4_hl3_reload_random') # model
SEQUENCE_TABLE  : str   = '/data/fiorello/pores/ext_test/ext_test_var_depth/test_set.txt'
OUTPUT_FOLDER   : str   = '/data/fiorello/test_del_test'#' # output folder name
CUDA            : bool  = True # cuda control variable
FRAME_BLOCK     : int   = 5 # 0 for full sequence

# === SCRIPT VARIABLES ===


# <<< MODEL VARIABLES <<<
MIN_SEQ         : int                   = 1
HIDDEN_UNITS    : int                   = 3 # cambia
INPUT_CHANNELS  : int                   = 1 
OUTPUT_CHANNELS : int                   = 1
HIDDEN_CHANNELS : int                   = 16 # cambia
KERNEL_SIZE     : int                   = 5 # cambia
PADDING_MODE    : str                   = ["circular", "reflect"]
SEPARABLE       : bool                  = False         # this is NOT to be modified
BIAS            : bool                  = True          # modify ONLY IF you know what you are doing
DIVERGENCE      : bool                  = True
NUM_PARAMS      : int                   = 0
DROPOUT         : bool                  = False
DROPOUT_PROB    : Union[float, None]    = None
CONSERVATIVE    : bool                  = False
# === MODEL VARIABLES ===


class CustomNameSpace():
    '''
    This is a dummy class used to have the correct interface with PF.utils
    '''
    pass

class OutputMan():
    def __init__(self, path, deltaNPY=-1, deltaPNG=-1):
        self.path = path
    
        self.deltaNPY=deltaNPY
        if(deltaNPY >0 ):
            os.mkdir(f'{path}/pred_npy')

        self.deltaPNG=deltaPNG
        if(deltaPNG >0):
            os.mkdir(f'{path}/pred_png')
            os.mkdir(f'{path}/diff_png')
            
            self.cmap = LinearSegmentedColormap.from_list(
                        "cwr",
                        ["cyan", "white", "red"]
                        )
        
        self.fileEVO = open(f"{self.path}/evo.txt", "w")
        self.fileEVO.write("# 1: time | 2: MAE | 3: MSE | 4: mean_True | 5: mean_Pred | 6: min_True | 7: min_Pred | 8: max_True | 9: max_Pred | 10: symdiff \n" )

        self.maxMae=0
        self.maxMse=0
        self.sumMae=0
        self.sumMse=0
        self.maxSymDiff=0
        self.sumSymDiff=0

        self.niter=0

    def __del__(self):
        self.fileEVO.close()

    def savePNG(self,fname,phi):
        phiclip=np.clip(phi, 0, 1.0)
        pix = (phiclip*255).astype(np.uint8)
        img = Image.fromarray(pix, mode='L')
        img.save(fname)

    def writeEVO( self, time, true=None, pred=None):
        if(pred is None):
            if(true is None):
                sys.exit("Both pred and true are None\n")
            self.fileEVO.write(f"nan\t nan \t {true.mean()} \t nan \t {true.min()} \t nan \t {true.max()} \t nan \t nan \n")
            self.savePNG(f"{self.path}/pred_png/{str(time).zfill(3)}.png", true)

        elif(true is None):
            self.fileEVO.write(f"nan\t nan \t nan \t {pred.mean()} \t nan \t pred.mean() \t nan \t {pred.max()} \t nan \n")
            if( self.deltaNPY>0 and time % self.deltaNPY==0):
                np.save(f'{self.path}/pred_npy/{str(time).zfill(3)}.npy', pred)

            if( self.deltaPNG>0 and time % self.deltaPNG==0):
                self.savePNG(f'{self.path}/pred_png/{str(time).zfill(3)}.png', pred)
 
        else:
            mae=(np.abs(pred-true)).mean()
            mse=( (pred-true)**2. ).mean()

            self.maxMae = max(self.maxMae, mae)
            self.maxMse = max(self.maxMse, mse)
            self.sumMae += mae
            self.sumMse += mse

            symDiff = (np.abs(pred.round()-true.round()) ).mean()

            self.maxSymDiff = max(self.maxSymDiff, symDiff)
            self.sumSymDiff += symDiff

            self.niter += 1

            self.fileEVO.write(f"{mae}\t {mse} \t {true.mean()} \t {pred.mean()} \t {true.min()} \t {pred.min()} \t {true.max()} \t {pred.max()} \t {symDiff}\n")

            if( self.deltaNPY>0 and time % self.deltaNPY==0):
                np.save(f'{self.path}/pred_npy/{str(time).zfill(3)}.npy', pred)

            if( self.deltaPNG>0 and time % self.deltaPNG==0):
                self.savePNG(f'{self.path}/pred_png/{str(time).zfill(3)}.png', pred)
                diff = np.clip(pred - true , -1.0,1.0)
                plt.imsave(f'{self.path}/diff_png/{str(time).zfill(3)}.png',
                            diff, 
                            cmap=self.cmap, 
                            vmin=-1, vmax=1)

    def writeBlock(self,time,true, pred):
        for t in range(pred.shape[1]):
            time += 1
            if(time>=len(true)):
                self.writeEVO(time, true=None, pred=pred[0,t,0,...].cpu().numpy())
            else:
                self.writeEVO(time,true[time],pred[0,t,0,...].cpu().numpy())

    def writeSTAT(self,fileSTAT,seq_name):
        fileSTAT.write("{seq_name} {self.maxMae} {self.maxMse} {self.sumMae/self.niter} {self.sumMse/self.niter} {self.maxSymDiff} {self.sumSymDiff/self.niter}\n")


def best_model_path(log_dir_name: str) -> Path:
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

    print(f"Best model trovato in {LOG_DIR}: \n 
          epoch={best_epoch}, valid_loss={min_loss}")

    return model_path


def main() -> None:
    '''
    This is the main function entrypoint
    '''

    overwrite_key = None
    if os.path.isdir(OUTPUT_FOLDER):
        overwrite_key = input(f'Output folder "{OUTPUT_FOLDER}" already exists. "d" delete, "a" abort - "ENTER or other key to continue).\n').lower()
    if overwrite_key == 'd':
        os.system(f'rm -r {OUTPUT_FOLDER}') # cleaning old folder
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

#    transform = transforms.Compose(
#            [
#                transforms.Grayscale( num_output_channels=1 ), # ensure that images have a single field channel
#            ]
#        )

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
    model.load_state_dict( torch.load(model_path, map_location=device) )
    model.eval() # better safe than sorry

    model.to(device)
    model.make_div_filters( torch.zeros(1, device=device ) )

    num_params=0
    countNPYout=0
    countEVOout=0
    countPNGout=0

    fileSTAT = open(f"{OUTPUT_FOLDER}/errors.txt", "w")
    fileSTAT.write("# 1: id | 2: maxMAE | 3: maxMSE | 4: overallMAE | 5: overallMSE | 6: max(symDiff) | 7: avg(symDiff) \n")


    with torch.no_grad():

        with open(SEQUENCE_TABLE,"r") as intable:
            listseq=intable.readlines()

        for seq in listseq:     #/home/0000/000.npy
            seq_name = seq.split()[0].split("/")[-2]    #0000
            seq_path = f'{OUTPUT_FOLDER}/{seq_name}'
            if os.path.isdir(seq_path):
                print(seq_path + " already exists!")
                pass
            os.mkdir(seq_path)
            os.symlink(seq.split()[0].replace(f'{seq.split()[0].split("/")[-1]}',""), f'{seq_path}/true_npy', target_is_directory=True)

            model.zero_grad()

            if(countNPYout < NUM_NPY):
                dNPY = DELTA_NPY
            else:
                dNPY = -1
            countNPYout += 1
            if(countPNGout < NUM_PNG):
                dPNG = DELTA_PNG
            else:
                dPNG = -1
            countPNGout += 1
 
            out=OutputMan(seq_path, dNPY, dPNG)

            iniSeq = []
            trueSeq = []
            for frame_path in seq.split():
                phi = np.load(frame_path)

                trueSeq.append(phi)
                if(len(iniSeq)<MIN_SEQ):
                    out.writeEVO(len(iniSeq),true=phi,pred=None)

                    phi = torch.from_numpy(phi).float()
                    while len(phi.shape) < 4:
                        phi = phi.unsqueeze(0)
                    iniSeq.append(phi)
            iniSeq = torch.stack( iniSeq, dim=1 )
            iniSeq = iniSeq.to(device)

            params = None
            if params is not None:
                params = params.to(device)

            time=len(iniSeq)-1
            if(PRED_FRAMES>0):
                pred_frames = PRED_FRAMES
            else:
                pred_frames = len(trueSeq)

            if(FRAME_BLOCK>MIN_SEQ):
                frame_block=FRAME_BLOCK-MIN_SEQ
            else:
                frame_block=pred_frames

            print(f'Predicting sequence {seq_name}...')
            predSeq = model(
                    iniSeq,
                    future = frame_block,
                    params = params,
                    approx_inference = False
                    )
            out.writeBlock(time,trueSeq,predSeq)
            time += frame_block

            while time < pred_frames:
                predSeq = model(
                        predSeq,
                        future = 0,
                        params = params,
                        approx_inference = False
                        )
                out.writeBlock(time,trueSeq,predSeq)
                time += frame_block

            out.writeSTAT(fileSTAT, seq_name)

            print('DONE!')
            print()
            print()
            print('='*30)

            del iniSeq
            del trueSeq
            del predSeq

    file_path = open(OUTPUT_FOLDER + '/model_path.txt', 'a')
    print(MODEL_PATH, file=file_path)
    file_path.close()

    fileSTAT.close()


if __name__ == '__main__':
    main()
