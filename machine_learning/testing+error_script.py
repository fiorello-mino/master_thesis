# <<< import external modules <<<
from sys import path
path.append('/home/fiorello/CRANE/')

import numpy as np

import torch
import torch.nn as nn
from torchvision import transforms

import os

import matplotlib.pyplot as plt

from typing import Union

from PIL import Image
# === import external modules ===

# <<< import CRANE modules <<<
from src.classes import ConvGRU, ConvGRUClassifier
from src.utils import *
from src.dataloaders import TabulatedSeries
# === import CRANE modules ===



NUM_PNG : int = 100
NUM_VTK : int = 0
NUM_NPY : int = 100
NUM_EVO : int = 25000

# <<< SCRIPT VARIABLES <<<
MODEL_PATH      : str   = '/home/fiorello/master_thesis/machine_learning/train/train_logs/lr_5e-5_hl_3_128/model/epoch_259.pt' # model
SEQUENCE_TABLE  : str   = '/data/fiorello/external_test_64_random/testing_set.txt'
IMG_SIZE        : int   = 64 # resizing dimension of images in dataset
OUTPUT_FOLDER   : str   = '/data/fiorello/external_test_64_random/test_lr_5e-5_hl_3_train128'#' # output folder name
CUDA            : bool  = True # cuda control variable
DELTA_PNG       : int   = 1  # output frequency for png. DO NOT SET TO < 1

# === SCRIPT VARIABLES ===


# <<< MODEL VARIABLES <<<
MIN_SEQ         : int                   = 1
HIDDEN_UNITS    : int                   = 3 # cambia
INPUT_CHANNELS  : int                   = 1 
OUTPUT_CHANNELS : int                   = 1
HIDDEN_CHANNELS : int                   = 16 # cambia
KERNEL_SIZE     : int                   = 5 # cambia
PADDING_MODE    : str                   = 'circular'    # this is NOT to be modified
SEPARABLE       : bool                  = False         # this is NOT to be modified
BIAS            : bool                  = True          # modify ONLY IF you know what you are doing
DIVERGENCE      : bool                  = True
NUM_PARAMS      : int                   = 0
DROPOUT         : bool                  = False
DROPOUT_PROB    : Union[float, None]    = None
# === MODEL VARIABLES ===


class CustomNameSpace():
    '''
    This is a dummy class used to have the correct interface with PF.utils
    '''
    pass




def load_sequences(
        path        : str,
        num_params  : int,
        transforms  : Union[transforms.transforms.Compose, None]
        ) -> tuple[ torch.Tensor, Union[list[float], None] ]:
    '''
    This iterator loads the sequence provided by path, apply transforms and returns a torch tensor object
    '''

    with open(path, 'r') as sequence_file:
        lines = sequence_file.readlines()

    for line in lines:
        if(num_params==0):
            snaps_paths = line.split()
        else:
            snaps_paths = line.split()[:-num_params] # need to remove last num_params values

        # dealing with parameter part
        if num_params == 0 or num_params is None:
            params = None
            print('params: ', params)
        else:
            params = torch.tensor( [[ float(val) for val in line.split()[-num_params:] ]] ).float()

        # dealing with sequence part
        sequence = []
        for snap_path in snaps_paths:
            if snap_path.endswith('.png'):
                state = Image.open(snap_path).convert('L')
                state = np.array(state)
            elif snap_path.endswith('.npy'):
                state = np.load(snap_path)
            else:
                raise ValueError(f'Extension {snap_path.split(".")[-1]} is not supported.')

            state = torch.from_numpy(state).float()

            while len(state.shape) < 4:
                state = state.unsqueeze(0) # add dimensions if necessary

            sequence.append(transforms(state))

        sequence = torch.stack( sequence, dim=1 )


        yield sequence, params


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

    transform = transforms.Compose(
            [
                transforms.Grayscale( num_output_channels=1 ), # ensure that images have a single field channel
                transforms.Resize( IMG_SIZE ), # resize image
                ]
        )

    model = ConvGRU(
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
            dropout_prob        = DROPOUT_PROB

        )

    # reload model
    model.load_state_dict( torch.load(MODEL_PATH, map_location=device) )
    model.eval() # better safe than sorry

    model.to(device)
    model.make_div_filters( torch.zeros(1, device=device ) )

    sequences_iterator = load_sequences( SEQUENCE_TABLE, NUM_PARAMS, transform)

    with open(SEQUENCE_TABLE, 'r') as sequence_file:
                list_sequences = sequence_file.readlines()

    jump = MIN_SEQ
    num_params=0
    countNPYout=0
    countEVOout=0
    countPNGout=0
    countVTKout=0

    file_MAE = open(f"{OUTPUT_FOLDER}/errors.txt", "w")
    file_MAE.write("# 1: id | 2: maxMAE | 3: maxMSE | 4: overallMAE | 5: overallMSE\n")


    with torch.no_grad():

        loss_fn = nn.MSELoss()

        for kk, (sequence, params) in enumerate(sequences_iterator):

            model.zero_grad()

            seq_name = list_sequences[kk].split()[0].split("/")[-2]
            kk_path = f'{OUTPUT_FOLDER}/{seq_name}'

#            kk_path = f'{OUTPUT_FOLDER}/{str(kk).zfill(4)}'
            if os.path.isdir(kk_path):
                print(kk_path + " already exists!")
                pass

            os.mkdir(kk_path)
            os.symlink(list_sequences[kk].split()[0].replace(f'/{list_sequences[kk].split()[0].split("/")[-1]}',"")  , f'{kk_path}/true_sequence_npy', target_is_directory=True)


            print(f'Predicting sequence {seq_name}...')

            sequence    = sequence.to(device)
            if params is not None:
                params      = params.to(device)

            initial_state = sequence[:,:jump,...]
            target_sequence = sequence[:,jump:,...]

            pred_sequence = model(
                    initial_state,
                    future = target_sequence.shape[1]-1,
                    params = params,
                    approx_inference = False
                    )

            pred_sequence = torch.cat( [initial_state, pred_sequence[:,jump-1:,...]], dim=1 ) # jump-1 since pred sequence is one off wrt true sequence (t=0 is t=1 in input!)

#            mse = loss_fn(pred_sequence, sequence).cpu().numpy()
#            mae = ( (pred_sequence-sequence).abs() ).mean().cpu().numpy()
            mae=np.zeros(pred_sequence.shape[1])
            mse=np.zeros(pred_sequence.shape[1])
            if(countEVOout < NUM_EVO):
                file_EVO = open(f"{kk_path}/evo.txt", "w")
                file_EVO.write("# 1: MAE | 2: MSE | 3: cov_True | 4: cov_Pred | 5: min_True | 6: min_Pred | 7: max_True | 8: max_Pred\n")
                for t in range(jump):
                    true=sequence[:,t,...].cpu().numpy()
                    avgT=true.mean()
                    minT=true.min()
                    maxT=true.max()
                    file_EVO.write(f"nan \t nan \t {avgT} \t nan \t {minT} \t nan \t {maxT} \t nan \n")
                countEVOout += 1

            for t in range(jump,pred_sequence.shape[1]):
                pred=pred_sequence[:,t,...].cpu().numpy()
                true=sequence[:,t,...].cpu().numpy()
                mae[t]=( np.abs(pred-true) ).mean()
                mse[t]=( (pred-true)**2. ).mean()

                if(countEVOout < NUM_EVO):
                    avgP=pred.mean()
                    minP=pred.min()
                    maxP=pred.max()
                    avgT=true.mean()
                    minT=true.min()
                    maxT=true.max()
                    file_EVO.write(f"{mae[t]} \t  {mse[t]} \t {avgT} \t {avgP} \t {minT} \t {minP} \t {maxT} \t {maxP}\n")

            file_MAE.write(f"{seq_name} {np.max(mae)} {np.max(mse)} {np.mean(mae)} {np.mean(mse)}\n")

#            errors.append( loss_fn(pred_sequence, sequence).cpu() )
            #errors.append( ((pred_sequence-sequence)**2).mean() )

            
            # save predicted sequence
            if(countNPYout < NUM_NPY):
                print('Saving npy output...')
                os.mkdir(f'{kk_path}/pred_sequence_npy')
                seq2npy( pred_sequence.cpu(), path=f'{kk_path}/pred_sequence_npy', fname="", nf=3 )
                countNPYout += 1

            if(countPNGout < NUM_PNG):
#                os.mkdir(f'{kk_path}/true_sequence_npy')
#                seq2npy( sequence.cpu(), path=f'{kk_path}/true_sequence_npy', fname="", nf=3 )
#                os.mkdir(f'{kk_path}/diff_sequence_npy')   # Inserito da Matteo
#                seq2npy( ( pred_sequence.cpu() - sequence.cpu() ) , path=f'{kk_path}/diff_sequence_npy', fname="", nf=3 )   # Inserito da Matteo (TOLTO abs!!!!)

                print('Saving png output...')
#               os.mkdir(f'{kk_path}/true_sequence_png')
                os.mkdir(f'{kk_path}/pred_sequence_png')
                os.mkdir(f'{kk_path}/diff_sequence_png')
                
#                args = CustomNameSpace()
#                args.nproc = 4
#                args.cmap = 'gray'
#                args.paths = {"png" : f'{kk_path}/true_sequence_png'}
#                args.vmin = 0.
#                args.vmax = 1.
#                seq2png_treaded( sequence[:,::DELTA_PNG,...].cpu(), name=f'snap', delta=DELTA_PNG, args=args)

                args = CustomNameSpace()
                args.nproc = 4
                args.cmap = 'gray'
                args.paths = {"png" : f'{kk_path}/pred_sequence_png'}
                args.vmin = 0.
                args.vmax = 1.
                seq2png_treaded( pred_sequence[:,::DELTA_PNG,...].cpu(), name=f'snap', delta=DELTA_PNG, args=args )

                args = CustomNameSpace()
                args.nproc = 4
                args.cmap = 'bwr'
                args.clim = [-1., 1.]
                args.vmin = -1.
                args.vmax = 1.
                args.paths = {"png" : f'{kk_path}/diff_sequence_png'}
                seq2png_treaded( (pred_sequence[:,::DELTA_PNG,...].cpu()-sequence[:,::DELTA_PNG,...].cpu()), name=f'snap', delta=DELTA_PNG, args=args )      # TOLTO abs!!!!
                countPNGout += 1

            if(countVTKout < NUM_VTK):
                print('Saving vtk output...')
                os.mkdir(f'{kk_path}/true_sequence_vtk')
                os.mkdir(f'{kk_path}/pred_sequence_vtk')
                seq2vtk( sequence.cpu(), path=f'{kk_path}/true_sequence_vtk' )
                seq2vtk( pred_sequence.cpu(), path=f'{kk_path}/pred_sequence_vtk' )
                countVTKout += 1


            print('DONE!')
            print()
            print()
            print('='*30)

            del pred_sequence
            del sequence

    #if isinstance(errors, list):
    #    errors = torch.tensor(errors, device='cpu')
    #    errors = errors.cpu().numpy()

    #errors = np.array(errors)

    file_path = open(OUTPUT_FOLDER + '/model_path.txt', 'a')    # Inserito da Matteo (serve per tenere traccia del modello utilizzato
    print(MODEL_PATH, file=file_path)
    file_path.close()


    file_MAE.close()




if __name__ == '__main__':
    main()
