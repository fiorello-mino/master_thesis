# <<< importing external stuff <<<
import os
import sys
import shutil

import torch
import torch.nn as nn
from torchvision import utils, datasets, transforms

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import PIL
from PIL import Image

from warnings import warn

from multiprocessing import Process

import pyvista as pv
# --- importing external stuff ---


# <<< import numba <<<
try:
    from numba import njit, prange
except ImportError:
    print(
        'It seems that "numba" is not installed on this machine or there are '
        'problems in importing it. Falling back on non-jitted versions of scripts. '
        'Some operations will be slower. Consider installing it (e.g. by '
        '"pip install numba").'
    )

    def njit(fun):
        return fun
# --- import numba ---


def build_train_logs_dir_tree(args):
    """
    Build output folder structure for training and attach paths to args.paths.
    """
    master = f'train_logs/{args.id}'

    if os.path.isdir(master):
        print(f'Naming conflict found with id "{args.id}".')
        num_existing_folders = len(
            [subfolder for subfolder in os.listdir('train_logs') if args.id in subfolder]
        )
        args.id = f'{args.id}_{num_existing_folders}'
        master = f'train_logs/{args.id}'
        print(f'Naming conflict handled. Training logs will be saved in {master}')

    os.makedirs(master, exist_ok=False)

    model_path = f'{master}/model'

    if args.graphics:
        pngs_path = f'{master}/pngs'
        lossplot_path = f'{master}/lossplot.png'
        vtk_path = f'{master}/vtk' if args.vtk else None
        npy_path = f'{master}/npy' if args.npy else None
        gifs_path = f'{master}/gifs' if args.gifs else None
    else:
        pngs_path = None
        lossplot_path = None
        gifs_path = None
        vtk_path = None
        npy_path = None

    trainloss_path = f'{master}/train_loss.txt'
    validloss_path = f'{master}/valid_loss.txt'

    args.paths = {
        'master': master,
        'model': model_path,
        'png': pngs_path,
        'gif': gifs_path,
        'lossplot': lossplot_path,
        'trainloss': trainloss_path,
        'validloss': validloss_path,
        'vtk': vtk_path,
        'npy': npy_path,
    }

    for name in ['model', 'png', 'gif', 'vtk', 'npy']:
        if args.paths[name] is not None:
            os.makedirs(args.paths[name], exist_ok=False)

    return args


def build_test_dir_tree(args):
    """
    Build output folder structure for testing.
    """
    master = f'test_outputs/{args.id}'

    if os.path.isdir(master):
        print(f'Naming conflict found with id "{args.id}".')
        num_existing_folders = len(
            [subfolder for subfolder in os.listdir('test_outputs') if args.id in subfolder]
        )
        args.id = f'{args.id}_{num_existing_folders}'
        master = f'test_outputs/{args.id}'
        print(f'Naming conflict handled. Testing outputs will be saved in {master}')

    os.makedirs(master, exist_ok=False)

    if args.graphics:
        png_path = f'{master}/png'
        gifs_path = f'{master}/gifs' if args.gifs else None
    else:
        warn(
            'Graphics was disabled for testing procedure. '
            'Little information for many calculations will be produced.'
        )
        png_path = None
        gifs_path = None

    area_path = f'{master}/area'
    progloss_path = f'{master}/progloss'
    AR_path = f'{master}/AR' if args.AR else None

    args.paths = {
        'master': master,
        'png': png_path,
        'gif': gifs_path,
        'area': area_path,
        'progloss': progloss_path,
        'AR': AR_path,
    }

    for name in ['png', 'gif', 'area', 'progloss', 'AR']:
        if args.paths[name] is not None:
            os.makedirs(args.paths[name], exist_ok=False)

    return args


def build_predict_dir_tree(args):
    """
    Build directory structure for prediction / evaluation runs.
    """
    master = f'out/{args.id}'

    if os.path.isdir(master):
        print(f'Naming conflict found with id "{args.id}".')
        num_existing_folders = len(
            [subfolder for subfolder in os.listdir('out') if args.id in subfolder]
        )
        args.id = f'{args.id}_{num_existing_folders}'
        master = f'out/{args.id}'
        print(f'Naming conflict handled. Prediction outputs will be saved in {master}')

    os.makedirs(master, exist_ok=False)

    pngs_path = f'{master}/pngs' if args.graphics else None
    AR_path = f'{master}/AR' if args.AR else None
    gif_path = f'{master}/gifs' if (args.gifs and args.graphics) else None

    phi_0_path = f'{master}/initial_condition.png'
    area_path = f'{master}/area'

    init_geo_source_path = f'{master}/init_geo.py' if getattr(args, 'gengeo', False) else None

    args.paths = {
        'png': pngs_path,
        'AR': AR_path,
        'phi_0': phi_0_path,
        'geo_source': init_geo_source_path,
        'gifs': gif_path,
        'area': area_path,
    }

    for name in ['png', 'gifs']:
        if args.paths[name] is not None:
            os.makedirs(args.paths[name], exist_ok=False)

    return args


def save_args(args):
    """
    Save argparse namespace to args.txt in the master folder.
    """
    with open(f'{args.paths["master"]}/args.txt', 'w+') as args_file:
        for key, value in vars(args).items():
            args_file.write(f'{key}\t:\t{value}\n')


def print_model_info(model):
    """
    Print basic model info (num params, committee info if applicable).
    """
    if not hasattr(model, 'model_list'):
        params_nums = sum(p.numel() for p in model.parameters())
        print()
        print('<<< model infos <<<')
        print(f'The number of parameters in the model is: {params_nums}')
        print('--- model infos ---')
        print()
    elif hasattr(model, 'model_list'):
        params_nums = sum(p.numel() for p in model.model_list[0].parameters())
        num_models = len(model)
        print()
        print('<<< model infos <<<')
        print(
            f'model is a CommitteeModel combining inferences from {num_models} models.'
        )
        print(
            f'Each model has a number of parameters: {params_nums}, '
            f'totalling {num_models * params_nums} parameters'
        )
        print()
    else:
        raise RuntimeError(
            'It seems that model is neither a torch.nn.Module nor a CommitteeModel.'
        )


def import_model(model, args):
    """
    Load a .pt model into provided model instance (state_dict expected).
    """
    if not os.path.isfile(args.reload_model):
        raise FileNotFoundError(
            f'No model found at path "{args.reload_model}".'
        )

    print('<<< Loading model ... <<<')
    state = torch.load(args.reload_model, map_location='cpu')
    if isinstance(state, dict) and 'state_dict' in state:
        model.load_state_dict(state['state_dict'])
    else:
        model.load_state_dict(state)
    print('--- Loading done! ---')

    return model


def log_epoch_start_info(epoch, args):
    """
    Print info at the beginning of an epoch.
    """
    print()
    print('<<< Epoch starting... <<<')
    print(
        f'Starting epoch {epoch} with subseq in range '
        f'[{args.subseq_min}/{args.subseq_max}]'
    )
    print(f'master folder for outputs is {args.paths["master"]}')
    print()


def log_epoch_end_info(epoch, epoch_time, vals, train_loss, valid_loss, args):
    """
    Print info at the end of an epoch and basic stats.
    """
    pred, true = vals
    with torch.no_grad():
        pred_last = pred[0, -1, ...]
        true_last = true[0, -1, ...]

        min_pred = pred_last.min().item()
        max_pred = pred_last.max().item()
        min_true = true_last.min().item()
        max_true = true_last.max().item()

        max_deviation = torch.abs(true_last - pred_last).max().item()

        area_pred = pred_last.mean().item()
        area_pred_init = pred[0, 0, ...].mean().item()
        area_true = true_last.mean().item()

        area_delta_last = 100 * (area_pred - area_true) / (area_true + 1e-6)
        area_delta_start_end = 100 * (area_pred_init - area_pred) / (area_pred_init + 1e-6)

    print()
    print('<<< Epoch ended <<<')
    print(f'Ended epoch {epoch} in {epoch_time:.2f} s')
    print(f'Mean epoch training loss is: {train_loss:.2e}')
    print(f'Mean epoch validation loss is: {valid_loss:.2e}')
    print()
    print('<<< Training stats <<<')
    print(f'Predicted min: {min_pred:.2e}')
    print(f'True min: {min_true:.2e}')
    print(f'Predicted max: {max_pred:.2e}')
    print(f'True max: {max_true:.2e}')
    print(f'Absolute max deviation: {max_deviation:.2e}')
    print(f'Relative variation in mass in final state is: {area_delta_last:.2f} %')
    print(f'Relative variation from initial state is: {area_delta_start_end:.2f} %')
    print('--- Training stats ---')
    print()


def withdual(tensor):
    """
    Concatenate tensor with its dual (1 - tensor) along batch dimension.
    """
    return torch.cat([tensor, 1 - tensor], dim=0)


def clip_series(series, in_seq_length):
    """
    Clip temporal length of series to in_seq_length (works for 2D and 3D).
    Expects shape (B, T, ...).
    """
    return series[:, :in_seq_length, ...]


def make_lossplot(train_losses, valid_losses, args):
    """
    Output training/validation loss vs epochs.
    """
    print('Outputting train/valid loss plot...', end='')

    plt.figure()
    plt.plot(np.arange(len(train_losses)), np.array(train_losses))
    plt.plot(np.arange(len(valid_losses)), np.array(valid_losses))
    plt.legend(['Training loss', 'Validation loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(args.paths['lossplot'])
    plt.close()

    print('done!')


def out_png(im_pred, im_target, path, cmap, var=None):
    """
    Output comparison png: predicted, true, error, (optional variance).
    """
    with torch.no_grad():
        im_pred_np = im_pred.detach().cpu().numpy()
        im_target_np = im_target.detach().cpu().numpy()

        if var is None:
            f, axarr = plt.subplots(1, 3)
            axarr[0].imshow(im_pred_np, cmap=cmap, vmin=0, vmax=1)
            axarr[1].imshow(im_target_np, cmap=cmap, vmin=0, vmax=1)
            axarr[2].imshow(np.abs(im_target_np - im_pred_np), cmap=cmap, vmin=0, vmax=1)

            axarr[0].set_title('Predicted')
            axarr[1].set_title('True')
            axarr[2].set_title('Error')
        else:
            var_np = var.detach().cpu().numpy()

            f, axarr = plt.subplots(2, 2)
            axarr[0, 0].imshow(im_pred_np, cmap=cmap, vmin=0, vmax=1)
            axarr[0, 1].imshow(im_target_np, cmap=cmap, vmin=0, vmax=1)
            axarr[1, 0].imshow(np.abs(im_target_np - im_pred_np), cmap=cmap, vmin=0, vmax=1)
            axarr[1, 1].imshow(var_np, cmap=cmap, vmin=0, vmax=1)

            axarr[0, 0].set_title('Predicted')
            axarr[0, 1].set_title('True')
            axarr[1, 0].set_title('Error')
            axarr[1, 1].set_title('Committee variance')

        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        del f, axarr


def out_gifs(y_pred, path):
    """
    Output gif of evolution.
    Expects y_pred shape (B=1, T, C=1, H, W).
    """
    imagelist = []
    for ii in range(y_pred.shape[1]):
        frame = (
            255
            * y_pred[0, ii, 0, :, :].detach().cpu().numpy()
        )
        imagelist.append(Image.fromarray(frame.astype(np.uint8)))

    imagelist[0].save(
        path,
        save_all=True,
        append_images=imagelist[1:],
        duration=100,
        loop=20,
    )

    del imagelist


def out_area_deviation(name, seq_pred, seq_target, graphic=True):
    """
    Output fractional area deviation (phi integral) (A_pred - A_true)/A_true.
    Saves .txt and optional .png.
    """
    print('Outputting area deviation analysis...')

    area_deviations = []
    with torch.no_grad():
        if seq_target.shape[1] == 1:
            area_init = seq_pred[0, 0, ...].sum().item()
            two_sequences = False
        else:
            two_sequences = True

        for kk in range(seq_pred.shape[1]):
            area_pred = seq_pred[0, kk, ...].sum().item()
            if two_sequences:
                area_true = seq_target[0, kk, ...].sum().item()
            else:
                area_true = area_init

            area_deviation = (area_pred - area_true) / (area_true + 1e-9)
            area_deviations.append(area_deviation)

    with open(f'{name}.txt', 'w') as area_deviation_file:
        for val in area_deviations:
            area_deviation_file.write(f'{val} ')

    if graphic:
        plt.figure()
        plt.plot(np.arange(len(area_deviations)) + 1, area_deviations)
        plt.xlabel('Frame number')
        plt.ylabel('Relative area error')
        plt.savefig(f'{name}.png')
        plt.close()

    print(f'Last area deviation is {100 * area_deviations[-1]:.2f} %')


def out_progloss(seq_pred, seq_target, name, graphic=True):
    """
    Output frame-wise MSE loss between seq_pred and seq_target.
    """
    progloss = []
    loss_fn = nn.MSELoss()
    with torch.no_grad():
        for kk in range(seq_pred.shape[1]):
            loss = loss_fn(
                seq_pred[0, kk, ...],
                seq_target[0, kk, ...],
            ).item()
            progloss.append(loss)

    with open(f'{name}.txt', 'w') as progloss_file:
        for val in progloss:
            progloss_file.write(f'{val} ')

    if graphic:
        plt.figure()
        plt.semilogy(np.arange(len(progloss)) + 1, progloss)
        plt.ylim(1e-6, 1e0)
        plt.xlabel('Frame number')
        plt.ylabel('MSE Loss')
        plt.savefig(f'{name}.png')
        plt.close()

    print(f'Last MSELoss value is {progloss[-1]:.2e}')


def save_model(model, path):
    """
    Save model state_dict to path.
    """
    torch.save(model.state_dict(), path)


def give_model_paths(master_path):
    """
    Return list of .pt model paths.
    If master_path is a .pt file, return [master_path].
    If directory, return all contained .pt files.
    """
    if master_path.endswith('.pt'):
        return [master_path]
    else:
        if master_path.endswith('/'):
            master_path = master_path[:-1]
        if not os.path.isdir(master_path):
            raise RuntimeError(
                f'Provided path "{master_path}" does not point '
                'to a directory or a .pt file.'
            )
        content_list = os.listdir(master_path)
        models_list = [m for m in content_list if m.endswith('.pt')]
        if len(models_list) == 0:
            raise RuntimeError(
                f'Provided path "{master_path}" points to a directory, '
                'but it seems that it does not contain any .pt file.'
            )
        model_paths = [f'{master_path}/{m}' for m in models_list]
        return model_paths


def make_square(im, min_size=1, fill_color=(0, 0, 0, 0), cropkey=True, crop_lim=(0.25, 0.75)):
    """
    Pad image to square with optional centered crop.
    """
    x, y = im.size
    size = max(min_size, x, y)
    new_im = Image.new('RGBA', (size, size), fill_color)
    new_im.paste(im, (int((size - x) / 2), int((size - y) / 2)))
    if cropkey:
        crop_low, crop_high = crop_lim
        new_im = new_im.crop(
            (
                int(crop_low * size),
                int(crop_low * size),
                int(crop_high * size),
                int(crop_high * size),
            )
        )
    return new_im


def copy_init_geo_source(args):
    """
    Copy geometry source code file to run folder.
    """
    if args.init_geo and args.paths.get('geo_source', None) is not None:
        shutil.copy(args.init_geo, args.paths['geo_source'])


def seq2png(seq, name, args, start_num):
    """
    Save sequence frames as png images.
    Expects seq shape (B=1, T_chunk, C=1, H, W).
    """
    for kk in range(seq.shape[1]):
        frame = seq[0, kk, 0, :, :].detach().cpu().numpy()
        plt.figure()
        plt.imshow(frame, cmap=args.cmap, vmin=0, vmax=1)
        plt.axis('off')
        plt.savefig(
            f'{args.paths["png"]}/{name}_frame_{kk + start_num}.png',
            bbox_inches='tight',
            pad_inches=0,
        )
        plt.close()


def seq2png_threaded(seq, name, args):
    """
    Use multiprocessing to output sequences into pngs.
    """
    seq_length = seq.shape[1]
    if seq_length == 0:
        raise RuntimeError('There was an attempt to output a sequence with length 0.')

    max_procs = os.cpu_count() or 1
    proc2use = min(args.nproc, max_procs)
    chunk_length = max(1, seq_length // proc2use)

    mp_args = []
    for kk in range(proc2use):
        start = kk * chunk_length
        end = (kk + 1) * chunk_length if kk != proc2use - 1 else seq_length
        if start >= seq_length:
            break
        mp_args.append((seq[:, start:end, ...], name, args, start))

    processes = []
    for arg in mp_args:
        process = Process(target=seq2png, args=arg)
        processes.append(process)
        process.start()
    for process in processes:
        process.join()


@njit
def estimate_AR(image):
    """
    Estimate aspect ratio of shapes along x and y directions (shape centered).
    """
    x_center = image.shape[0] // 2
    y_center = image.shape[1] // 2

    x_search = range(x_center - 1, x_center + 1)
    y_axis = 0.0
    for x in x_search:
        y_slice = image[x, :]
        total = y_slice.sum()
        if total >= y_axis:
            y_axis = total

    y_search = range(y_center - 1, y_center + 1)
    x_axis = 0.0
    for y in y_search:
        x_slice = image[:, y]
        total = x_slice.sum()
        if total >= x_axis:
            x_axis = total

    AR = x_axis / (y_axis + 1e-9)
    if AR < 1.0:
        AR = 1.0 / AR

    return AR


def out_AR(name, seq, graphic, true_seq=None):
    """
    Estimate AR for each frame in seq (and optionally true_seq).
    """
    print('Estimating sequence AR... ', end='')

    AR_list = []
    for kk in range(seq.shape[1]):
        im = seq[0, kk, 0, :, :].detach().cpu().numpy()
        AR = estimate_AR(im)
        AR_list.append(AR)

    ARs = np.array(AR_list)

    if true_seq is not None:
        AR_true_list = []
        for kk in range(true_seq.shape[1]):
            im = true_seq[0, kk, 0, :, :].detach().cpu().numpy()
            AR = estimate_AR(im)
            AR_true_list.append(AR)
        ARs_true = np.array(AR_true_list)
    else:
        ARs_true = None

    with open(f'{name}.txt', 'w') as AR_file:
        for AR in ARs:
            AR_file.write(f'{AR} ')
        if ARs_true is not None:
            AR_file.write('\n')
            for AR in ARs_true:
                AR_file.write(f'{AR} ')

    if graphic:
        plt.figure()
        plt.plot(np.arange(len(ARs)) + 1, ARs, label='Predicted AR')
        if ARs_true is not None:
            plt.plot(np.arange(len(ARs_true)) + 1, ARs_true, label='True AR')
        plt.legend()
        plt.ylim((np.min(ARs) - 0.2, np.max(ARs) + 0.2))
        plt.xlabel('Frame number')
        plt.ylabel('Estimated Aspect Ratio')
        plt.savefig(f'{name}.png')
        plt.close()

    print('done!')


def draw_config(phi, list_add, list_remove):
    """
    Draw shapes from list_add and list_remove into phi.val.
    """
    print('Drawing configuration from list definition... ', end='')
    for shape in list_add:
        phi.paint_shape(shape, filler_value=1.0)
    for shape in list_remove:
        phi.paint_shape(shape, filler_value=0.0)
    print('done!')


def save_vtk(array, path):
    data = pv.wrap(array)
    data.save(path, binary=True)


def seq2vtk(y_pred, path, start_snap=0):
    """
    Save sequence as vtk snapshots.
    For 3D: y_pred ndim == 6 (B,T,C,D,H,W).
    For 2D: y_pred ndim == 5 (B,T,C,H,W) -> add fake depth.
    """
    if y_pred.ndim == 6:
        for kk in range(y_pred.shape[1]):
            snap = y_pred[0, kk, 0, ...].detach().cpu().numpy()
            save_vtk(snap, path=f'{path}/snap_{kk + start_snap}.vtk')
    elif y_pred.ndim == 5:
        for kk in range(y_pred.shape[1]):
            snap = y_pred[0, kk, 0, ...].detach().cpu().numpy()
            snap = np.expand_dims(snap, 0)
            save_vtk(snap, path=f'{path}/snap_{kk + start_snap}.vtk')
    else:
        raise ValueError(
            f'{y_pred.ndim} dimensional data has been passed for vtk output. '
            'Only 5D (area/time) and 6D (volumetric/time) data should be considered.'
        )


def seq2npy(y_pred, path, start_snap=0):
    """
    Save sequence frames as .npy files.
    """
    if y_pred.ndim not in (5, 6):
        raise ValueError(
            f'{y_pred.ndim} dimensional data has been passed for npy output. '
            'Only 5D (area/time) and 6D (volumetric/time) data should be considered.'
        )

    for kk in range(y_pred.shape[1]):
        snap = y_pred[0, kk, 0, ...].detach().cpu().numpy()
        np.save(f'{path}/snap_{kk + start_snap}.npy', snap)