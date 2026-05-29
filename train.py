# <<< import external stuff <<<
import torch
import torch.nn as nn
from torchvision import utils, datasets, transforms

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

import os
import sys

import numpy as np

import PIL
from PIL import Image

import time
# --- import external stuff ---


# <<< import my stuff <<<
from src.classes import ConvGRU, ConvGRU3D, ConvGRUClassifier
from src.utils import (
    build_train_logs_dir_tree,
    log_epoch_start_info,
    log_epoch_end_info,
    print_model_info,
    save_args,
    save_model,
    clip_series,
    withdual,
    out_png,
    out_gifs,
    seq2vtk,
    seq2npy,
    make_lossplot,
)
from src.dataloaders import give_dataloaders, give_3D_dataloaders
from src.parser import TrainingParser
# --- import my stuff ---


# <<< training function <<<
def train(model, loss_fn, optimizer, loaders, args):
    valid_losses = []
    train_losses = []
    trainloader, validloader = loaders
    len_trainloader = len(trainloader)
    len_validloader = len(validloader)

    use_amp = str(args.device).startswith("cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    for epoch in range(args.epochs):
        start_epoch = time.time()
        log_epoch_start_info(epoch, args)
        optimizer.zero_grad(set_to_none=True)
        epoch_train_losses = []
        model.train()

        # <<< training loop <<<
        for j, series_with_params in enumerate(trainloader):

            if args.num_params != 0:
                series, params = series_with_params
            else:
                series, params = series_with_params, None

            # debug break
            if j >= 1 and args.debug:
                print('Breaking because of DEBUG mode.')
                break

            # LR ramping in first epoch after reload
            if epoch == 0 and args.reload:
                for g in optimizer.param_groups:
                    temp_lr = ((j + 1) / (len_trainloader + 1)) * args.lr
                    g['lr'] = temp_lr
                print(f'Learning rate updated to: {temp_lr:.4e}')
            elif epoch == 1 and args.reload:
                for g in optimizer.param_groups:
                    g['lr'] = args.lr

            # choose input sequence length
            if args.ramp:
                in_seq_length = int(
                    args.subseq_max * (1 - (epoch + args.start_ramp) / args.ramp_length)
                )
                in_seq_length = min(series.shape[1] - 1, in_seq_length)
                in_seq_length = max(args.subseq_min, in_seq_length)
            else:
                in_seq_length = np.random.randint(args.subseq_min, args.subseq_max + 1)

            future = series.shape[1] - in_seq_length - 1

            if j % args.logfreq == 0 and not args.extract_param:
                print(
                    f'Passing example[{j}/{len_trainloader - 1}] in epoch {epoch} '
                    f'with {future} f-frames'
                )
            elif j % args.logfreq == 0:
                print(f'Passing example[{j}/{len_trainloader - 1}] in epoch {epoch}')

            inputdata = clip_series(series, in_seq_length).to(
                args.device, non_blocking=True
            )

            if not args.extract_param:
                targetdata = series[:, 1:, :, :, :].to(args.device, non_blocking=True)
                if args.dual:
                    inputdata = withdual(inputdata)
                    targetdata = withdual(targetdata)
                if params is not None:
                    for pp in range(len(params)):
                        params[pp] = torch.cat([params[pp], params[pp]])
            else:
                if args.dual:
                    inputdata = withdual(inputdata)
                    if params is not None:
                        for pp in range(len(params)):
                            params[pp] = torch.cat([params[pp], params[pp]])
                targetdata = torch.cat(
                    [p.unsqueeze(1) for p in params], dim=1
                ).to(args.device, non_blocking=True)

            targetdata = targetdata.float()

            with torch.cuda.amp.autocast(enabled=use_amp):
                if not args.extract_param:
                    ypred = model(
                        inputdata,
                        future=future,
                        params=params,
                        noisereg=args.noisereg,
                        approxinference=False,
                    )
                else:
                    ypred = model(
                        inputdata,
                        noisereg=args.noisereg,
                        approxinference=False,
                    )

                loss = loss_fn(ypred, targetdata)

            scaler.scale(loss).backward()

            if j % args.superbatch == 0 or j == len_trainloader - 1:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            loss4print = loss.item()
            epoch_train_losses.append(loss4print)

            if j % args.logfreq == 0:
                print(
                    f'Loss: {loss4print:.4e} \t '
                    f'Running mean loss: {np.mean(epoch_train_losses):.4e}'
                )

        train_losses.append(np.mean(epoch_train_losses))
        with open(args.paths["trainloss"], 'a+') as train_loss_file:
            train_loss_file.write(f'{train_losses[-1]}\n')
        # --- training loop ---

        # <<< validation loop <<<
        with torch.no_grad():
            model.eval()
            epoch_valid_losses = []

            y_preds = [] if args.extract_param else None
            y_trues = [] if args.extract_param else None

            for j, series_with_params in enumerate(validloader):
                if args.num_params != 0:
                    series, params = series_with_params
                else:
                    series, params = series_with_params, None

                if j == 3 and args.debug:
                    print('Breaking because of DEBUG mode.')
                    break

                in_seq_length = args.subseq_min
                future = series.shape[1] - in_seq_length - 1

                if not args.extract_param:
                    inputdata = clip_series(series, in_seq_length).to(
                        args.device, non_blocking=True
                    )
                    targetdata = series[:, 1:, :, :, :].to(
                        args.device, non_blocking=True
                    )
                    if args.dual:
                        inputdata = withdual(inputdata)
                        targetdata = withdual(targetdata)
                    if params is not None:
                        for pp in range(len(params)):
                            params[pp] = torch.cat([params[pp], params[pp]])
                else:
                    inputdata = series.to(args.device, non_blocking=True)
                    if args.dual:
                        inputdata = withdual(inputdata)
                        if params is not None:
                            for pp in range(len(params)):
                                params[pp] = torch.cat([params[pp], params[pp]])
                    targetdata = torch.cat(
                        [p.unsqueeze(1) for p in params], dim=1
                    ).to(args.device, non_blocking=True)

                targetdata = targetdata.float()

                with torch.cuda.amp.autocast(enabled=use_amp):
                    if not args.extract_param:
                        ypred = model(inputdata, future=future, params=params)
                    else:
                        ypred = model(inputdata)

                    loss = loss_fn(ypred, targetdata)

                if args.extract_param:
                    y_preds.append(ypred.detach().cpu())
                    y_trues.append(targetdata.detach().cpu())

                epoch_valid_losses.append(loss.item())

            valid_losses.append(np.mean(epoch_valid_losses))
            with open(args.paths["validloss"], 'a+') as valid_loss_file:
                valid_loss_file.write(f'{valid_losses[-1]}\n')

            optimizer.zero_grad(set_to_none=True)
        # --- validation loop ---

        # <<< graphic output <<<
        if args.graphics and not args.threeD and not args.extract_param:
            make_lossplot(train_losses, valid_losses, args)

            y_pred_cpu = ypred.detach().cpu()
            target_data_cpu = targetdata.detach().cpu()

            im_last_pred = y_pred_cpu[0, -1, :, :, :].permute(1, 2, 0)
            im_last_target = target_data_cpu[0, -1, :, :, :].permute(1, 2, 0)

            out_png(
                im_pred=im_last_pred,
                im_target=im_last_target,
                path=f'{args.paths["png"]}/epoch_{epoch}.png',
                cmap=args.cmap,
            )

            if args.gifs:
                out_gifs(
                    y_pred=y_pred_cpu,
                    path=f'{args.paths["gif"]}/epoch_{epoch}.gif',
                )

                out_gifs(
                    y_pred=target_data_cpu,
                    path=f'{args.paths["gif"]}/epoch_{epoch}_TRUE.gif',
                )

            if args.vtk:
                epoch_path = f'{args.paths["vtk"]}/epoch_{epoch}'
                epoch_path_TRUE = f'{args.paths["vtk"]}/epoch_{epoch}_TRUE'

                os.mkdir(epoch_path)
                os.mkdir(epoch_path_TRUE)

                seq2vtk(y_pred=y_pred_cpu, path=epoch_path)
                seq2vtk(y_pred=target_data_cpu, path=epoch_path_TRUE)

            if args.npy:
                epoch_path = f'{args.paths["npy"]}/epoch_{epoch}'
                epoch_path_TRUE = f'{args.paths["npy"]}/epoch_{epoch}_TRUE'

                os.mkdir(epoch_path)
                os.mkdir(epoch_path_TRUE)

                seq2npy(y_pred=y_pred_cpu, path=epoch_path)
                seq2npy(y_pred=target_data_cpu, path=epoch_path_TRUE)

        elif args.graphics and args.threeD and args.vtk:
            make_lossplot(train_losses, valid_losses, args)

            y_pred_cpu = ypred.detach().cpu()
            target_data_cpu = targetdata.detach().cpu()

            epoch_path = f'{args.paths["vtk"]}/epoch_{epoch}'
            epoch_path_TRUE = f'{args.paths["vtk"]}/epoch_{epoch}_TRUE'

            os.mkdir(epoch_path)
            os.mkdir(epoch_path_TRUE)

            seq2vtk(y_pred=y_pred_cpu, path=epoch_path)
            seq2vtk(y_pred=target_data_cpu, path=epoch_path_TRUE)

        elif args.graphics and args.extract_param:
            make_lossplot(train_losses, valid_losses, args)

            for kk in range(args.num_params):
                preds = []
                trues = []

                for y_pred_cpu, target_data_cpu in zip(y_preds, y_trues):
                    for bb in range(y_pred_cpu.shape[0]):
                        preds.append(y_pred_cpu[bb, kk])
                        trues.append(target_data_cpu[bb, kk])

                target_data_np = np.array(trues)
                y_pred_np = np.array(preds)

                plt.scatter(target_data_np, y_pred_np)
                _min, _max = np.min(target_data_np), np.max(target_data_np)
                plt.plot([_min, _max], [_min, _max])
                plt.title(f'Regression plot param {kk}')
                plt.xlabel('True value')
                plt.ylabel('Predicted value')
                plt.savefig(f'{args.paths["png"]}/epoch_{epoch}_param{kk}.png')
                plt.close()
        # --- graphic output ---

        # <<< epoch end logging <<<
        end_epoch = time.time()
        epoch_time = end_epoch - start_epoch

        if not args.extract_param:
            log_epoch_end_info(
                epoch,
                epoch_time,
                (ypred, targetdata),
                train_losses[-1],
                valid_losses[-1],
                args,
            )

        save_model(
            model=model,
            path=f'{args.paths["model"]}/epoch_{epoch}.pt',
        )
        # --- epoch end logging ---


# <<< main function <<<
def main():
    """
    Main function: instantiation of models and dataloaders and launching of training function
    """

    # Parse arguments
    parser = TrainingParser()
    args = parser.parse_args()

    # create folder structure
    args = build_train_logs_dir_tree(args)

    # Set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Instantiate dataloaders + pick model class
    if not args.threeD and not args.extract_param:
        dataloaders = give_dataloaders(args)
        model_class = ConvGRU
    elif args.extract_param:
        dataloaders = give_dataloaders(args)
        model_class = ConvGRUClassifier
    elif args.threeD and not args.extract_param:
        dataloaders = give_3D_dataloaders(args)
        model_class = ConvGRU3D
    else:
        raise NotImplementedError(
            'train.py was not able to recognize the training mode. Aborting.'
        )

    train_loader = dataloaders["train_set"]
    valid_loader = dataloaders["valid_set"]

    # Define model and put to device
    model = model_class(
        hidden_units=args.hidden,
        input_channels=1,  # hardcoded for now
        output_channels=1 if not args.extract_param else args.num_params,
        hidden_channels=args.channels,
        kernel_size=args.kernel_size,
        padding_mode=args.padding,
        separable=False,
        bias=args.bias,
        divergence=args.divergence,
        conservative=args.conservative,
        num_params=args.num_params if not args.extract_param else 0,
        dropout=args.dropout,
        dropout_prob=args.dropout_prob,
    )

    print_model_info(model)

    # Opzionale: i filtri di divergenza vengono creati lazy nel forward,
    # quindi non è strettamente necessario chiamare make_div_filters qui.

    model.to(args.device)

    if getattr(args, "compile", False):
        model = torch.compile(model)

    # save inputs
    save_args(args)

    # define optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weightd,
    )

    # define loss function
    if args.extract_param:
        loss_fn = nn.MSELoss()
    elif not args.threeD:
        loss_fn = lambda x, y: nn.MSELoss()(x, y) + args.massW * nn.MSELoss()(
            torch.mean(x, axis=(-1, -2)),
            torch.mean(y, axis=(-1, -2)),
        )
    else:
        loss_fn = lambda x, y: nn.MSELoss()(x, y) + args.massW * nn.MSELoss()(
            torch.mean(x, axis=(-1, -2, -3)),
            torch.mean(y, axis=(-1, -2, -3)),
        )

    # training loop
    train(model, loss_fn, optimizer, (train_loader, valid_loader), args)
# --- main function ---


# <<< main calling <<<
if __name__ == '__main__':
    main()
# --- main calling ---