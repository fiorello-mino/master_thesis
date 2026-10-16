"""2D adaptation of CRANE2/train.py.

Place beside CRANE2/src and run from the CRANE2 project directory.
Arrays use (B,T,C,Nx,Ny), with x on axis -2 and y on axis -1.
Loss/diagnostic operators use fixed PBC in x and homogeneous Neumann in y.
The existing ConvGRU, its --padding and its update rule are preserved.
--dx/--dy and --dt do not rescale the legacy model divergence update.
--massW is a legacy parser option; the objective matches TrainingLoss3D
(MSE + w_grad*gradient + w_energy*energy + w_bounds*bounds).
"""
# <<< import external stuff <<<
import torch
import torch.nn as nn
from torchvision import utils, datasets, transforms

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
import sys
from pathlib import Path

import numpy as np

import PIL
from PIL import Image

import time
import json
# --- import external stuff ---

# <<< import my stuff <<<
from src.classes import *
from src.utils import *
from src.dataloaders import give_dataloaders
from src.parser import TrainingParser
# --- import my stuff ---

class TrainingLoss2D(nn.Module):
    """MSE + gradient matching + energy matching + soft bounds.

    Shape: (B, T, C, Nx, Ny); x = axis -2, y = axis -1.
    Fixed mixed BC: PBC in x, replicated Neumann ghost values in y.
    Model convolutions use --padding.
    All terms are computed on every call, regardless of their weights.
    mu is a diagnostic in training and validation, excluded from the objective.
    """

    def __init__(self, args):
        super().__init__()
        self.w_grad = args.w_grad
        self.w_energy = args.w_energy
        self.w_bounds = args.w_bounds
        self.epsilon = args.epsilon
        self.dx, self.dy = args.dx, args.dy
        self.bounds_peak_weight = 0.05

    def neighbors(self, phi):
        if phi.ndim != 5 or min(phi.shape[-2:]) < 1:
            raise ValueError(f'Expected (B,T,C,Nx,Ny), got {tuple(phi.shape)}')
        xp = torch.roll(phi, -1, dims=-2)
        xm = torch.roll(phi, 1, dims=-2)
        # Homogeneous Neumann in y: replicate the boundary ghost values.
        yp = torch.cat((phi[..., 1:], phi[..., -1:]), dim=-1)
        ym = torch.cat((phi[..., :1], phi[..., :-1]), dim=-1)
        return xp, xm, yp, ym

    def gradient(self, phi):
        xp, xm, yp, ym = self.neighbors(phi)
        return (xp - xm) / (2*self.dx), (yp - ym) / (2*self.dy)

    def laplacian(self, phi):
        xp, xm, yp, ym = self.neighbors(phi)
        return ((xp - 2*phi + xm) / self.dx**2
                + (yp - 2*phi + ym) / self.dy**2)

    def W(self, phi):
        return (18.0 / self.epsilon) * phi.square() * (1.0 - phi).square()

    def mse_loss(self, pred, target):
        return nn.functional.mse_loss(pred, target)

    def energy_matching_loss(self, pred, target):
        return self.mse_loss(self.free_energy(pred), self.free_energy(target))

    def bounds_loss(self, pred):
        violation_sq = torch.relu(-pred).square() + torch.relu(pred - 1).square()
        return (violation_sq.mean()
                + self.bounds_peak_weight * violation_sq.amax(dim=(-2, -1)).mean())

    def free_energy(self, phi):
        gx, gy = self.gradient(phi)
        density = self.W(phi) + (self.epsilon / 2.0) * (gx.square() + gy.square())
        return density.sum(dim=(-2, -1)) * self.dx * self.dy

    def mu_matching_loss(self, pred, target):
        # Same variational-potential diagnostic as the supplied LossFunction.py.
        # This is q(phi), not the stabilized AMDiS chemical potential.
        def potential(phi):
            return ((36.0 / self.epsilon) * phi * (1.0 - phi)
                    * (1.0 - 2.0 * phi) - self.epsilon * self.laplacian(phi))

        return (potential(pred) - potential(target)).square().mean()

    def forward(self, pred, target, diagnostics=False):
        if pred.shape != target.shape:
            raise ValueError(f'Prediction {pred.shape} != target {target.shape}')
        # Always evaluate every term, even when its weight is zero.
        # diagnostics is retained for compatibility with existing callers.
        mse = self.mse_loss(pred, target)
        grad = sum((a - b).square().mean() for a, b in
                   zip(self.gradient(pred), self.gradient(target)))
        energy = self.energy_matching_loss(pred, target)
        bounds = self.bounds_loss(pred)
        # mu remains a diagnostic, as in the original training objective.
        with torch.no_grad():
            mu = self.mu_matching_loss(pred, target)

        # Weights affect only the objective, never whether a term is computed.
        # Exclude zero-weight contributions to avoid 0 * inf producing NaN.
        total = mse
        if self.w_grad:
            total = total + self.w_grad * grad
        if self.w_energy:
            total = total + self.w_energy * energy
        if self.w_bounds:
            total = total + self.w_bounds * bounds
        return total, {name: value.detach() for name, value in {
            'loss_total': total, 'loss_mse': mse, 'loss_grad': grad,
            'loss_energy': energy, 'loss_bounds': bounds, 'loss_mu': mu,
        }.items()}


def parse_training_args():
    """Extend the project's existing parser without editing src/parser.py."""
    parser = TrainingParser()
    cli = parser.parser

    def add_if_missing(flag, **kwargs):
        if flag not in cli._option_string_actions:
            cli.add_argument(flag, **kwargs)

    # Accept both spellings, including when src/parser.py defines one already.
    voxel_action = next((cli._option_string_actions[name]
                         for name in ('--voxel-size', '--voxel_size')
                         if name in cli._option_string_actions), None)
    if voxel_action is None:
        cli.add_argument('--voxel-size', '--voxel_size', dest='voxel_size',
                         type=float, default=None,
                         help='Isotropic pixel spacing; overrides dx,dy together (loss operators).')
    else:
        for name in ('--voxel-size', '--voxel_size'):
            if name not in cli._option_string_actions:
                voxel_action.option_strings.append(name)
                cli._option_string_actions[name] = voxel_action
    for flag, default in [('--dt', 0.005), ('--dx', 0.025), ('--dy', 0.025),
                          ('--epsilon', 0.1), ('--w_grad', 0.0),
                          ('--w_energy', 0.0), ('--w_bounds', 0.0), ('--w_mu', 0.0)]:
        add_if_missing(flag, type=float, default=default)
    # Physical defaults for this dataset. Explicit command-line values win.
    cli.set_defaults(dx=0.025, dy=0.025, dt=0.005, epsilon=0.1,
                     w_mu=0.0)
    raw_args = cli.parse_args()
    if raw_args.superbatch < 1:
        cli.error('--superbatch must be >= 1')
    args = parser.parse_args()
    # Some CRANE parser versions divide lr by superbatch. Gradients are now
    # averaged explicitly, so use exactly the learning rate requested by the user.
    args.lr = raw_args.lr
    if args.voxel_size is not None:
        args.dx = args.dy = args.voxel_size
    args.voxel_size = (args.dx, args.dy)
    import math
    for name in ('dx', 'dy', 'dt', 'epsilon', 'lr'):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            cli.error(f'{name} must be finite and positive')
    if args.subseq_min < 1 or args.subseq_max < args.subseq_min:
        cli.error('Require 1 <= subseq_min <= subseq_max')
    if args.logfreq < 1 or args.epochs < 1:
        cli.error('logfreq and epochs must be >= 1')
    if args.ramp and args.ramp_length < 1:
        cli.error('ramp_length must be >= 1')
    for name in ('w_grad', 'w_energy', 'w_bounds', 'w_mu', 'noise_reg'):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0:
            cli.error(f'{name} must be finite and nonnegative')
    if args.w_mu != 0:
        cli.error('This first-derivative-only training requires --w_mu 0')
    if args.threeD:
        cli.error('train_2D.py accepts only 2D datasets; omit --threeD')
    if args.extract_param and (args.w_grad or args.w_energy or args.w_bounds):
        cli.error('Physical loss weights require field prediction, not --extract_param')
    return args


def sequence_length(args, total_frames, epoch=None):
    """Choose a valid teacher-forcing length, reserving at least one target."""
    if total_frames <= args.subseq_min:
        raise ValueError('Each sequence needs more frames than subseq_min.')
    upper = min(args.subseq_max, total_frames - 1)
    if epoch is None:
        return args.subseq_min
    if args.ramp:
        length = int(args.subseq_max * (1 - (epoch + args.start_ramp) / args.ramp_length))
        return max(args.subseq_min, min(upper, length))
    return int(np.random.randint(args.subseq_min, upper + 1))


def optimizer_step_average(model, optimizer, sample_count):
    """Average accumulated sample-weighted losses, including partial groups."""
    for parameter in model.parameters():
        if parameter.grad is not None:
            parameter.grad.div_(sample_count)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def save_2d_config(model, args, path):
    """Record the existing 2D architecture and the loss's physical parameters."""
    config = {
        'architecture': 'ConvGRU',
        'update': 'legacy_centered_divergence' if model.div_mode else
                  ('conservative' if args.conservative else 'sigmoid'),
        'operator_bc': 'pbc_x_neumann_y',
        'loss_spacing': [args.dx, args.dy], 'epsilon': args.epsilon,
        'dt': args.dt,  # metadata only; existing 2D model does not use dt/dx/dy.
        'hidden_units': args.hidden, 'input_channels': 1,
        'hidden_channels': args.channels, 'kernel_size': args.kernel_size,
        'padding_mode': args.padding, 'bias': args.bias,
        'divergence': args.divergence, 'conservative': args.conservative,
        'num_params': args.num_params, 'dropout': args.dropout,
        'dropout_prob': args.dropout_prob,
    }
    with open(path, 'w') as stream:
        json.dump(config, stream, indent=2)


def make_lossplots(train_losses, valid_losses, args):
    """Save the existing linear plot and a second plot with logarithmic y."""
    make_lossplot(train_losses, valid_losses, args)
    linear_path = Path(args.paths['lossplot'])
    log_path = linear_path.with_name(linear_path.stem + '_log' + linear_path.suffix)
    fig, ax = plt.subplots()
    try:
        has_positive = False
        for values, label in ((train_losses, 'Training loss'),
                              (valid_losses, 'Validation loss')):
            values = np.asarray(values, dtype=float)
            # Zero/nonfinite values cannot be represented on a logarithmic axis.
            valid = np.isfinite(values) & (values > 0)
            has_positive = has_positive or bool(valid.any())
            ax.plot(np.arange(len(values)), np.where(valid, values, np.nan), label=label)
        ax.set_yscale('log')
        if not has_positive:
            ax.set_ylim(1e-12, 1.0)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, which='both', alpha=0.3)
        fig.tight_layout()
        fig.savefig(log_path)
    finally:
        plt.close(fig)


# <<< training function <<<
def train(model, loss_fn, optimizer, loaders, args):
    '''
    This function trains the model given selected loss function
    '''
    valid_losses = []

    valid_mse_losses = []

    valid_grad_losses = []
    valid_e_losses = []
    valid_mu_losses = []
    valid_bounds_losses = []

    valid_grad_ratios = []
    valid_e_ratios = []
    valid_mu_ratios = []
    valid_bounds_ratios = []

    valid_phi_mins = []
    valid_phi_maxs = []
    valid_oobs = []

    train_losses = []

    train_loader, valid_loader = loaders

    len_train_loader = len(train_loader)
    len_valid_loader = len(valid_loader)
    if not len_train_loader or not len_valid_loader:
        raise ValueError('Training and validation loaders must both be nonempty.')

    for epoch in range(args.epochs):

        start_epoch = time.time()
        log_epoch_start_info(epoch, args)

        optimizer.zero_grad()

        epoch_train_losses = []
        epoch_train_counts = []
        accumulated_samples = 0
        accumulated_batches = 0

        model.train()

        # <<< training loop <<<
        for j, series_with_params in enumerate(train_loader):

            if args.num_params != 0:
                series = series_with_params[0]
                params = list(series_with_params[1])
            else:
                series = series_with_params
                params = None

            # breaking if in debug mode
            if j >= 1 and args.debug:
                print('Breaking because of DEBUG mode.')
                break

            # first epoch in reloading has a ramping lr (this way Adam can re-recongnize slow and fast modes in loss function landscape)
            if epoch == 0 and args.reload:
                for g in optimizer.param_groups:
                    temp_lr = ((j + 1) / (len_train_loader + 1)) * args.lr
                    g['lr'] = temp_lr
                    print(f'Learning rate updated to: {temp_lr:.4e}')
            elif epoch == 1 and args.reload:
                for g in optimizer.param_groups:
                    g['lr'] = args.lr

            in_seq_length = sequence_length(args, series.shape[1], epoch)

            future = series.shape[1] - in_seq_length - 1

            if j % args.logfreq == 0 and not args.extract_param:  # <- print sub-epoch infos
                print(f'Passing example[{j}/{len_train_loader - 1}] in epoch {epoch} with {future} f-frames')
            elif j % args.logfreq == 0:
                print(f'Passing example[{j}/{len_train_loader - 1}] in epoch {epoch}')

            input_data = clip_series(series, in_seq_length).to(args.device)

            if not args.extract_param:
                target_data = series[:, 1:, :, :, :].to(args.device)

                if args.dual:
                    input_data = withdual(input_data)
                    target_data = withdual(target_data)
                    if params is not None:
                        for pp, param in enumerate(params):
                            params[pp] = torch.cat([params[pp], params[pp]])

            else:
                if args.dual:
                    input_data = withdual(input_data)
                    if params is not None:
                        for pp, param in enumerate(params):
                            params[pp] = torch.cat([params[pp], params[pp]])

                target_data = torch.cat([p.unsqueeze(1) for p in params], dim=1).to(args.device)
                target_data = target_data.float()

            if not args.extract_param:
                y_pred = model(input_data, future=future, params=params, noise_reg=args.noise_reg,
                               approx_inference=False)
            else:
                y_pred = model(input_data, noise_reg=args.noise_reg)

            if not args.extract_param:
                total_loss, _ = loss_fn(y_pred, target_data)
                loss = total_loss
            else:
                loss = loss_fn(y_pred, target_data)

            if not torch.isfinite(loss):
                raise FloatingPointError(f'Non-finite training loss at epoch {epoch}, batch {j}')
            sample_count = input_data.shape[0]
            (loss * sample_count).backward()
            accumulated_samples += sample_count
            accumulated_batches += 1
            if accumulated_batches == args.superbatch:
                optimizer_step_average(model, optimizer, accumulated_samples)
                accumulated_samples = accumulated_batches = 0

            loss4print = loss.item()

            epoch_train_losses.append(loss4print)
            epoch_train_counts.append(sample_count)

            if j % args.logfreq == 0:
                print(f'Loss: {loss4print:.4e} \t Running mean loss: {np.mean(epoch_train_losses):.4e}')

        if accumulated_batches:
            optimizer_step_average(model, optimizer, accumulated_samples)
        train_losses.append(np.average(epoch_train_losses, weights=epoch_train_counts))
        with open(f'{args.paths["trainloss"]}', 'a+') as train_loss_file:
            train_loss_file.write(f'{train_losses[-1]}\n')
        # --- training loop ---

        # <<< validation loop <<<
        with torch.no_grad():

            model.eval()

            epoch_valid_losses = []
            epoch_valid_counts = []
            epoch_mass_update_error = []
            epoch_mass_target_drift = []

            epoch_valid_mse_losses = []

            epoch_valid_grad_losses = []
            epoch_valid_e_losses = []
            epoch_valid_mu_losses = []
            epoch_valid_bounds_losses = []

            epoch_valid_grad_ratio = []
            epoch_valid_e_ratio = []
            epoch_valid_mu_ratio = []
            epoch_valid_bounds_ratio = []

            epoch_valid_phi_min = []
            epoch_valid_phi_max = []
            epoch_valid_oob = []

            y_preds = []
            y_trues = []

            for j, series_with_params in enumerate(valid_loader):

                if args.num_params != 0:
                    series = series_with_params[0]
                    params = list(series_with_params[1])
                else:
                    series = series_with_params
                    params = None

                if j >= 3 and args.debug:
                    print('Breaking because of DEBUG mode.')
                    break

                in_seq_length = sequence_length(args, series.shape[1])
                future = series.shape[1] - in_seq_length - 1

                if not args.extract_param:
                    input_data = clip_series(series, in_seq_length).to(args.device)
                    target_data = series[:, 1:, :, :, :].to(args.device)

                    if args.dual:
                        input_data = withdual(input_data)
                        target_data = withdual(target_data)
                        if params is not None:
                            for pp, param in enumerate(params):
                                params[pp] = torch.cat([params[pp], params[pp]])

                else:
                    input_data = series.to(args.device)

                    if args.dual:
                        input_data = withdual(input_data)
                        if params is not None:
                            for pp, param in enumerate(params):
                                params[pp] = torch.cat([params[pp], params[pp]])

                    target_data = torch.cat([p.unsqueeze(1) for p in params], dim=1).to(args.device)
                    target_data = target_data.float()

                if not args.extract_param:
                    y_pred = model(input_data, future=future, params=params)
                else:
                    y_pred = model(input_data)
                    y_preds.append(y_pred.detach().cpu())
                    y_trues.append(target_data.detach().cpu())

                if not args.extract_param:
                    total_loss, metrics = loss_fn(y_pred, target_data, diagnostics=True)
                    loss4print = total_loss.item()

                    # Float64 reduction makes this diagnostic sensitive to drift.
                    pred_mean = y_pred.mean(dim=(-2, -1), dtype=torch.float64)
                    input_mean = input_data.mean(dim=(-2, -1), dtype=torch.float64)
                    reference = torch.cat((input_mean, input_mean[:, -1:].expand(
                        -1, future, -1)), dim=1)
                    epoch_mass_update_error.append((pred_mean - reference).abs().max().item())
                    target_mean = target_data.mean(dim=(-2, -1), dtype=torch.float64)
                    epoch_mass_target_drift.append((target_mean - reference).abs().max().item())
                    mse_loss = metrics["loss_mse"]
                    grad_loss = metrics["loss_grad"]
                    e_loss = metrics["loss_energy"]
                    mu_loss = metrics["loss_mu"]
                    bounds_loss = metrics["loss_bounds"]

                    eps = 1e-12

                    grad_ratio = (
                            args.w_grad * grad_loss
                            / (mse_loss + eps)
                    )

                    e_ratio = (
                            args.w_energy * e_loss
                            / (mse_loss + eps)
                    )

                    mu_ratio = (
                            args.w_mu * mu_loss
                            / (mse_loss + eps)
                    )

                    bounds_ratio = (
                            args.w_bounds * bounds_loss
                            / (mse_loss + eps)
                    )

                    phi_min = y_pred.amin()
                    phi_max = y_pred.amax()

                    # Frazione dei pixel predetti fuori dall'intervallo fisico [0, 1]
                    oob_fraction = (
                            (y_pred < 0.0) | (y_pred > 1.0)
                    ).float().mean()

                    epoch_valid_losses.append(loss4print)

                    epoch_valid_mse_losses.append(mse_loss.cpu().item())
                    epoch_valid_grad_losses.append(grad_loss.cpu().item())
                    epoch_valid_e_losses.append(e_loss.cpu().item())
                    epoch_valid_mu_losses.append(mu_loss.cpu().item())
                    epoch_valid_bounds_losses.append(bounds_loss.cpu().item())

                    epoch_valid_grad_ratio.append(grad_ratio.cpu().item())
                    epoch_valid_e_ratio.append(e_ratio.cpu().item())
                    epoch_valid_mu_ratio.append(mu_ratio.cpu().item())
                    epoch_valid_bounds_ratio.append(bounds_ratio.cpu().item())

                    epoch_valid_phi_min.append(phi_min.cpu().item())
                    epoch_valid_phi_max.append(phi_max.cpu().item())
                    epoch_valid_oob.append(oob_fraction.cpu().item())
                else:
                    # MSE per la regressione dei parametri
                    loss = loss_fn(y_pred, target_data)
                    loss4print = loss.item()

                    epoch_valid_losses.append(loss4print)

                if not np.isfinite(loss4print):
                    raise FloatingPointError(f'Non-finite validation loss at epoch {epoch}, batch {j}')
                epoch_valid_counts.append(input_data.shape[0])

            valid_losses.append(np.average(epoch_valid_losses, weights=epoch_valid_counts))

            if not args.extract_param:
                valid_mse_losses.append(np.average(epoch_valid_mse_losses, weights=epoch_valid_counts))
                valid_grad_losses.append(np.average(epoch_valid_grad_losses, weights=epoch_valid_counts))
                valid_e_losses.append(np.average(epoch_valid_e_losses, weights=epoch_valid_counts))
                valid_mu_losses.append(np.average(epoch_valid_mu_losses, weights=epoch_valid_counts))
                valid_bounds_losses.append(np.average(epoch_valid_bounds_losses, weights=epoch_valid_counts))

                valid_grad_ratios.append(np.average(epoch_valid_grad_ratio, weights=epoch_valid_counts))
                valid_e_ratios.append(np.average(epoch_valid_e_ratio, weights=epoch_valid_counts))
                valid_mu_ratios.append(np.average(epoch_valid_mu_ratio, weights=epoch_valid_counts))
                valid_bounds_ratios.append(np.average(epoch_valid_bounds_ratio, weights=epoch_valid_counts))

                valid_phi_mins.append(np.min(epoch_valid_phi_min))
                valid_phi_maxs.append(np.max(epoch_valid_phi_max))
                valid_oobs.append(np.average(epoch_valid_oob, weights=epoch_valid_counts))

            with open(f'{args.paths["validloss"]}', 'a+') as valid_loss_file:
                valid_loss_file.write(f'{valid_losses[-1]}\n')

            if not args.extract_param:
                valid_loss_path = Path(args.paths["validloss"])
                valid_terms_path = valid_loss_path.with_name(
                    "valid_loss_terms.txt"
                )

                header = (
                    "# total\t"
                    "mse\t"
                    "grad\t"
                    "grad_ratio\t"
                    "energy\t"
                    "energy_ratio\t"
                    "mu\t"
                    "mu_ratio\t"
                    "bounds\t"
                    "bounds_ratio\t"
                    "phi_min\t"
                    "phi_max\t"
                    "oob_fraction\n"
                )

                with open(valid_terms_path, "a+") as f:
                    f.seek(0)
                    if f.read() == "":
                        f.write(header)
                    f.write(
                        f"{valid_losses[-1]:.6e}\t"
                        f"{valid_mse_losses[-1]:.6e}\t"
                        f"{valid_grad_losses[-1]:.6e}\t"
                        f"{valid_grad_ratios[-1]:.6e}\t"
                        f"{valid_e_losses[-1]:.6e}\t"
                        f"{valid_e_ratios[-1]:.6e}\t"
                        f"{valid_mu_losses[-1]:.6e}\t"
                        f"{valid_mu_ratios[-1]:.6e}\t"
                        f"{valid_bounds_losses[-1]:.6e}\t"
                        f"{valid_bounds_ratios[-1]:.6e}\t"
                        f"{valid_phi_mins[-1]:.6e}\t"
                        f"{valid_phi_maxs[-1]:.6e}\t"
                        f"{valid_oobs[-1]:.6e}\n"
                    )

            if not args.extract_param:
                mass_path = Path(args.paths['validloss']).with_name('valid_mass_2d.txt')
                new_file = not mass_path.exists()
                with open(mass_path, 'a') as stream:
                    if new_file:
                        stream.write('# epoch max_pred_mean_drift max_target_mean_drift\n')
                    stream.write(f'{epoch} {max(epoch_mass_update_error):.8e} '
                                 f'{max(epoch_mass_target_drift):.8e}\n')
                print(f'Max mean(phi) drift: prediction={max(epoch_mass_update_error):.3e}, '
                      f'target={max(epoch_mass_target_drift):.3e}')

        optimizer.zero_grad()  # <- better safe than sorry
        # --- validation loop ---

        # <<< graphic output <<<
        if args.graphics and not args.extract_param:

            y_pred_cpu = y_pred.detach().cpu()
            target_data_cpu = target_data.detach().cpu()

            im_last_pred = y_pred_cpu[0, -1, :, :, :].permute(1, 2, 0)
            im_last_target = target_data_cpu[0, -1, :, :, :].permute(1, 2, 0)

            out_png(
                im_pred=im_last_pred,
                im_target=im_last_target,
                path=f'{args.paths["png"]}/epoch_{epoch}.png',
                cmap=args.cmap
            )

            make_lossplots(train_losses, valid_losses, args)

            if args.gifs:
                out_gifs(
                    y_pred=y_pred_cpu,
                    path=f'{args.paths["gif"]}/epoch_{epoch}.gif'
                )

                out_gifs(
                    y_pred=target_data_cpu,
                    path=f'{args.paths["gif"]}/epoch_{epoch}_TRUE.gif'
                )

            if args.vtk:
                epoch_path = f'{args.paths["vtk"]}/epoch_{epoch}'
                epoch_path_TRUE = f'{args.paths["vtk"]}/epoch_{epoch}_TRUE'

                os.mkdir(epoch_path)
                os.mkdir(epoch_path_TRUE)

                seq2vtk(
                    y_pred=y_pred_cpu,
                    path=epoch_path
                )

                seq2vtk(
                    y_pred=target_data_cpu,
                    path=epoch_path_TRUE
                )

                del epoch_path, epoch_path_TRUE

            if args.npy:
                epoch_path = f'{args.paths["npy"]}/epoch_{epoch}'
                epoch_path_TRUE = f'{args.paths["npy"]}/epoch_{epoch}_TRUE'

                os.mkdir(epoch_path)
                os.mkdir(epoch_path_TRUE)

                seq2npy(
                    y_pred=y_pred_cpu,
                    path=epoch_path
                )

                seq2npy(
                    y_pred=target_data_cpu,
                    path=epoch_path_TRUE
                )

                del epoch_path, epoch_path_TRUE


        elif args.graphics and args.extract_param:

            make_lossplots(train_losses, valid_losses, args)

            for kk in range(args.num_params):

                preds = []
                trues = []

                for y_pred, target_data in zip(y_preds, y_trues):
                    for bb in range(y_pred.shape[0]):
                        preds.append(y_pred[bb, kk])
                        trues.append(target_data[bb, kk])

                target_data_cpu = np.array(trues)
                y_pred_cpu = np.array(preds)

                plt.scatter(target_data_cpu, y_pred_cpu)
                plt.plot(
                    [np.min(target_data_cpu), np.max(target_data_cpu)],
                    [np.min(target_data_cpu), np.max(target_data_cpu)]
                )
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
            log_epoch_end_info(epoch, epoch_time, (y_pred, target_data), train_losses[-1], valid_losses[-1], args)

        save_model(
            model=model,
            path=f'{args.paths["model"]}/epoch_{epoch}.pt'
        )
        if not args.extract_param:
            save_2d_config(model, args, Path(args.paths['model']) / f'epoch_{epoch}.json')
        # --- epoch end logging ---


# <<< main function <<<
def main():
    '''
    Main function: istantiation of models and dataloaders and launcing of training function
    '''

    # Parse arguments
    args = parse_training_args()

    # crate folder structure
    args = build_train_logs_dir_tree(args)

    # Set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Use the project's existing 2D data pipeline and architecture.
    dataloaders = give_dataloaders(args)
    model_class = ConvGRUClassifier if args.extract_param else ConvGRU

    train_loader = dataloaders["train_set"]
    valid_loader = dataloaders["valid_set"]

    # Define model and put to device
    model_kwargs = dict(
        hidden_units=args.hidden,
        input_channels=1,  # this is hardcoded for the moment... waiting for multidimensional data!
        output_channels=None if not args.extract_param else args.num_params,
        hidden_channels=args.channels,
        kernel_size=args.kernel_size,
        padding_mode=args.padding,
        separable=False,
        bias=args.bias,
        divergence=args.divergence,
        conservative=args.conservative,
        num_params=args.num_params if not args.extract_param else 0,
        dropout=args.dropout,
        dropout_prob=args.dropout_prob
    )

    model = model_class(**model_kwargs)
    print_model_info(model)

    if args.divergence and not args.extract_param:
        model.make_div_filters(torch.zeros(1, device=args.device))
        # The original helper sets a module attribute instead of freezing weights.
        for operator in model.divergence_filters:
            operator.requires_grad_(False)

    # model = torch.compile(model)

    # Reload operation
    if args.reload:
        model = import_model(model, args)

    if args.symm_kernel:
        model.symmetrize()

    model.to(args.device)

    # save inputs
    save_args(args)

    # define optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weightd
    )

    # Same physical objective as the supplied 3D script, integrated in 2D.
    loss_fn = nn.MSELoss() if args.extract_param else TrainingLoss2D(args).to(args.device)

    # training loop
    train(model, loss_fn, optimizer, (train_loader, valid_loader), args)


# --- main function ---

# <<< main calling <<<
if __name__ == '__main__':
    main()
# --- main calling ---
