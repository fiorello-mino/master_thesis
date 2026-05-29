# <<< import external stuff <<<
import torch
import torch.nn as nn

import torch.nn.utils.parametrize as parametrize

import torch.nn.functional as f

from torchvision import utils, datasets, transforms

import PIL
from PIL import Image

import numpy as np

import matplotlib.pyplot as plt
import matplotlib

from torch.fft import fft2, ifft2, rfft2, irfft2
# --- import external stuff ---

# <<< import my stuff <<<
from src.utils import make_square
# --- import my stuff

# <<< import numba <<<
try:
    from numba import njit, prange
except ImportError:
    def njit(fun): # <- alternative definition of njit
        return fun
    def prange(x): # <- alternative definition of prange
        return range(x)
# --- import numba ---

# <<< NN-related classes <<<
class ConvGRUCell_parallel(nn.Module):
    '''
    This class is the same as the ConvGRUCell class. However, gates act in a parallel manner
    (e.g. ResetInput, UpdateInput and CandidateInput are performed using a single Conv2d).
    '''
    def __init__(self, in_channels, hidden_channels, kernel_size, padding_mode, separable=False, bias=True, legacy=False):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.padding_mode = padding_mode

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        self.bias = bias
        self.legacy = legacy

        self.i2h = nn.Sequential(
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=3 * self.hidden_channels,
                kernel_size=self.kernel_size,
                stride=1,
                padding=self.kernel_size // 2,
                padding_mode=self.padding_mode,
                bias=self.bias
            )
        )

        if not self.legacy:
            self.i2h.add_module(
                'extra_non_linear',
                nn.Sequential(
                    nn.Tanh(),
                    nn.Conv2d(
                        in_channels=3 * self.hidden_channels,
                        out_channels=3 * self.hidden_channels,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        padding_mode=self.padding_mode,
                        bias=self.bias
                    )
                )
            )

        self.h2h = nn.Sequential(
            nn.Conv2d(
                in_channels=self.hidden_channels,
                out_channels=2 * self.hidden_channels,
                kernel_size=self.kernel_size,
                stride=1,
                padding=self.kernel_size // 2,
                padding_mode=self.padding_mode,
                bias=self.bias
            )
        )

        if not self.legacy:
            self.h2h.add_module(
                'extra_non_linear',
                nn.Sequential(
                    nn.Tanh(),
                    nn.Conv2d(
                        in_channels=2 * self.hidden_channels,
                        out_channels=2 * self.hidden_channels,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        padding_mode=self.padding_mode,
                        bias=self.bias
                    )
                )
            )

        self.h2candidate = nn.Sequential(
            nn.Conv2d(
                in_channels=self.hidden_channels,
                out_channels=self.hidden_channels,
                kernel_size=self.kernel_size,
                stride=1,
                padding=self.kernel_size // 2,
                padding_mode=self.padding_mode,
                bias=self.bias
            )
        )

        if not self.legacy:
            self.h2candidate.add_module(
                'extra_non_linear',
                nn.Sequential(
                    nn.Tanh(),
                    nn.Conv2d(
                        in_channels=self.hidden_channels,
                        out_channels=self.hidden_channels,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        padding_mode=self.padding_mode,
                        bias=self.bias
                    )
                )
            )

        if separable:
            raise NotImplementedError('Separable convolution not implemented in parallel ConvGRU cell.')

    def forward(self, x, H):
        reset_i, update_i, candidate_i = self.i2h(x).split(self.hidden_channels, dim=1)
        reset_h, update_h = self.h2h(H).split(self.hidden_channels, dim=1)

        reset = self.sigmoid(reset_i + reset_h)
        update = self.sigmoid(update_i + update_h)
        candidate = self.tanh(candidate_i + self.h2candidate(reset * H))

        return (1 - update) * candidate + update * H

    def symmetrize(self):
        raise UserWarning('The symmetrization of the convolutional layers has not been properly tested... use at your own risk.')
        # If you want to enable this later, you must register on the actual Conv layers,
        # not on the Sequential containers.
        # Example:
        # parametrize.register_parametrization(self.i2h[0], 'weight', BiSymmetric())
        # parametrize.register_parametrization(self.h2h[0], 'weight', BiSymmetric())
        # parametrize.register_parametrization(self.h2candidate[0], 'weight', BiSymmetric())


class BiSymmetric(nn.Module):
    def forward(self, X):
        a = X + X.transpose(-1, -2)
        b = torch.rot90(a, k=2, dims=(-1, -2))
        return a + b


class ConvGRUCell_parallel_3D(ConvGRUCell_parallel):
    '''
    This subclass implements the convGRU with 3D convolutions
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.i2h = nn.Sequential(
            nn.Conv3d(
                in_channels=self.in_channels,
                out_channels=3 * self.hidden_channels,
                kernel_size=self.kernel_size,
                stride=1,
                padding=self.kernel_size // 2,
                padding_mode=self.padding_mode,
                bias=self.bias
            ),
            nn.Tanh(),
            nn.Conv3d(
                in_channels=3 * self.hidden_channels,
                out_channels=3 * self.hidden_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                padding_mode=self.padding_mode,
                bias=self.bias
            )
        )

        self.h2h = nn.Sequential(
            nn.Conv3d(
                in_channels=self.hidden_channels,
                out_channels=2 * self.hidden_channels,
                kernel_size=self.kernel_size,
                stride=1,
                padding=self.kernel_size // 2,
                padding_mode=self.padding_mode,
                bias=self.bias
            ),
            nn.Tanh(),
            nn.Conv3d(
                in_channels=2 * self.hidden_channels,
                out_channels=2 * self.hidden_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                padding_mode=self.padding_mode,
                bias=self.bias
            )
        )

        self.h2candidate = nn.Sequential(
            nn.Conv3d(
                in_channels=self.hidden_channels,
                out_channels=self.hidden_channels,
                kernel_size=self.kernel_size,
                stride=1,
                padding=self.kernel_size // 2,
                padding_mode=self.padding_mode,
                bias=self.bias
            ),
            nn.Tanh(),
            nn.Conv3d(
                in_channels=self.hidden_channels,
                out_channels=self.hidden_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                padding_mode=self.padding_mode,
                bias=self.bias
            )
        )

    def symmetrize(self):
        raise NotImplementedError('Symmetrization is not implemented for 3D convolutions')


class ConvGRU(nn.Module):
    '''
    This class defines a module stacking multiple ConvGRU together.
    The module also provides an additional output layer producing an output image/field.
    '''

    def __init__(
        self,
        hidden_units: int,
        input_channels: int,
        output_channels: int,
        hidden_channels: int,
        kernel_size: int,
        padding_mode: str,
        separable: bool = False,
        reduce_out: bool = True,
        squash_out: bool = True,
        bias: bool = True,
        divergence: bool = True,
        conservative: bool = True,
        num_params: int = 0,
        dropout: bool = False,
        dropout_prob: float | None = None
    ):
        super().__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.hidden_units = hidden_units
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.padding_mode = padding_mode

        self.num_params = num_params
        self.separable = separable

        self.reduce_out = reduce_out
        self.squash_out = squash_out

        self.dropout = dropout
        self.dropout_prob = dropout_prob

        self.bias = bias
        self.div_mode = divergence
        self.conservative = conservative

        self.GRU_list = nn.ModuleList()
        self.divergence_filters = None

        if self.separable:
            raise NotImplementedError('Separable convolution is not implemented in ConvGRU.')

        out_channels = self.output_channels if not self.div_mode else 2 * self.output_channels

        self.toOut = nn.Sequential(
            nn.Conv2d(
                in_channels=self.hidden_channels,
                out_channels=self.hidden_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                padding_mode=self.padding_mode,
                bias=self.bias
            ),
            nn.Tanh(),
            nn.Conv2d(
                in_channels=self.hidden_channels,
                out_channels=self.hidden_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                padding_mode=self.padding_mode,
                bias=self.bias
            ),
            nn.Tanh(),
            nn.Conv2d(
                in_channels=self.hidden_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode=self.padding_mode,
                bias=self.bias
            )
        )

        self.sigmoid = nn.Sigmoid()

        for kk in range(self.hidden_units):
            if kk == 0:
                in_channels = self.input_channels + self.num_params
            else:
                in_channels = self.hidden_channels

            self.GRU_list.append(
                ConvGRUCell_parallel(
                    in_channels=in_channels,
                    hidden_channels=self.hidden_channels,
                    kernel_size=self.kernel_size,
                    padding_mode=self.padding_mode,
                    separable=self.separable,
                    bias=self.bias
                )
            )

    def make_div_filters(self, x):
        '''
        This method constructs the divergence filters
        '''
        print('Constructing differential operators as filters...', end='')

        grad1 = nn.Conv2d(
            1, 1, kernel_size=3, stride=1, padding=1,
            bias=False, padding_mode=self.padding_mode
        )
        grad2 = nn.Conv2d(
            1, 1, kernel_size=3, stride=1, padding=1,
            bias=False, padding_mode=self.padding_mode
        )

        gradx_matrix = np.array([[0, 0, 0], [1, 0, -1], [0, 0, 0]], dtype=np.float32)
        grady_matrix = np.array([[0, 1, 0], [0, 0, 0], [0, -1, 0]], dtype=np.float32)

        grad1.weight = nn.Parameter(torch.from_numpy(gradx_matrix).unsqueeze(0).unsqueeze(0), requires_grad=False)
        grad2.weight = nn.Parameter(torch.from_numpy(grady_matrix).unsqueeze(0).unsqueeze(0), requires_grad=False)

        grad1 = grad1.to(x.device)
        grad2 = grad2.to(x.device)

        self.divergence_filters = [grad1, grad2]

        print('DONE!')

    def divergence(self, x):
        '''
        This method calculates the divergence of the given field using finite differences approximation
        '''
        if self.divergence_filters is None:
            self.make_div_filters(x)

        Jx, Jy = torch.split(x, 1, dim=1)

        Jx = Jx - torch.mean(Jx, dim=(-1, -2), keepdim=True)
        Jy = Jy - torch.mean(Jy, dim=(-1, -2), keepdim=True)

        gradx = self.divergence_filters[0](Jx)
        grady = self.divergence_filters[1](Jy)

        return gradx + grady

    def symmetrize(self):
        for GRU_cell in self.GRU_list:
            GRU_cell.symmetrize()

    def make_dropout_list(self, in_sequence, approx_inference):
        """
        Build a dropout mask of shape:
        (batch, hidden_units, hidden_channels)
        """
        if not self.dropout:
            self.dropoutmask = None
            return

        if self.dropout_prob is None:
            raise ValueError("dropout_prob must be set when dropout=True")

        if not (0.0 <= self.dropout_prob < 1.0):
            raise ValueError("dropout_prob must satisfy 0 <= dropout_prob < 1")

        keep_prob = 1.0 - self.dropout_prob
        shape = (in_sequence.shape[0], self.hidden_units, self.hidden_channels)

        if approx_inference:
            self.dropoutmask = torch.ones(
                shape,
                device=in_sequence.device,
                dtype=in_sequence.dtype
            )
        else:
            self.dropoutmask = (
                (torch.rand(shape, device=in_sequence.device) < keep_prob)
                .to(in_sequence.dtype) / keep_prob
            )

    def forward_old(self, in_sequence, future=0, params=None, noise_reg=0.0, approx_inference=True):
        '''
        This method is called from forward if you are not in divergence mode
        '''
        if self.dropout:
            self.make_dropout_list(in_sequence, approx_inference)

        outputs = []
        hidden_list = []

        device = in_sequence.device

        for _ in range(self.hidden_units):
            hidden_list.append(
                torch.zeros(
                    in_sequence.size(0),
                    self.hidden_channels,
                    in_sequence.size(3),
                    in_sequence.size(4),
                    device=device,
                    requires_grad=False
                )
            )

        for input_t in in_sequence.split(1, dim=1):
            input_t_old = input_t

            if noise_reg != 0:
                noise = noise_reg * torch.randn_like(input_t)
                if self.conservative:
                    noise = noise - torch.mean(noise, dim=(-1, -2), keepdim=True)
                input_t = input_t + noise

            input_t = self.cat_params(input_t, params)

            for kk in range(self.hidden_units):
                if kk == 0:
                    hidden_list[kk] = self.GRU_list[kk](input_t.squeeze(1), hidden_list[kk])
                else:
                    hidden_list[kk] = self.GRU_list[kk](hidden_list[kk - 1], hidden_list[kk])

                if self.dropout:
                    mask = self.dropoutmask[:, kk, :, None, None]
                    hidden_list[kk] = hidden_list[kk] * mask

            if self.reduce_out:
                output = self.toOut(hidden_list[-1])
                if self.squash_out:
                    if self.conservative:
                        output = output - torch.mean(output, dim=(-1, -2), keepdim=True)
                    output = input_t_old.squeeze(1) + output
            else:
                output = hidden_list[-1]

            outputs.append(output)

        for _ in range(future):
            output_old = output

            if noise_reg != 0:
                noise = noise_reg * torch.randn_like(output)
                if self.conservative:
                    noise = noise - torch.mean(noise, dim=(-1, -2), keepdim=True)
                output = output + noise

            output_with_params = self.cat_params(output, params)

            for kk in range(self.hidden_units):
                if kk == 0:
                    hidden_list[kk] = self.GRU_list[kk](output_with_params, hidden_list[kk])
                else:
                    hidden_list[kk] = self.GRU_list[kk](hidden_list[kk - 1], hidden_list[kk])

                if self.dropout:
                    mask = self.dropoutmask[:, kk, :, None, None]
                    hidden_list[kk] = hidden_list[kk] * mask

            if self.reduce_out:
                output = self.toOut(hidden_list[-1])
                if self.squash_out:
                    if self.conservative:
                        output = output - torch.mean(output, dim=(-1, -2), keepdim=True)
                    output = output_old + output
            else:
                output = hidden_list[-1]

            outputs.append(output)

        outputs = torch.stack(outputs, dim=1)
        return outputs

    def forward_div(self, in_sequence, future=0, params=None, noise_reg=0.0, approx_inference=True):
        '''
        This method is called in divergence mode; BETA
        '''
        if self.dropout:
            self.make_dropout_list(in_sequence, approx_inference)

        outputs = []
        hidden_list = []

        device = in_sequence.device

        for _ in range(self.hidden_units):
            hidden_list.append(
                torch.zeros(
                    in_sequence.size(0),
                    self.hidden_channels,
                    in_sequence.size(3),
                    in_sequence.size(4),
                    device=device,
                    requires_grad=False
                )
            )

        for input_t in in_sequence.split(1, dim=1):
            input_t_old = input_t
            input_t = self.cat_params(input_t, params)

            for kk in range(self.hidden_units):
                if kk == 0:
                    hidden_list[kk] = self.GRU_list[kk](input_t.squeeze(1), hidden_list[kk])
                else:
                    hidden_list[kk] = self.GRU_list[kk](hidden_list[kk - 1], hidden_list[kk])

                if self.dropout:
                    mask = self.dropoutmask[:, kk, :, None, None]
                    hidden_list[kk] = hidden_list[kk] * mask

            if self.reduce_out:
                output = self.toOut(hidden_list[-1])
                if self.squash_out:
                    output = input_t_old.squeeze(1) + self.divergence(output)
                else:
                    output = self.divergence(output)
            else:
                output = input_t_old.squeeze(1) + self.divergence(hidden_list[-1])

            outputs.append(output)

            if noise_reg != 0:
                noise = noise_reg * torch.randn_like(output)
                noise = noise - torch.mean(noise, dim=(-1, -2), keepdim=True)
                output = output + noise

        for _ in range(future):
            output_old = output
            output = self.cat_params(output, params)

            for kk in range(self.hidden_units):
                if kk == 0:
                    hidden_list[kk] = self.GRU_list[kk](output, hidden_list[kk])
                else:
                    hidden_list[kk] = self.GRU_list[kk](hidden_list[kk - 1], hidden_list[kk])

                if self.dropout:
                    mask = self.dropoutmask[:, kk, :, None, None]
                    hidden_list[kk] = hidden_list[kk] * mask

            if self.reduce_out:
                output = self.toOut(hidden_list[-1])
                if self.squash_out:
                    output = output_old + self.divergence(output)
                else:
                    output = self.divergence(output)
            else:
                output = output_old + self.divergence(hidden_list[-1])

            outputs.append(output)

            if noise_reg != 0:
                noise = noise_reg * torch.randn_like(output)
                noise = noise - torch.mean(noise, dim=(-1, -2), keepdim=True)
                output = output + noise

        outputs = torch.stack(outputs, dim=1)
        return outputs

    def cat_params(self, in_sequence, params):
        '''
        This method extends the parameter tensor with external parameters
        '''
        if params is None:
            return in_sequence

        num_params_from_loader = len(params)
        if num_params_from_loader != self.num_params:
            raise ValueError(
                'The number of parameters provided by the dataloader is not consistent with the one in the NN model.'
            )

        shapes = list(in_sequence.shape)
        shapes[-3] = shapes[-3] + self.num_params
        in_sequence = in_sequence.expand(shapes).clone()

        for cc, params_batch in enumerate(params):
            for bb, param in enumerate(params_batch):
                value = param.to(device=in_sequence.device, dtype=in_sequence.dtype)
                if len(shapes) == 5:
                    in_sequence[bb, :, -(cc + 1), :, :] = value
                else:
                    in_sequence[bb, -(cc + 1), :, :] = value

        return in_sequence

    def forward(self, in_sequence, future=0, params=None, noise_reg=0.0, approx_inference=True):
        if self.div_mode:
            return self.forward_div(
                in_sequence,
                future,
                params=params,
                noise_reg=noise_reg,
                approx_inference=approx_inference
            )
        else:
            return self.forward_old(
                in_sequence,
                future,
                params=params,
                noise_reg=noise_reg,
                approx_inference=approx_inference
            )

class ConvGRU3D(nn.Module):
    '''
    This class implements the 3D version of the ConvGRU
    '''

    def __init__(
        self,
        hidden_units,
        input_channels,
        hidden_channels,
        kernel_size,
        padding_mode,
        bias=True,
        divergence=True,
        separable=False,
        num_params=0,
        dropout=False,
        dropout_prob=None,
        output_channels=None
    ):
        '''
        Constructor method
        '''
        super().__init__()

        if separable:
            raise NotImplementedError('Separable convolution is not implemented in ConvGRU3D yet...')

        self.input_channels = input_channels
        self.hidden_units = hidden_units
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.padding_mode = padding_mode

        self.num_params = num_params
        self.output_channels = input_channels if output_channels is None else output_channels

        self.GRU_list = nn.ModuleList()

        self.bias = bias
        self.div_mode = divergence

        self.dropout = dropout
        self.dropout_prob = dropout_prob

        self.divergence_filters = None

        out_channels = self.output_channels if not self.div_mode else 3 * self.output_channels

        self.toOut = nn.Sequential(
            nn.Conv3d(
                in_channels=self.hidden_channels,
                out_channels=self.hidden_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                padding_mode=self.padding_mode,
                bias=self.bias
            ),
            nn.Tanh(),
            nn.Conv3d(
                in_channels=self.hidden_channels,
                out_channels=self.hidden_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                padding_mode=self.padding_mode,
                bias=self.bias
            ),
            nn.Tanh(),
            nn.Conv3d(
                in_channels=self.hidden_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode=self.padding_mode,
                bias=self.bias
            )
        )

        self.sigmoid = nn.Sigmoid()

        for kk in range(self.hidden_units):
            if kk == 0:
                in_channels = self.input_channels + self.num_params
            else:
                in_channels = self.hidden_channels

            self.GRU_list.append(
                ConvGRUCell_parallel_3D(
                    in_channels=in_channels,
                    hidden_channels=self.hidden_channels,
                    kernel_size=self.kernel_size,
                    padding_mode=self.padding_mode,
                    bias=self.bias
                )
            )

    def make_div_filters(self, x):
        '''
        This method constructs the divergence filters
        '''
        print('Constructing differential operators as filters...', end='')

        grad1 = nn.Conv3d(1, 1, kernel_size=3, stride=1, padding=1, bias=False, padding_mode=self.padding_mode)
        grad2 = nn.Conv3d(1, 1, kernel_size=3, stride=1, padding=1, bias=False, padding_mode=self.padding_mode)
        grad3 = nn.Conv3d(1, 1, kernel_size=3, stride=1, padding=1, bias=False, padding_mode=self.padding_mode)

        gradx_matrix = np.zeros((3, 3, 3), dtype=np.float32)
        grady_matrix = np.zeros((3, 3, 3), dtype=np.float32)
        gradz_matrix = np.zeros((3, 3, 3), dtype=np.float32)

        gradx_matrix[0, 1, 1] = 1.
        gradx_matrix[2, 1, 1] = -1.

        grady_matrix[1, 0, 1] = 1.
        grady_matrix[1, 2, 1] = -1.

        gradz_matrix[1, 1, 0] = 1.
        gradz_matrix[1, 1, 2] = -1.

        grad1.weight = nn.Parameter(
            torch.from_numpy(gradx_matrix).unsqueeze(0).unsqueeze(0),
            requires_grad=False
        )
        grad2.weight = nn.Parameter(
            torch.from_numpy(grady_matrix).unsqueeze(0).unsqueeze(0),
            requires_grad=False
        )
        grad3.weight = nn.Parameter(
            torch.from_numpy(gradz_matrix).unsqueeze(0).unsqueeze(0),
            requires_grad=False
        )

        grad1 = grad1.to(x.device)
        grad2 = grad2.to(x.device)
        grad3 = grad3.to(x.device)

        self.divergence_filters = [grad1, grad2, grad3]

        print('DONE!')

    def divergence(self, x):
        '''
        This method calculates the divergence of the given field using finite differences approximation
        '''
        if self.divergence_filters is None:
            self.make_div_filters(x)

        Jx, Jy, Jz = torch.split(x, 1, dim=1)

        Jx = Jx - torch.mean(Jx, dim=(-1, -2, -3), keepdim=True)
        Jy = Jy - torch.mean(Jy, dim=(-1, -2, -3), keepdim=True)
        Jz = Jz - torch.mean(Jz, dim=(-1, -2, -3), keepdim=True)

        gradx = self.divergence_filters[0](Jx)
        grady = self.divergence_filters[1](Jy)
        gradz = self.divergence_filters[2](Jz)

        return gradx + grady + gradz

    def cat_params(self, in_sequence, params):
        '''
        This method extends the parameter tensor with external parameters
        '''
        if params is None:
            return in_sequence

        num_params_from_loader = len(params)
        if num_params_from_loader != self.num_params:
            raise ValueError('The number of parameters provided by the dataloader is not consistent with the one in the NN model.')

        shapes = list(in_sequence.shape)

        if len(shapes) == 6:
            shapes[2] = shapes[2] + self.num_params
        elif len(shapes) == 5:
            shapes[1] = shapes[1] + self.num_params
        else:
            raise ValueError('Unsupported input shape in cat_params for ConvGRU3D.')

        in_sequence = in_sequence.expand(shapes).clone()

        for cc, params_batch in enumerate(params):
            for bb, param in enumerate(params_batch):
                value = param.to(device=in_sequence.device, dtype=in_sequence.dtype)
                if len(shapes) == 6:
                    in_sequence[bb, :, -(cc + 1), :, :, :] = value
                else:
                    in_sequence[bb, -(cc + 1), :, :, :] = value

        return in_sequence

    def make_dropout_list(self, in_sequence, approx_inference):
        """
        Build a dropout mask of shape:
        (batch, hidden_units, hidden_channels)
        """
        if not self.dropout:
            self.dropoutmask = None
            return

        if self.dropout_prob is None:
            raise ValueError("dropout_prob must be set when dropout=True")

        if not (0.0 <= self.dropout_prob < 1.0):
            raise ValueError("dropout_prob must satisfy 0 <= dropout_prob < 1")

        keep_prob = 1.0 - self.dropout_prob
        shape = (in_sequence.shape[0], self.hidden_units, self.hidden_channels)

        if approx_inference:
            self.dropoutmask = torch.ones(
                shape,
                device=in_sequence.device,
                dtype=in_sequence.dtype
            )
        else:
            self.dropoutmask = (
                (torch.rand(shape, device=in_sequence.device) < keep_prob)
                .to(in_sequence.dtype) / keep_prob
            )

    def forward_old(self, in_sequence, future=0, params=None, noise_reg=0.0, approx_inference=True):
        """This method is called from forward if you are not in divergence mode"""

        outputs = []
        hidden_list = []

        if self.dropout:
            self.make_dropout_list(in_sequence, approx_inference)

        device = in_sequence.device
        hidden_list_shape = list(in_sequence.shape)
        hidden_list_shape[2] = self.hidden_channels
        hidden_list_shape.pop(1)

        for _ in range(self.hidden_units):
            hidden_list.append(torch.zeros(hidden_list_shape, device=device, requires_grad=False))

        for input_t in in_sequence.split(1, dim=1):
            if noise_reg != 0:
                input_t = input_t + noise_reg * torch.randn_like(input_t)

            input_t = self.cat_params(input_t, params)

            for kk in range(self.hidden_units):
                if kk == 0:
                    hidden_list[kk] = self.GRU_list[kk](input_t.squeeze(1), hidden_list[kk])
                else:
                    hidden_list[kk] = self.GRU_list[kk](hidden_list[kk - 1], hidden_list[kk])

                if self.dropout:
                    mask = self.dropoutmask[:, kk, :, None, None, None]
                    hidden_list[kk] = hidden_list[kk] * mask

            output = self.toOut(hidden_list[-1])
            output = self.sigmoid(output)
            outputs.append(output)

        for _ in range(future):
            if noise_reg != 0:
                output = output + noise_reg * torch.randn_like(output)

            output_with_params = self.cat_params(output, params)

            for kk in range(self.hidden_units):
                if kk == 0:
                    hidden_list[kk] = self.GRU_list[kk](output_with_params, hidden_list[kk])
                else:
                    hidden_list[kk] = self.GRU_list[kk](hidden_list[kk - 1], hidden_list[kk])

                if self.dropout:
                    mask = self.dropoutmask[:, kk, :, None, None, None]
                    hidden_list[kk] = hidden_list[kk] * mask

            output = self.toOut(hidden_list[-1])
            output = self.sigmoid(output)
            outputs.append(output)

        outputs = torch.stack(outputs, dim=1)
        return outputs

    def forward_div(self, in_sequence, future=0, params=None, noise_reg=0.0, approx_inference=True):
        '''
        This method is called in divergence mode; BETA
        '''

        outputs = []
        hidden_list = []

        if self.dropout:
            self.make_dropout_list(in_sequence, approx_inference)

        device = in_sequence.device

        hidden_list_shape = list(in_sequence.shape)
        hidden_list_shape[2] = self.hidden_channels
        hidden_list_shape.pop(1)

        for _ in range(self.hidden_units):
            hidden_list.append(torch.zeros(hidden_list_shape, device=device, requires_grad=False))

        for input_t in in_sequence.split(1, dim=1):
            input_t_old = input_t.squeeze(1)
            input_t_with_params = self.cat_params(input_t, params)

            for kk in range(self.hidden_units):
                if kk == 0:
                    hidden_list[kk] = self.GRU_list[kk](input_t_with_params.squeeze(1), hidden_list[kk])
                else:
                    hidden_list[kk] = self.GRU_list[kk](hidden_list[kk - 1], hidden_list[kk])

                if self.dropout:
                    mask = self.dropoutmask[:, kk, :, None, None, None]
                    hidden_list[kk] = hidden_list[kk] * mask

            output = self.toOut(hidden_list[-1])
            output = input_t_old + self.divergence(output)
            outputs.append(output)

            if noise_reg != 0:
                noise = noise_reg * torch.randn_like(output)
                noise = noise - torch.mean(noise, dim=(-1, -2, -3), keepdim=True)
                output = output + noise

        for _ in range(future):
            output_old = output
            output_with_params = self.cat_params(output, params)

            for kk in range(self.hidden_units):
                if kk == 0:
                    hidden_list[kk] = self.GRU_list[kk](output_with_params, hidden_list[kk])
                else:
                    hidden_list[kk] = self.GRU_list[kk](hidden_list[kk - 1], hidden_list[kk])

                if self.dropout:
                    mask = self.dropoutmask[:, kk, :, None, None, None]
                    hidden_list[kk] = hidden_list[kk] * mask

            output = self.toOut(hidden_list[-1])
            output = output_old + self.divergence(output)
            outputs.append(output)

            if noise_reg != 0:
                noise = noise_reg * torch.randn_like(output)
                noise = noise - torch.mean(noise, dim=(-1, -2, -3), keepdim=True)
                output = output + noise

        outputs = torch.stack(outputs, dim=1)
        return outputs

    def forward(self, in_sequence, future=0, params=None, noise_reg=0.0, approx_inference=True):
        '''
        The forward method
        '''
        if self.div_mode:
            return self.forward_div(
                in_sequence,
                future,
                params=params,
                noise_reg=noise_reg,
                approx_inference=approx_inference
            )
        else:
            return self.forward_old(
                in_sequence,
                future,
                params=params,
                noise_reg=noise_reg,
                approx_inference=approx_inference
            )

class ConvGRUClassifier(ConvGRU):
    '''
    This class subclasses ConvGRU to obtain a classifier for video data
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.reduce_out = False  # we need to convert the hidden layer only at the end of the sequence

        self.toOut = nn.Sequential(
            nn.Conv2d(
                in_channels=self.hidden_channels,
                out_channels=2 * self.hidden_channels,
                kernel_size=2,
                stride=2,
                padding=0,
                padding_mode=self.padding_mode,
                bias=self.bias
            ),
            nn.Tanh(),
            nn.Conv2d(
                in_channels=2 * self.hidden_channels,
                out_channels=2 * self.hidden_channels,
                kernel_size=2,
                stride=2,
                padding=0,
                padding_mode=self.padding_mode,
                bias=self.bias
            ),
            nn.Tanh(),
            nn.MaxPool2d(2),
            nn.Conv2d(
                in_channels=2 * self.hidden_channels,
                out_channels=4 * self.hidden_channels,
                kernel_size=2,
                stride=2,
                padding=0,
                padding_mode=self.padding_mode,
                bias=self.bias
            ),
            nn.Tanh(),
            nn.MaxPool2d(2),
            nn.Conv2d(
                in_channels=4 * self.hidden_channels,
                out_channels=4 * self.hidden_channels,
                kernel_size=2,
                stride=2,
                padding=0,
                padding_mode=self.padding_mode,
                bias=self.bias
            ),
            nn.Tanh(),
            nn.Conv2d(
                in_channels=4 * self.hidden_channels,
                out_channels=self.output_channels,
                kernel_size=2,
                stride=2,
                padding=0,
                padding_mode=self.padding_mode,
                bias=self.bias
            )
        )

    def forward(self, in_sequence, future=0, params=None, noise_reg=0.0):
        GRU_result = self.forward_old(
            in_sequence,
            future,
            params=params,
            noise_reg=noise_reg
        )

        out = self.toOut(GRU_result[:, -1, :, :, :]).squeeze(-1).squeeze(-1)
        return out


class CommitteeModel:
    '''
    This class is a wrapper for individual models in order to have committee predictions
    and committee uncertainty estimation.
    '''
    def __init__(
        self,
        PATH_iter,
        device,
        hidden_units,
        input_channels,
        output_channels,
        hidden_channels,
        kernel_size,
        padding_mode,
        separable=False,
        reduce_out=True,
        squash_out=True,
        divergence=True,
        bias=False,
        num_params=0
    ):
        self.model_list = []
        self.device = device

        for path in PATH_iter:
            model = ConvGRU(
                hidden_units=hidden_units,
                input_channels=input_channels,
                output_channels=output_channels,
                hidden_channels=hidden_channels,
                kernel_size=kernel_size,
                padding_mode=padding_mode,
                separable=separable,
                reduce_out=reduce_out,
                squash_out=squash_out,
                divergence=divergence,
                bias=bias,
                num_params=num_params
            )

            model.load_state_dict(torch.load(path, map_location=self.device))
            model.eval()
            model.to(self.device)

            self.model_list.append(model)

        self.num_models = len(self.model_list)

    def __len__(self):
        return self.num_models

    def to(self, device):
        self.device = device
        for model in self.model_list:
            model.to(self.device)

    def pass2proc(self, x, future, model, queue):
        queue.put(model(x, future=future).detach().cpu().numpy())

    def __call__(self, x, future=0, params=None, scatter=False):
        out_list = []

        with torch.no_grad():
            for jj, model in enumerate(self.model_list, start=1):
                print(f'Model {jj} is predicting...', end='')
                out_list.append(model(x, future=future, params=params))
                print('done!')

        if scatter:
            return out_list
        return self.committee_decision(out_list)

    def eval(self):
        for model in self.model_list:
            model.eval()

    def committee_decision(self, data_list):
        with torch.no_grad():
            data_mean = torch.zeros_like(data_list[0])

            for val in data_list:
                data_mean += val
            data_mean /= len(data_list)

            if len(data_list) >= 2:
                data_var = torch.zeros_like(data_list[0])
                for val in data_list:
                    data_var += (val - data_mean) ** 2
                data_var /= (len(data_list) - 1)
                data_var = torch.sqrt(data_var)
            else:
                data_var = torch.full_like(data_mean, -1.0)

            return data_mean, data_var


# --- NN-related classes ---


# <<< phase field related classes <<<
class Phi:
    '''
    This class implements an abstraction of a phase-field object.
    It is able to add shapes and import images (possibly both)
    '''
    def __init__(self, res=(100, 100)):
        self.val = np.zeros(res)
        x = np.arange(0, self.val.shape[0])
        y = np.arange(0, self.val.shape[1])
        self.meshgrid = np.meshgrid(x, y)

    def set_center(self, x, y):
        self.center = (x, y)

    def paint_shape(self, shape_fun, filler_value):
        '''
        This method is a wrapper for the jitted function paint_shape_jitted.
        It "colors" with the given filler_value val inside Phi
        '''
        self.val = paint_shape_jitted(
            self.val,
            shape_fun,
            self.val.shape[0],
            self.val.shape[1],
            filler_value
        )

    def give_frame(self, rescale_down=1):
        '''
        This method returns the content of phi.val as a np array.
        Rescale down is used to "heal" sharp pixels in rounded shapes.
        '''
        image = PIL.Image.fromarray(self.val)
        image = image.resize(
            (self.val.shape[0] // rescale_down, self.val.shape[1] // rescale_down)
        )
        return np.asfarray(image)

    def plot(self, path, rescale_down=1, cmap='gray'):
        '''
        This method saves the content of self.val in the provided path.
        '''
        matplotlib.use('Agg')

        image = PIL.Image.fromarray(self.val)
        image = image.resize(
            (self.val.shape[0] // rescale_down, self.val.shape[1] // rescale_down)
        )

        val = np.asarray(image)

        plt.imshow(val, cmap=cmap, vmin=0, vmax=1)
        plt.tight_layout()
        plt.savefig(path)
        plt.close()

    def import_image(self, path, side, cropkey, crop_lim=(0.25, 0.75)):
        '''
        This method imports an external image into phi.val
        '''
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize(side),
            transforms.ToTensor()
        ])

        val = PIL.Image.open(path)
        val = make_square(val, cropkey=cropkey, crop_lim=crop_lim)
        val = transform(val)
        self.val = np.array(val[0])

    def phi4ConvGRU(self, rescale_down=1):
        '''
        This function returns self.val to an appropriate tensor for ConvGRU
        '''
        image = PIL.Image.fromarray(self.val)
        image = image.resize(
            (self.val.shape[0] // rescale_down, self.val.shape[1] // rescale_down)
        )

        val = np.asarray(image)

        tensor = torch.zeros(1, 1, 1, val.shape[0], val.shape[1])
        tensor[0, 0, 0, :, :] = torch.from_numpy(val).float()

        return tensor

    def dual(self):
        '''
        This method produces the dual image of self.val (1->0 and 0->1)
        '''
        self.val = -self.val + 1