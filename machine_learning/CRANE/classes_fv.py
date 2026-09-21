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
    This class is the same as the ConvGRUCell class. However, gates acts in a parallel manner (e.g. ResetInput, UpdateInput and CandidateInput are performed using a single Conv2d with groups=2)
    '''
    def __init__(self, in_channels, hidden_channels, kernel_size, padding_mode, separable=False, bias=True, legacy=False):
        
        super(ConvGRUCell_parallel, self).__init__()
        
        self.in_channels        = in_channels
        self.hidden_channels    = hidden_channels
        self.kernel_size        = kernel_size
        self.padding_mode       = padding_mode
        
        self.sigmoid            = nn.Sigmoid()
        self.tanh               = nn.Tanh()
        
        self.bias               = bias
        self.legacy             = legacy # flag controlling the presence of the "reaction diffusion trick"
        
        self.i2h                = nn.Sequential(
            nn.Conv2d(
                    in_channels     = self.in_channels,
                    out_channels    = 3*self.hidden_channels,
                    kernel_size     = self.kernel_size,
                    stride          = 1,
                    padding         = self.kernel_size//2,
                    padding_mode    = self.padding_mode,
                    bias            = self.bias
                    )
            )
        
        if not self.legacy:
            self.i2h.add_module( 'extra_non_linear',
                nn.Sequential(
                    nn.Tanh(),
                    nn.Conv2d(
                            in_channels     = 3*self.hidden_channels,
                            out_channels    = 3*self.hidden_channels,
                            kernel_size     = 1,
                            stride          = 1,
                            padding         = 0,
                            padding_mode    = self.padding_mode,
                            bias            = self.bias
                            )
                ) )
        
        self.h2h                = nn.Sequential(
            nn.Conv2d(
                    in_channels     = self.hidden_channels,
                    out_channels    = 2*self.hidden_channels,
                    kernel_size     = self.kernel_size,
                    stride          = 1,
                    padding         = self.kernel_size//2,
                    padding_mode    = self.padding_mode,
                    bias            = self.bias
                    )
            )
            
        if not self.legacy:
            self.h2h.add_module( 'extra_non_linear',
                nn.Sequential(
                    nn.Tanh(),
                    nn.Conv2d(
                        in_channels     = 2*self.hidden_channels,
                        out_channels    = 2*self.hidden_channels,
                        kernel_size     = 1,
                        stride          = 1,
                        padding         = 0,
                        padding_mode    = self.padding_mode,
                        bias            = self.bias
                        )
                    ) )
        
        self.h2candidate        = nn. Sequential(
            nn.Conv2d(
                    in_channels     = self.hidden_channels,
                    out_channels    = self.hidden_channels,
                    kernel_size     = self.kernel_size,
                    stride          = 1,
                    padding         = self.kernel_size//2,
                    padding_mode    = self.padding_mode,
                    bias            = self.bias
                    )
            )
            
        if not self.legacy:
            self.h2candidate.add_module( 'extra_non_linear',
                nn.Sequential(
                    nn.Tanh(),
                    nn.Conv2d(
                        in_channels     = self.hidden_channels,
                        out_channels    = self.hidden_channels,
                        kernel_size     = 1,
                        stride          = 1,
                        padding         = 0,
                        padding_mode    = self.padding_mode,
                        bias            = self.bias
                        )
                    )
                )
        
        
        if separable: raise NotImplementedError('Separable convolution not implemented in parallel ConvGRUC cell.')
        
    
    def forward(self, x, H):
        
        reset_i, update_i, candidate_i = self.i2h(x).split(self.hidden_channels, dim=1)
        reset_h, update_h = self.h2h(H).split(self.hidden_channels, dim=1)
        
        reset       = self.sigmoid(reset_i + reset_h)
        update      = self.sigmoid(update_i + update_h)
        candidate   = self.tanh(candidate_i + self.h2candidate(reset*H))
        
        return (1-update)*candidate + update*H
        
        
    def symmetrize(self):
        raise UserWarning('The symmetrization of the convolutional layer has not been properly tested... use at your own risk.')
        parametrize.register_parametrization(self.i2h, 'weight', BiSymmetric())
        parametrize.register_parametrization(self.h2h, 'weight', BiSymmetric())
        parametrize.register_parametrization(self.h2candidate, 'weight', BiSymmetric())


class BiSymmetric(nn.Module):
    def forward(self, X):
        a = X+X.transpose(-1,-2)
        b = torch.rot90(a, k=2, dims=(-1,-2))
        
        return a+b


class ConvGRUCell_parallel_3D(ConvGRUCell_parallel):
    '''
    This subclass implements the convGRU with 3D convolutions
    '''
    def __init__(self, *args, **kwargs):
        
        super().__init__(*args, **kwargs) # call the super generator
        
        # override the 2D convolutions with 3D ones
        self.i2h = nn.Sequential(
            nn.Conv3d(
                in_channels     = self.in_channels,
                out_channels    = 3*self.hidden_channels,
                kernel_size     = self.kernel_size,
                stride          = 1,
                padding         = self.kernel_size//2,
                padding_mode    = self.padding_mode,
                bias            = self.bias
            ),
            self.tanh,
            nn.Conv3d(
                in_channels     = 3*self.hidden_channels,
                out_channels    = 3*self.hidden_channels,
                kernel_size     = 1,
                stride          = 1,
                padding         = 0,
                padding_mode    = self.padding_mode,
                bias            = self.bias
            )
            )
        
        self.h2h = nn.Sequential(
            nn.Conv3d(
                in_channels     = self.hidden_channels,
                out_channels    = 2*self.hidden_channels,
                kernel_size     = self.kernel_size,
                stride          = 1,
                padding         = self.kernel_size//2,
                padding_mode    = self.padding_mode,
                bias            = self.bias
            ),
            self.tanh,
            nn.Conv3d(
                in_channels     = 2*self.hidden_channels,
                out_channels    = 2*self.hidden_channels,
                kernel_size     = 1,
                stride          = 1,
                padding         = 0,
                padding_mode    = self.padding_mode,
                bias            = self.bias
            )
            )
        
        self.h2candidate = nn.Sequential(
            nn.Conv3d(
                in_channels     = self.hidden_channels,
                out_channels    = self.hidden_channels,
                kernel_size     = self.kernel_size,
                stride          = 1,
                padding         = self.kernel_size//2,
                padding_mode    = self.padding_mode,
                bias            = self.bias
            ),
            self.tanh,
            nn.Conv3d(
                in_channels     = self.hidden_channels,
                out_channels    = self.hidden_channels,
                kernel_size     = 1,
                stride          = 1,
                padding         = 0,
                padding_mode    = self.padding_mode,
                bias            = self.bias
            )
            )
        
    def symmetrize(self):
        raise NotImplementedError('Symmetrization is not implemented for 3D convolutions')


class ConvGRU(nn.Module):
    '''
    This class defines a module stacking multiple ConvGRU together. The module also provides an additional output layer producing a B&W image using a sigmoid activation function
    '''
    
    def __init__(
        self,
        hidden_units        : int,
        input_channels      : int,
        output_channels     : int,
        hidden_channels     : int,
        kernel_size         : int,
        padding_mode        : str,
        separable           : bool          = False,
        reduce_out          : bool          = True,
        squash_out          : bool          = True,
        bias                : bool          = True,
        divergence          : bool          = True,
        conservative        : bool          = True,
        num_params          : int           = 0,
        dropout             : bool          = False,
        dropout_prob        : bool | None   = None
        ):
        '''
        init function
        '''
        
        super(ConvGRU, self).__init__()
        
        self.input_channels     = input_channels # number of "real" channels (evolving fields WITHOUT considering external parameters)
        self.hidden_units       = hidden_units
        self.hidden_channels    = hidden_channels
        self.kernel_size        = kernel_size
        self.padding_mode       = padding_mode
        
        self.output_channels    = output_channels

        self.num_params         = num_params # number of additional channels due to external parameters
        
        self.separable          = separable
        
        self.reduce_out         = reduce_out
        self.squash_out         = squash_out
        
        self.dropout            = dropout
        self.dropout_prob       = dropout_prob
        
        self.GRU_list   = nn.ModuleList()
        
        self.bias       = bias
        
        self.div_mode   = divergence
        self.conservative = conservative
        
        self.toOut      = nn.Sequential(
            nn.Conv2d(
                in_channels     = self.hidden_channels,
                out_channels    = self.hidden_channels,
                kernel_size     = 1,
                stride          = 1,
                padding         = 0,
                padding_mode    = self.padding_mode,
                bias            = self.bias
                
                ),
            nn.Tanh(),
            nn.Conv2d(
                in_channels     = self.hidden_channels,
                out_channels    = self.hidden_channels,
                kernel_size     = 1,
                stride          = 1,
                padding         = 0,
                padding_mode    = self.padding_mode,
                bias            = self.bias
                
                ),
            nn.Tanh(),
            nn.Conv2d(
                in_channels     = self.hidden_channels,
                out_channels    = self.input_channels if not self.div_mode else 2*self.input_channels,
                kernel_size     = 3,
                stride          = 1,
                padding         = 1,
                padding_mode    = self.padding_mode,
                bias            = self.bias
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
                    in_channels     = in_channels,
                    hidden_channels = self.hidden_channels,
                    kernel_size     = self.kernel_size,
                    padding_mode    = self.padding_mode,
                    separable       = self.separable,
                    bias            = self.bias
                    )
                )
                
        
    def make_div_filters(self, x):
        '''
        This method constructs the divergence filters
        '''
        
        print('Constructing differential operators as filters...', end='')
        
        grad1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False, padding_mode=self.padding_mode)
        grad2 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False, padding_mode=self.padding_mode)
        
        gradx_matrix = np.array([[0,0,0],[1,0,-1],[0,0,0]])
        grady_matrix = np.array([[0,1,0],[0,0,0],[0,-1,0]])
        
        grad1.weight = nn.Parameter(torch.from_numpy(gradx_matrix).float().unsqueeze(0).unsqueeze(0))
        grad2.weight = nn.Parameter(torch.from_numpy(grady_matrix).float().unsqueeze(0).unsqueeze(0))
        
        grad1.requires_grad = False
        grad2.requires_grad = False
        
        grad1.to(x.device)
        grad2.to(x.device)
        
        self.divergence_filters = [grad1, grad2]
        
        print('DONE!')
    
    def divergence(self, x):
        '''
        This method calculates the divergence of the given field using finite differences approximation
        '''
        
        Jx, Jy = torch.split(x, 1, dim=1)

        Jx = Jx - torch.mean(Jx, dim=(-1,-2), keepdim=True)
        Jy = Jy - torch.mean(Jy, dim=(-1,-2), keepdim=True)
        
        gradx = self.divergence_filters[0](Jx)
        grady = self.divergence_filters[1](Jy)
        
        divergence = gradx+grady
        
        return divergence
    
    
    def symmetrize(self):
        for GRU_cell in self.GRU_list:
            GRU_cell.symmetrize()
            
            
    def make_dropout_list(self, in_sequence, approx_inference):
        '''
        This method makes a dropout list for hidden list
        '''
        device = in_sequence.device
        dropout_tensor_shape = (in_sequence.shape[0], self.hidden_units, self.hidden_channels)  # shape is ( num_examples x num_convolutions x num_channels_for_convolution) 
        dropout_tensor = torch.rand(dropout_tensor_shape, device=device)
        
        if approx_inference:
            self.dropout_mask =  (dropout_tensor <= self.dropout_prob).float()/(1-self.dropout_prob) # <-- this is already ok for mutliplication with hidden channels
        else:
            self.dropout_mask = 1
    
    
    def forward_old(self, in_sequence, future=0, params=None, noise_reg=0.0, approx_inference=True):
        '''
        This method is called from forward if you are not in divergence mode
        '''
        
        # selecting dropout channels in hidden state
        if self.dropout: self.make_dropout_list(in_sequence, approx_inference)
        
        outputs = []
        hidden_list = []
        
        device = in_sequence.device #'cuda' if in_sequence.is_cuda else 'cpu'

        for ll in range(self.hidden_units):
            hidden_list.append(torch.zeros(in_sequence.size(0), self.hidden_channels, in_sequence.size(3), in_sequence.size(4), device=device, requires_grad=False))

        for input_t in in_sequence.split(1, dim=1):
            
            input_t_old = input_t
            
            if noise_reg != 0:
                noise = noise_reg*torch.randn(input_t.shape, device=input_t.device)
                if self.conservative: noise = noise - torch.mean(noise, dim=(-1,-2), keepdim=True) # remove mean noise value
                input_t = input_t + noise

            input_t = self.cat_params(input_t, params)
            
            for kk in range(self.hidden_units):

                if kk==0:
                    hidden_list[kk] = self.GRU_list[kk](input_t.squeeze(1), hidden_list[kk])
                else: hidden_list[kk] = self.GRU_list[kk](hidden_list[kk-1], hidden_list[kk])
                    
                if self.dropout:
                    for example in range(in_sequence.shape[0]): # iterate in the batch dimension
                        for channel in range(self.hidden_channels):
                            hidden_list[kk][example,channel,:,:] = hidden_list[kk][example,channel,:,:]*self.dropout_mask[example, kk, channel] # this will zero-out some of the hidden shapes
            
            if self.reduce_out:
                output = self.toOut(hidden_list[-1])
                if self.squash_out:
                    if self.conservative: output = output - torch.mean(output, dim=(-1,-2), keepdim=True)
                    output = input_t_old.squeeze(1)+ output#self.sigmoid(output)
            else:
                output = hidden_list[-1]
                
            outputs += [output]
            
        for _ in range(future):

            output_old = output
            
            if noise_reg != 0:
                noise =  noise_reg*torch.randn(output.shape, device=output.device)
                if self.conservative: noise = noise - torch.mean(noise, dim=(-1,-2), keepdim=True)
                output = output + noise
            
            for kk in range(self.hidden_units):
                
                if kk==0: hidden_list[kk] = self.GRU_list[kk](self.cat_params(output, params), hidden_list[kk])
                else: hidden_list[kk] = self.GRU_list[kk](hidden_list[kk-1], hidden_list[kk])
                
                if self.dropout:
                    for example in range(in_sequence.shape[0]): # iterate in the batch dimension
                        for channel in range(self.hidden_channels):
                            hidden_list[kk][example,channel,:,:] = hidden_list[kk][example,channel,:,:]*self.dropout_mask[example, kk, channel] # this will zero-out some of the hidden shapes
            
            if self.reduce_out:
                output = self.toOut(hidden_list[-1])
                if self.squash_out:
                    if self.conservative: output = output - torch.mean(output, dim=(-1,-2), keepdim=True)
                    output = output_old + output #self.sigmoid(output)
            else:
                output = hidden_list[-1]
            outputs += [output]
        
        outputs = torch.stack(outputs, dim=1)

        return outputs
    
    
    def forward_div(self, in_sequence, future=0, params=None, noise_reg=0.0, approx_inference=True):
        '''
        This method is called in divergence mode; BETA
        '''
        
        # dropout stuff
        if self.dropout: self.make_dropout_list(in_sequence,approx_inference)
        
        outputs = []
        hidden_list = []
        
        device = in_sequence.device#'cuda' if in_sequence.is_cuda else 'cpu'

        for ll in range(self.hidden_units):
            hidden_list.append(torch.zeros(in_sequence.size(0), self.hidden_channels, in_sequence.size(3), in_sequence.size(4), device=device, requires_grad=False))

        for input_t in in_sequence.split(1, dim=1):
            
            input_t_old = input_t
            input_t = self.cat_params(input_t, params)

            for kk in range(self.hidden_units):

                if kk==0:
                    hidden_list[kk] = self.GRU_list[kk](input_t.squeeze(1), hidden_list[kk])
                else: hidden_list[kk] = self.GRU_list[kk](hidden_list[kk-1], hidden_list[kk])
                
                if self.dropout:
                    for example in range(in_sequence.shape[0]): # iterate in the batch dimension
                        for channel in range(self.hidden_channels):
                            hidden_list[kk][example,channel,:,:] = hidden_list[kk][example,channel,:,:]*self.dropout_mask[example, kk, channel] # this will zero-out some of the hidden shapes
            
            if self.reduce_out:
                output = self.toOut(hidden_list[-1])
                if self.squash_out:
                    #output = input_t.squeeze(1)+self.divergence(output)
                    output = input_t_old.squeeze(1)+self.divergence(output)
            else:
                output = hidden_list[-1]
                output = input_t_old.squeeze(1)+self.divergence(output)
                
            outputs += [output]    
            
            if noise_reg != 0:
                noise   = noise_reg*torch.randn(output.shape, device=output.device)
                noise   = noise - torch.mean(noise, dim=(-1,-2), keepdim=True)
                output  = output + noise
            
        for _ in range(future):
            
            output_old = output
            output = self.cat_params(output, params)
            
            for kk in range(self.hidden_units):
                
                if kk==0: hidden_list[kk] = self.GRU_list[kk](output, hidden_list[kk])
                else: hidden_list[kk] = self.GRU_list[kk](hidden_list[kk-1], hidden_list[kk])
                
                if self.dropout:
                    for example in range(in_sequence.shape[0]): # iterate in the batch dimension
                        for channel in range(self.hidden_channels):
                            hidden_list[kk][example,channel,:,:] = hidden_list[kk][example,channel,:,:]*self.dropout_mask[example, kk, channel] # this will zero-out some of the hidden shapes
            
            if self.reduce_out:
                output = self.toOut(hidden_list[-1])
                if self.squash_out:
                    output = output_old+self.divergence(output)
            else:
                output = hidden_list[-1]
                output = output_old+self.divergence(output)
            
            outputs += [output]
            
            if noise_reg != 0:
                noise = noise_reg*torch.randn(output.shape, device=output.device)
                noise = noise - torch.mean(noise, dim=(-1,-2), keepdim=True)
                output = output + noise
        
        outputs = torch.stack(outputs, dim=1)

        return outputs
    


    def cat_params(self, in_sequence, params):
        '''
        This method extends the parameter tensor with external parameters
        '''
        if params is not None:
            #raise NotImplementedError('Parameter passing to the NN is not implemented yet... Aborting.')

            num_params_from_loader = len(params)

            if num_params_from_loader != self.num_params:
                raise ValueError('The number of parameters provided by the dataloader is not consistent with the one in the NN model.')

            shapes = list(in_sequence.shape)
            shapes[-3] = shapes[-3]+self.num_params
            in_sequence = in_sequence.expand(shapes).clone()

            for cc, params_batch in enumerate(params): # cycling on batches
                for bb, param in enumerate(params_batch): # cycling on channels
                    if len(shapes) == 5: in_sequence[bb,:,-(cc+1),:,:] = in_sequence[bb,:,-(cc+1),:,:]*0 + param.float()
                    else: in_sequence[bb,-(cc+1),:,:] = in_sequence[bb,-(cc+1),:,:]*0 + param.float()

            return in_sequence
        
        else:
            return in_sequence


    def forward(self, in_sequence, future=0, params=None, noise_reg=0.0, approx_inference=True):
        
        #in_sequence = self.cat_params(in_sequence, params)
        
        if self.div_mode:
            return self.forward_div(in_sequence, future, params=params, noise_reg=noise_reg, approx_inference=approx_inference) # conservative dynamics
        else:
            return self.forward_old(in_sequence, future, params=params, noise_reg=noise_reg, approx_inference=approx_inference) # non conservative dynamics
    

class ConvGRU3D(nn.Module):
    '''
    ConvGRU3D with learned face fluxes and a conservative FV update.
    Input: (B,T,C,X,Y,Z). Output: (B,T+future,C,X,Y,Z).
    Uniform, equal-volume cells; zero normal flux at all outer faces.
    No mobility, chemical potential, clipping or bound guarantee.
    voxel_size and dt must be saved with the model configuration.
    '''
    
    def __init__(self, hidden_units, input_channels, hidden_channels, kernel_size, padding_mode, bias=True, divergence=True, separable=False, num_params=0, dropout=False, dropout_prob=None, output_channels=None, conservative=False, *, voxel_size=0.025, dt=0.005):
        '''
        Constructor method
        '''
        super(ConvGRU3D, self).__init__()
        
        if separable:
            raise NotImplementedError('Separable convolution is not implemented in ConvGRU3D yet...')
        
        import math
        if hidden_units < 1 or input_channels < 1 or hidden_channels < 1:
            raise ValueError('Layer and channel counts must be positive.')
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError('kernel_size must be positive and odd.')
        if num_params < 0:
            raise ValueError('num_params must be nonnegative.')
        if output_channels not in (None, input_channels):
            raise ValueError('The number of output fields must match input_channels.')
        if isinstance(voxel_size, (int, float)):
            voxel_size = (float(voxel_size),) * 3
        else:
            voxel_size = tuple(float(h) for h in voxel_size)
        if len(voxel_size) != 3 or any(not math.isfinite(h) or h <= 0 for h in voxel_size):
            raise ValueError('voxel_size must be a positive scalar or (hx,hy,hz).')
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError('dt must be finite and positive.')
        if dropout and (dropout_prob is None or not 0 <= dropout_prob < 1):
            raise ValueError('With dropout=True, set dropout_prob in [0,1).')
        self.voxel_size = voxel_size
        self.dt = float(dt)
        # The legacy flag cannot disable conservation in FV mode.
        self.conservative = bool(divergence)
        self.output_channels = input_channels

        self.input_channels     = input_channels # number of "real" channels (evolving fields WITHOUT considering external parameters)
        self.hidden_units       = hidden_units
        self.hidden_channels    = hidden_channels
        self.kernel_size        = kernel_size
        self.padding_mode       = padding_mode

        self.num_params         = num_params # number of additional channels due to external parameters
        
        self.GRU_list   = nn.ModuleList()
        self.Norm_list  = nn.ModuleList()
        
        self.bias       = bias
        
        self.div_mode   = divergence
        
        self.dropout        = dropout
        self.dropout_prob   = dropout_prob
        
        self.toOut      = nn.Sequential(
            nn.Conv3d(
                in_channels     = self.hidden_channels,
                out_channels    = self.hidden_channels,
                kernel_size     = 1,
                stride          = 1,
                padding         = 0,
                padding_mode    = self.padding_mode,
                bias            = self.bias
                
                ),
            nn.Tanh(),
            nn.Conv3d(
                in_channels     = self.hidden_channels,
                out_channels    = self.hidden_channels,
                kernel_size     = 1,
                stride          = 1,
                padding         = 0,
                padding_mode    = self.padding_mode,
                bias            = self.bias
                
                ),
            nn.Tanh(),
            nn.Conv3d(
                in_channels     = self.hidden_channels,
                out_channels    = self.input_channels if not self.div_mode else 3*self.input_channels,
                kernel_size     = 3,
                stride          = 1,
                padding         = 1,
                padding_mode    = self.padding_mode,
                bias            = self.bias
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
                    in_channels     = in_channels,
                    hidden_channels = self.hidden_channels,
                    kernel_size     = self.kernel_size,
                    padding_mode    = self.padding_mode,
                    bias            = self.bias
                    )
                )
        
        
    def make_div_filters(self, x=None):
        """Compatibility with train.py: no filters or parameters are needed."""
        return None

    def face_fluxes(self, raw):
        """Decode (B,3*C,X,Y,Z) into oriented face fluxes, including boundaries.

        The three channel blocks contain Jx, Jy, Jz. Index i in Jx
        represents face i+1/2, NOT a cell-centred flux. The last plane
        along each component's own axis is unused. Outer faces are zero.
        """
        if raw.ndim != 5 or raw.shape[1] != 3 * self.input_channels:
            raise ValueError('Expected raw flux shape (B,3*C,X,Y,Z).')
        jx, jy, jz = raw.split(self.input_channels, dim=1)
        return (
            torch.nn.functional.pad(jx[:, :, :-1, :, :], (0, 0, 0, 0, 1, 1)),
            torch.nn.functional.pad(jy[:, :, :, :-1, :], (0, 0, 1, 1, 0, 0)),
            torch.nn.functional.pad(jz[:, :, :, :, :-1], (1, 1, 0, 0, 0, 0)),
        )

    def divergence(self, raw):
        """Physical FV divergence. No dt, sign reversal or mean subtraction.

        Uniform cell-centred control volumes, closed on all six sides.
        Spatial tensor axes are X,Y,Z, in exactly this order.
        """
        # Keep conservative arithmetic at least float32 under mixed precision.
        if raw.dtype in (torch.float16, torch.bfloat16):
            raw = raw.float()
        jx, jy, jz = self.face_fluxes(raw)
        hx, hy, hz = self.voxel_size
        return (
            (jx[:, :, 1:, :, :] - jx[:, :, :-1, :, :]) / hx
            + (jy[:, :, :, 1:, :] - jy[:, :, :, :-1, :]) / hy
            + (jz[:, :, :, :, 1:] - jz[:, :, :, :, :-1]) / hz
        )

    def cat_params(self, field, params):
        """Append constant parameter channels to a 5D field or 6D sequence.

        params: legacy list of P tensors of shape (B,), or tensor (B,P).
        Preserve the original ordering: parameter 0 is the LAST channel.
        """
        if field.ndim not in (5, 6):
            raise ValueError('Expected a 5D field or 6D sequence.')
        if self.num_params == 0:
            if params is not None and len(params) != 0:
                raise ValueError('This model has num_params=0.')
            return field
        if params is None:
            raise ValueError('params is required when num_params > 0.')
        if torch.is_tensor(params):
            values = params.to(device=field.device, dtype=field.dtype)
        else:
            if len(params) != self.num_params:
                raise ValueError('Wrong number of external parameters.')
            values = torch.stack([
                torch.as_tensor(p, device=field.device, dtype=field.dtype).reshape(-1)
                for p in params
            ], dim=1)
        if values.shape != (field.shape[0], self.num_params):
            raise ValueError('Parameters must have shape (B,P).')
        values = values.flip(1)
        if field.ndim == 5:
            maps = values[:, :, None, None, None].expand(-1, -1, *field.shape[-3:])
            return torch.cat((field, maps), dim=1)
        maps = values[:, None, :, None, None, None].expand(
            -1, field.shape[1], -1, *field.shape[-3:])
        return torch.cat((field, maps), dim=2)

    def make_dropout_list(self, in_sequence, approx_inference):
        """One channel mask per layer/sample, shared across all time steps.

        Preserve caller convention: approx_inference=False enables dropout;
        True disables it. eval() also disables dropout.
        """
        shape = (in_sequence.shape[0], self.hidden_units, self.hidden_channels)
        mask = in_sequence.new_ones(shape)
        if self.dropout and self.training and not approx_inference:
            keep = 1.0 - self.dropout_prob
            mask = mask.bernoulli_(keep) / keep
        self.dropout_mask = mask
        return mask

    def _forward_sequence(self, in_sequence, future, params, noise_reg,
                          approx_inference, use_divergence):
        if in_sequence.ndim != 6 or in_sequence.shape[2] != self.input_channels:
            raise ValueError('Expected input shape (B,T,C,X,Y,Z).')
        if in_sequence.shape[1] < 1 or min(in_sequence.shape[-3:]) < 1:
            raise ValueError('The input sequence and spatial dimensions must be nonempty.')
        if not isinstance(future, int) or future < 0:
            raise ValueError('future must be a nonnegative integer.')
        if noise_reg < 0:
            raise ValueError('noise_reg must be nonnegative.')
        if in_sequence.dtype in (torch.float16, torch.bfloat16):
            in_sequence = in_sequence.float()
        hidden = [in_sequence.new_zeros(
            in_sequence.shape[0], self.hidden_channels, *in_sequence.shape[-3:]
        ) for _ in range(self.hidden_units)]
        masks = self.make_dropout_list(in_sequence, approx_inference)
        outputs = []
        observed = in_sequence.shape[1]
        for t in range(observed + future):
            # Each observed input predicts its successor; future continues
            # autoregressively from the LAST predicted field.
            base = in_sequence[:, t] if t < observed else output
            recurrent_input = base
            if noise_reg and t >= observed:
                noise = noise_reg * torch.randn_like(base)
                noise = noise - noise.mean(dim=(-3, -2, -1), keepdim=True)
                recurrent_input = base + noise
            # Noise regularizes the recurrent input, not the conservative base.
            layer_input = self.cat_params(recurrent_input, params)
            for k, cell in enumerate(self.GRU_list):
                state = cell(layer_input, hidden[k])
                state = state * masks[:, k, :, None, None, None]
                hidden[k] = state
                layer_input = state
            raw = self.toOut(hidden[-1])
            if use_divergence:
                output = base - self.dt * self.divergence(raw)
            else:
                output = self.sigmoid(raw)
            outputs.append(output)
        return torch.stack(outputs, dim=1)

    def forward_div(self, in_sequence, future=0, params=None, noise_reg=0.0,
                    approx_inference=True):
        if not self.div_mode:
            raise ValueError('forward_div requires divergence=True at construction.')
        return self._forward_sequence(in_sequence, future, params, noise_reg,
                                      approx_inference, True)

    def forward_old(self, in_sequence, future=0, params=None, noise_reg=0.0,
                    approx_inference=True):
        if self.div_mode:
            raise ValueError('forward_old requires divergence=False at construction.')
        return self._forward_sequence(in_sequence, future, params, noise_reg,
                                      approx_inference, False)

    def forward(self, in_sequence, future=0, params=None, noise_reg=0.0,
                approx_inference=True):
        return self._forward_sequence(in_sequence, future, params, noise_reg,
                                      approx_inference, self.div_mode)


class ConvGRUClassifier(ConvGRU):
    '''
    This class subclasses ConvGRU to obtain a classifier for video data
    '''
    def __init__(self, *args, **kwargs):
        '''
        Constructor method
        '''
        super().__init__(*args, **kwargs)
        
        self.reduce_out = False # we need to convert the hidden layer only at the end of the sequence
        
        self.activation = nn.Tanh()
        
        self.toOut      = nn.Sequential(
            #nn.MaxPool2d(2), 
            nn.Conv2d(
                in_channels     = self.hidden_channels,
                out_channels    = 2*self.hidden_channels,
                kernel_size     = 2,
                stride          = 2,
                padding         = 0,
                padding_mode    = self.padding_mode,
                bias            = self.bias
                ),
            self.activation,# -> 128x128
            #nn.MaxPool2d(2), # -> 64x64
            nn.Conv2d(
                in_channels     = 2*self.hidden_channels,
                out_channels    = 2*self.hidden_channels,
                kernel_size     = 2,
                stride          = 2,
                padding         = 0,
                padding_mode    = self.padding_mode,
                bias            = self.bias
                ),
            self.activation, # -> 32x32
            nn.MaxPool2d(2), # -> 16x16
            nn.Conv2d(
                in_channels     = 2*self.hidden_channels,
                out_channels    = 4*self.hidden_channels,
                kernel_size     = 2,
                stride          = 2,
                padding         = 0,
                padding_mode    = self.padding_mode,
                bias            = self.bias
                ),
            self.activation, # -> 8x8
            nn.MaxPool2d(2), # -> 4x4
            nn.Conv2d(
                in_channels     = 4*self.hidden_channels,
                out_channels    = 4*self.hidden_channels,
                kernel_size     = 2,
                stride          = 2,
                padding         = 0,
                padding_mode    = self.padding_mode,
                bias            = self.bias
                ),
            self.activation, # -> 2x2
            #nn.MaxPool2d(2), # -> 2x2
            nn.Conv2d(
                in_channels     = 4*self.hidden_channels,
                out_channels    = self.output_channels,
                kernel_size     = 2,
                stride          = 2,
                padding         = 0,
                padding_mode    = self.padding_mode,
                bias            = self.bias
                )
            )
            
    def forward(self, in_sequence, future=0, params=None, noise_reg=0.0):
        
        in_sequence = self.cat_params(in_sequence, params)
        GRU_result = self.forward_old(in_sequence, future, params=params, noise_reg=noise_reg) # non conservative dynamics
        
        out = self.toOut(GRU_result[:,-1,:,:,:]).squeeze(-1).squeeze(-1)
        
        return out


class CommitteeModel():
    '''
    This class is a wrapper for individual models in order to have committee predictions and committee uncertainty estimation.
    '''
    def __init__(self, PATH_iter, device, hidden_units, input_channels, hidden_channels, kernel_size, padding_mode, separable=False, reduce_out=True, squash_out=True, divergence=True, bias=False, num_params=0):
        
        self.model_list = []
        self.device = device
        
        for path in PATH_iter:
            self.model_list.append(
                ConvGRU(
                    hidden_units        = hidden_units,
                    input_channels      = input_channels,
                    hidden_channels     = hidden_channels,
                    kernel_size         = kernel_size,
                    padding_mode        = padding_mode,
                    separable           = separable,
                    reduce_out          = reduce_out,
                    squash_out          = squash_out,
                    divergence          = divergence,
                    bias                = bias,
                    num_params          = num_params
                    )
                )
                
            self.model_list[-1].load_state_dict( torch.load(path, map_location=self.device) )
            
        for ii in range( len(self.model_list) ):
            self.model_list[ii].eval()
            self.model_list[ii].to(self.device)
            
        self.num_models = len(self.model_list)
        
        
    def __len__(self):
        return self.num_models
    
    def to(self, device):
        self.device = device
        for ii in range( len(self.model_list) ):
            self.model_list[ii].to(self.device)
            
    def pass2proc(self, x, future, model, queue):
        queue.put( model(x, future=future).detach().cpu().numpy() )
            
    def __call__(self, x, future=0, params=None, scatter=False):
        out_list    = []
        with torch.no_grad():
            jj = 0
            for model in self.model_list:
                jj += 1
                print(f'Model {jj} is predicting...', end='')
                out_list.append( model(x, future=future, params=params) )
                #torch.cuda.empty_cache()
                print('done!')
        if not scatter:
            return self.committee_decision(out_list)
        else:
            return out_list
    
    def eval(self):
        for model in self.model_list:
            model.eval()
    
    def committee_decision(self, data_list):
        
        with torch.no_grad():
            
            data_mean = torch.zeros( data_list[0].shape )
            data_var  = torch.zeros( data_list[0].shape )
            data_mean = data_mean.to(self.device)
            data_var  = data_var.to(self.device)
            
            for val in data_list:
                data_mean += val
            data_mean /= len(data_list)
            
            if len(data_list) > 2: # <- variance cannot be estimated if only two models are present
                for val in data_list:
                    data_var += (val-data_mean)**2
            else:
                #raise UserWarning('Only two models are present in the committee model. Prediction variance cannot be estimated.')
                data_var = None
            
            if data_var is not None:
                data_var /= len(data_list)-1
                data_var = torch.sqrt(data_var)
            else:
                data_var = data_mean*0-1
            
            return data_mean, data_var

        
# --- NN-related classes ---

# <<< phase field related classes <<<
class Phi():
    '''
    This class implement an abstraction of a phase-field object. It is able to add shapes and import images (possibly both)
    '''
    def __init__(self, res=(100,100)):
        self.val    = np.zeros(res)
        x = np.arange(0, self.val.shape[0])
        y = np.arange(0, self.val.shape[1])
        self.meshgrid = np.meshgrid(x, y)
       
    def set_center(x,y):
        self.center = (x,y)
    
    
    def paint_shape(self, shape_fun, filler_value):
        '''
        This method is a wrapper for the jitted function paint_shape_jitted. It "colors" with the given filler_value val inside Phi
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
        This method returns the content of phi.val as a np array. Rescale down is used to "heal" sharp pixels in rounded shapes.
        '''
        image = PIL.Image.fromarray(self.val)
        image = image.resize( (self.val.shape[0]//rescale_down,
                               self.val.shape[1]//rescale_down) )
        return np.asfarray(image)
    
    
    def plot(self, path, rescale_down=1, cmap='gray'):
        '''
        This method saves the content of self.val in the provided path. rescale_down kwarg is required for reducing sharp pixels in rounded shapes (reccomended if paint_shape has been used).
        '''
        matplotlib.use('Agg')
        
        image = PIL.Image.fromarray(self.val)
        image = image.resize( (self.val.shape[0]//rescale_down,
                               self.val.shape[1]//rescale_down) )
        
        val = np.asarray(image)
        
        plt.imshow(val, cmap=cmap, vmin=0, vmax=1)
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        
    def import_image(self, path, side, cropkey, crop_lim=(0.25,0.75)):
        '''
        This method imports an external image into phi.val
        '''
        transform = transforms.Compose(
                    [
                        transforms.Grayscale(num_output_channels=1),
                        transforms.Resize(side),
                        transforms.ToTensor()
                    ]
                )
        val = PIL.Image.open(path)
        val = make_square(val, cropkey=cropkey, crop_lim=crop_lim)
        val = transform(val)
        self.val = np.array(val[0])
        
    def phi4ConvGRU(self, rescale_down=1):
        '''
        This function returns self.val to an appropriate tensor for ConvGRU
        '''
        image = PIL.Image.fromarray(self.val)
        image = image.resize( (self.val.shape[0]//rescale_down,
                               self.val.shape[1]//rescale_down) )
        
        val = np.asarray(image)
        
        tensor = torch.zeros(1,1,1, val.shape[0], val.shape[1])
        tensor[0,0,0,:,:] = torch.from_numpy( val ).float()
        
        return tensor
    
    
    def dual(self):
        '''
        This method produces the dual image of self.val (1->0 and 0->1)
        '''
        self.val = -self.val+1
    
# --- phase field related classes ---

# <<< Phi class jitted functions <<<
@njit
def paint_shape_jitted(val, shape_fun, xsize, ysize, filler_value):
    '''
    This is a jitted function to fill-in pixels with filler value faster
    '''
    for x in prange(xsize):
        for y in range(ysize):
            for Lx in [-xsize, 0, xsize]:
                for Ly in [-ysize, 0, ysize]:
                    if shape_fun(x+Lx,y+Ly) < 0:
                        val[x,y] = filler_value
    return val
# --- Phi class jitted functions ---
