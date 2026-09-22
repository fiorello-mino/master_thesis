from sys import path
#path.append('/home/roberto/Scaricati/CRANE-main/')
path.append('/home/fiorello/CRANE/')
import torch
# <<< import CRANE modules <<<
from src.utils import *
from src.classes import ConvGRU, ConvGRUClassifier, ConvGRU3D

class PersistentModel( ConvGRU ):
    def __init__(self, *args, **kwargs) -> None:
         super(PersistentModel, self).__init__(*args, **kwargs) # simply initialize the model as we know
         self.hidden_list = None # hidden_list is now an attribute

    def set_hidden(
         self,
         in_sequence     : torch.Tensor
         ) -> None:
         '''
         This method is simply setting the hidden state to zero (to be used to erase memory or as an initialization step
         '''
         print('Resetting hidden state...', end='', flush=True)
         self.hidden_list = []

         for ll in range(self.hidden_units):
             self.hidden_list.append(
                 torch.zeros(
                     in_sequence.size(0),
                     self.hidden_channels,
                     in_sequence.size(3),
                     in_sequence.size(4),
                     device = in_sequence.device,
                     requires_grad = False
                     )
                 )
 
         print('DONE!')

    #@torch.compile()
    def forward_old(self, in_sequence, future=0, params=None, noise_reg=0.0, approx_inference=True):
        '''
        This method is called from forward if you are not in divergence mode
        '''
        
        # selecting dropout channels in hidden state
        if self.dropout: self.make_dropout_list(in_sequence, approx_inference)
        
        outputs = []
        hidden_list = []
        
        device = in_sequence.device #'cuda' if in_sequence.is_cuda else 'cpu'

        if self.hidden_list is None: self.set_hidden(in_sequence)

        for input_t in in_sequence.split(1, dim=1):
            
            input_t_old = input_t
            
            if noise_reg != 0:
                input_t = input_t + noise_reg*torch.randn(input_t.shape, device=input_t.device)
            
            input_t = self.cat_params(input_t, params)
            
            for kk in range(self.hidden_units):

                if kk==0:
                    self.hidden_list[kk] = self.GRU_list[kk](input_t.squeeze(1), self.hidden_list[kk])
                else: self.hidden_list[kk] = self.GRU_list[kk](self.hidden_list[kk-1], self.hidden_list[kk])
                    
                if self.dropout:
                    for example in range(in_sequence.shape[0]): # iterate in the batch dimension
                        for channel in range(self.hidden_channels):
                            self.hidden_list[kk][example,channel,:,:] = self.hidden_list[kk][example,channel,:,:]*self.dropout_mask[example, kk, channel] # this will zero-out some of the hidden shapes
            self.conservative = False
            if self.reduce_out:
                output = self.toOut(self.hidden_list[-1])
                if self.squash_out:
                    if self.conservative: output = output - torch.mean(output, dim=(-1,-2), keepdim=True)
                    output = input_t_old.squeeze(1) + output#self.sigmoid(output)
            else:
                output = hidden_list[-1]
                
            outputs += [output]
            
        for _ in range(future):

            output_old = output
            
            if noise_reg != 0:
                output = output + noise_reg*torch.randn(output.shape, device=output.device)
            
            
            for kk in range(self.hidden_units):
                
                if kk==0: hidden_list[kk] = self.GRU_list[kk](self.cat_params(output, params), self.hidden_list[kk])
                else: self.hidden_list[kk] = self.GRU_list[kk](self.hidden_list[kk-1], self.hidden_list[kk])
                
                if self.dropout:
                    for example in range(in_sequence.shape[0]): # iterate in the batch dimension
                        for channel in range(self.hidden_channels):
                            self.hidden_list[kk][example,channel,:,:] = self.hidden_list[kk][example,channel,:,:]*self.dropout_mask[example, kk, channel] # this will zero-out some of the hidden shapes
            
            self.conservative = False
            if self.reduce_out:
                output = self.toOut(self.hidden_list[-1])
                if self.squash_out:
                    if self.conservative: output = output - torch.mean(output, dim=(-1,-2), keepdim=True)
                    output = output_old + output #self.sigmoid(output)
            else:
                output = hidden_list[-1]
                
            outputs += [output]
            
        for _ in range(future):

            output_old = output
            
            if noise_reg != 0:
                output = output + noise_reg*torch.randn(output.shape, device=output.device)
            
            
            for kk in range(self.hidden_units):
                
                if kk==0: hidden_list[kk] = self.GRU_list[kk](self.cat_params(output, params), self.hidden_list[kk])
                else: self.hidden_list[kk] = self.GRU_list[kk](self.hidden_list[kk-1], self.hidden_list[kk])
                
                if self.dropout:
                    for example in range(in_sequence.shape[0]): # iterate in the batch dimension
                        for channel in range(self.hidden_channels):
                            self.hidden_list[kk][example,channel,:,:] = self.hidden_list[kk][example,channel,:,:]*self.dropout_mask[example, kk, channel] # this will zero-out some of the hidden shapes
            
            self.conservative = False
            if self.reduce_out:
                output = self.toOut(self.hidden_list[-1])
                if self.squash_out:
                    if self.conservative: output = output - torch.mean(output, dim=(-1,-2), keepdim=True)
                    output = output_old + output #self.sigmoid(output)
            else:
                output = hidden_list[-1]
            outputs += [output]
        
        outputs = torch.stack(outputs, dim=1)

        return outputs
 

    #@torch.compile()
    def forward_div(self, in_sequence, future=0, params=None, noise_reg=0.0, approx_inference=True):
        '''
        This method is called in divergence mode; BETA
        '''
        
        # dropout stuff
        if self.dropout: self.make_dropout_list(in_sequence,approx_inference)
        
        outputs = []
        hidden_list = []
        
        device = in_sequence.device#'cuda' if in_sequence.is_cuda else 'cpu'

        if self.hidden_list is None: self.set_hidden(in_sequence)

        for input_t in in_sequence.split(1, dim=1):
            
            input_t_old = input_t
            input_t = self.cat_params(input_t, params)

            for kk in range(self.hidden_units):

                if kk==0:
                    self.hidden_list[kk] = self.GRU_list[kk](input_t.squeeze(1), self.hidden_list[kk])
                else: self.hidden_list[kk] = self.GRU_list[kk](self.hidden_list[kk-1], self.hidden_list[kk])

                
                if self.dropout:
                    for example in range(in_sequence.shape[0]): # iterate in the batch dimension
                        for channel in range(self.hidden_channels):
                            self.hidden_list[kk][example,channel,:,:] = self.hidden_list[kk][example,channel,:,:]*self.dropout_mask[example, kk, channel] # this will zero-out some of the hidden shapes
            
            if self.reduce_out:
                output = self.toOut(self.hidden_list[-1])
                if self.squash_out:
                    #output = input_t.squeeze(1)+self.divergence(output)
                    output = input_t_old.squeeze(1)+self.divergence(output)
            else:
                output = self.hidden_list[-1]
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
                
                if kk==0: self.hidden_list[kk] = self.GRU_list[kk](output, self.hidden_list[kk])
                else: self.hidden_list[kk] = self.GRU_list[kk](self.hidden_list[kk-1], self.hidden_list[kk])
                
                if self.dropout:
                    for example in range(in_sequence.shape[0]): # iterate in the batch dimension
                        for channel in range(self.hidden_channels):
                            self.hidden_list[kk][example,channel,:,:] = self.hidden_list[kk][example,channel,:,:]*self.dropout_mask[example, kk, channel] # this will zero-out some of the hidden shapes
            
            if self.reduce_out:
                output = self.toOut(self.hidden_list[-1])
                if self.squash_out:
                    output = output_old+self.divergence(output)
            else:
                output = self.hidden_list[-1]
                output = output_old+self.divergence(output)
            
            outputs += [output]
            
            if noise_reg != 0:
                noise = noise_reg*torch.randn(output.shape, device=output.device)
                noise = noise - torch.mean(noise, dim=(-1,-2), keepdim=True)
                output = output + noise
        
        outputs = torch.stack(outputs, dim=1)

        return outputs



class PersistentModel3D( ConvGRU3D ):
    def __init__(self, *args, **kwargs) -> None:
         super(PersistentModel3D, self).__init__(*args, **kwargs)
         # Unlike ConvGRU3D, the recurrent state survives across forward calls.
         self.hidden_list = None
         # Kept for backward compatibility with the legacy forward_old method.
         self.reduce_out = True
         self.squash_out = True

    def set_hidden(
         self,
         in_sequence     : torch.Tensor
         ) -> None:
         '''
         This method is simply setting the hidden state to zero (to be used to erase memory or as an initialization step
         '''
         if in_sequence.ndim != 6:
             raise ValueError('Expected input shape (B,T,C,X,Y,Z).')
         self.hidden_list = [
             in_sequence.new_zeros(
                 in_sequence.shape[0],
                 self.hidden_channels,
                 *in_sequence.shape[-3:]
             )
             for _ in range(self.hidden_units)
         ]

    def reset_hidden(self) -> None:
        '''Discard the persistent state; it will be rebuilt on the next call.'''
        self.hidden_list = None

    #@torch.compile()
    def forward_old(self, in_sequence, future=0, params=None, noise_reg=0.0, approx_inference=True):
        '''
        This method is called from forward if you are not in divergence mode
        '''

        # selecting dropout channels in hidden state
        if self.dropout: self.make_dropout_list(in_sequence, approx_inference)

        outputs = []
        hidden_list = []

        device = in_sequence.device #'cuda' if in_sequence.is_cuda else 'cpu'

        if self.hidden_list is None: self.set_hidden(in_sequence)

        for input_t in in_sequence.split(1, dim=1):

            input_t_old = input_t

            if noise_reg != 0:
                input_t = input_t + noise_reg*torch.randn(input_t.shape, device=input_t.device)

            input_t = self.cat_params(input_t, params)

            for kk in range(self.hidden_units):

                if kk==0:
                    self.hidden_list[kk] = self.GRU_list[kk](input_t.squeeze(1), self.hidden_list[kk])
                else: self.hidden_list[kk] = self.GRU_list[kk](self.hidden_list[kk-1], self.hidden_list[kk])

                if self.dropout:
                    for example in range(in_sequence.shape[0]): # iterate in the batch dimension
                        for channel in range(self.hidden_channels):
                            self.hidden_list[kk][example,channel,:,:,:] = self.hidden_list[kk][example,channel,:,:,:]*self.dropout_mask[example, kk, channel] # this will zero-out some of the hidden shapes
            self.conservative = False
            if self.reduce_out:
                output = self.toOut(self.hidden_list[-1])
                if self.squash_out:
                    if self.conservative: output = output - torch.mean(output, dim=(-1,-2,-3), keepdim=True)
                    output = input_t_old.squeeze(1) + output#self.sigmoid(output)
            else:
                output = hidden_list[-1]

            outputs += [output]


        for _ in range(future):

            output_old = output

            if noise_reg != 0:
                output = output + noise_reg*torch.randn(output.shape, device=output.device)


            for kk in range(self.hidden_units):

                if kk==0: hidden_list[kk] = self.GRU_list[kk](self.cat_params(output, params), self.hidden_list[kk])
                else: self.hidden_list[kk] = self.GRU_list[kk](self.hidden_list[kk-1], self.hidden_list[kk])

                if self.dropout:
                    for example in range(in_sequence.shape[0]): # iterate in the batch dimension
                        for channel in range(self.hidden_channels):
                            self.hidden_list[kk][example,channel,:,:,:] = self.hidden_list[kk][example,channel,:,:,:]*self.dropout_mask[example, kk, channel] # this will zero-out some of the hidden shapes

            self.conservative = False
            if self.reduce_out:
                output = self.toOut(self.hidden_list[-1])
                if self.squash_out:
                    if self.conservative: output = output - torch.mean(output, dim=(-1,-2-3), keepdim=True)
                    output = output_old + output #self.sigmoid(output)
            else:
                output = hidden_list[-1]
            outputs += [output]

        outputs = torch.stack(outputs, dim=1)

        return outputs


    #@torch.compile()
    def forward_div(self, in_sequence, future=0, params=None, noise_reg=0.0, approx_inference=True):
        '''
        Persistent counterpart of ConvGRU3D.forward_div.

        It uses the same finite-volume update as ConvGRU3D but keeps the GRU
        hidden state between calls. Call reset_hidden() (or set_hidden(x))
        before starting an independent trajectory.
        '''
        if not self.div_mode:
            raise ValueError('forward_div requires divergence=True at construction.')
        if in_sequence.ndim != 6 or in_sequence.shape[2] != self.input_channels:
            raise ValueError('Expected input shape (B,T,C,X,Y,Z).')
        if in_sequence.shape[1] < 1 or min(in_sequence.shape[-3:]) < 1:
            raise ValueError('The input sequence and spatial dimensions must be nonempty.')
        if not isinstance(future, int) or future < 0:
            raise ValueError('future must be a nonnegative integer.')
        if noise_reg < 0:
            raise ValueError('noise_reg must be nonnegative.')

        # Match ConvGRU3D: conservative arithmetic is at least float32.
        if in_sequence.dtype in (torch.float16, torch.bfloat16):
            in_sequence = in_sequence.float()

        expected_hidden_shape = (
            in_sequence.shape[0],
            self.hidden_channels,
            *in_sequence.shape[-3:]
        )
        if self.hidden_list is None:
            self.set_hidden(in_sequence)
        elif (len(self.hidden_list) != self.hidden_units
              or any(tuple(state.shape) != expected_hidden_shape
                     or state.device != in_sequence.device
                     or state.dtype != in_sequence.dtype
                     for state in self.hidden_list)):
            raise ValueError(
                'Persistent hidden state is incompatible with this input. '
                'Call reset_hidden() before starting the new trajectory.'
            )

        masks = self.make_dropout_list(in_sequence, approx_inference)
        outputs = []
        observed = in_sequence.shape[1]

        for t in range(observed + future):
            # Observed frames use teacher forcing; later frames are autoregressive.
            base = in_sequence[:, t] if t < observed else output
            recurrent_input = base

            # As in ConvGRU3D, noise regularizes only the recurrent rollout
            # input. It does not perturb the conservative update base/output.
            if noise_reg and t >= observed:
                noise = noise_reg * torch.randn_like(base)
                noise = noise - noise.mean(dim=(-3, -2, -1), keepdim=True)
                recurrent_input = base + noise

            layer_input = self.cat_params(recurrent_input, params)
            for k, cell in enumerate(self.GRU_list):
                state = cell(layer_input, self.hidden_list[k])
                state = state * masks[:, k, :, None, None, None]
                self.hidden_list[k] = state
                layer_input = state

            raw_flux = self.toOut(self.hidden_list[-1])
            output = base - self.dt * self.divergence(raw_flux)
            outputs.append(output)

        return torch.stack(outputs, dim=1)

    def forward(self, in_sequence, future=0, params=None, noise_reg=0.0,
                approx_inference=True):
        # ConvGRU3D.forward calls its private stateless sequence routine, so
        # dispatch explicitly to preserve state in this subclass.
        if self.div_mode:
            return self.forward_div(in_sequence, future, params, noise_reg,
                                    approx_inference)
        return self.forward_old(in_sequence, future, params, noise_reg,
                                approx_inference)
