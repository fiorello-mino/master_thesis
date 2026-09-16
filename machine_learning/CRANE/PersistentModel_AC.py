from sys import path
#path.append('/home/roberto/Scaricati/CRANE-main/')
path.append('/home/fiorello/CRANE/')
import torch
# <<< import CRANE modules <<<
from src.utils import *
from src.classes_AC import ConvGRU, ConvGRUClassifier, ConvGRU3D

class MassConservingSigmoid(nn.Module):
    '''
    Layer di attivazione che forza l'output tra 0 e 1 (tramite Sigmoid) 
    e trova un moltiplicatore di Lagrange (lambd) per conservare la massa totale.
    '''
    def __init__(self, max_iter=15, tol=1e-5):
        super(MassConservingSigmoid, self).__init__()
        self.max_iter = max_iter
        self.tol = tol

    def forward(self, H, target_mass):
        # H: output grezzo della rete (Batch, 1, X, Y, Z)
        # target_mass: massa da conservare (Batch, 1, 1, 1, 1)
        B = H.shape[0]
        lambd = torch.zeros((B, 1, 1, 1, 1), device=H.device)
        
        for _ in range(self.max_iter):
            phi = torch.sigmoid(H + lambd)
            current_mass = phi.sum(dim=(-1, -2, -3), keepdim=True)
            
            diff = current_mass - target_mass
            
            if diff.abs().max() < self.tol:
                break
                
            # Derivata rispetto a lambda: sum( phi * (1 - phi) )
            deriv = (phi * (1.0 - phi)).sum(dim=(-1, -2, -3), keepdim=True)
            
            # Newton-Raphson update
            lambd = lambd - diff / (deriv + 1e-8)
            
        return torch.sigmoid(H + lambd)
       
 
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
                H_raw = self.toOut(self.hidden_list[-1])
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
         super(PersistentModel3D, self).__init__(*args, **kwargs) # simply initialize the model as we know
         self.hidden_list = None # hidden_list is now an attribute
         self.reduce_out = True ###
         self.squash_out = True ###
         self.conservative = False

    def set_hidden(
         self,
         in_sequence     : torch.Tensor
         ) -> None:
         '''
         This method is simply setting the hidden state to zero (to be used to erase memory or as an initialization step
         '''
         print('Resetting hidden state...', end='', flush=True)
         self.hidden_list = []

         batch_size = in_sequence.size(0)
         nx = in_sequence.size(3)
         ny = in_sequence.size(4)
         nz = in_sequence.size(5)

         for ll in range(self.hidden_units):
             self.hidden_list.append(
                 torch.zeros(
                     batch_size,
                     self.hidden_channels,
                     nx,
                     ny,
                     nz,
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
                            self.hidden_list[kk][example,channel,:,:,:] = self.hidden_list[kk][example,channel,:,:,:]*self.dropout_mask[example, kk, channel] # this will zero-out some of the hidden shapes

            if self.reduce_out:
                H_raw = self.toOut(self.hidden_list[-1])
                
                
                if self.squash_out:
                    # Nuova versione Allen-Cahn con moltiplicatore di Lagrange per conservare la massa
                    H_raw = self.toOut(hidden_list[-1])
                    phi_t = input_t_old.squeeze(1)
            
                    target_mass = phi_t.sum(dim=(-1,-2,-3), keepdim=True)
            
                    # Il layer applica il sigmoide e shifta il potenziale per conservare la massa
                    output = self.mass_cons_sigmoid(H_raw, target_mass)
            else:
                output = self.hidden_list[-1]
                output = input_t_old.squeeze(1)+self.divergence(output)

            outputs += [output]

            if noise_reg != 0:
                noise   = noise_reg*torch.randn(output.shape, device=output.device)
                noise   = noise - torch.mean(noise, dim=(-1,-2,-3), keepdim=True)
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
                            self.hidden_list[kk][example,channel,:,:,:] = self.hidden_list[kk][example,channel,:,:,:]*self.dropout_mask[example, kk, channel] # this will zero-out some of the hidden shapes

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
                noise = noise - torch.mean(noise, dim=(-1,-2,-3), keepdim=True)
                output = output + noise

        outputs = torch.stack(outputs, dim=1)

        return outputs


