# <<< import external stuff <<<
import torch
import torch.nn as nn
import torch.nn.functional as F
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
from src.classes import *
from src.utils import *
from src.dataloaders import give_dataloaders, give_3D_dataloaders
from src.parser import TrainingParser
# --- import my stuff ---

# <<< training function <<<
def train(model, loss_fn, optimizer, loaders, args):


    '''
    This function trains the model given selected loss function
    '''
    valid_losses =      []
    valid_mse_losses=   []
    valid_e_losses =    []
    valid_grad_losses = []
    valid_massW_losses = []
    train_losses =      []
    
    train_loader, valid_loader = loaders
    
    len_train_loader = len(train_loader)
    len_valid_loader = len(valid_loader)
    
    for epoch in range(args.epochs):
        
        start_epoch = time.time()
        log_epoch_start_info(epoch, args)
        
        optimizer.zero_grad()
        
        epoch_train_losses = []
        
        model.train()
        
        # <<< training loop <<<
        for j, series_with_params in enumerate(train_loader):
            
            if args.num_params != 0:
                series = series_with_params[0]
                params = series_with_params[1]
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
                    temp_lr = ((j+1)/(len_train_loader+1))*args.lr
                    g['lr'] = temp_lr
                    print(f'Learning rate updated to: {temp_lr:.4e}')
            elif epoch == 1 and args.reload:
                for g in optimizer.param_groups:
                    g['lr'] = args.lr
            
            if args.ramp:
                in_seq_length = int( args.subseq_max*(1-(epoch+args.start_ramp)/args.ramp_length) )
                in_seq_length = min(series.shape[1]-1, in_seq_length)
                in_seq_length = max(args.subseq_min, in_seq_length)
            else:
                in_seq_length   = np.random.randint( args.subseq_min, args.subseq_max+1 ) 
            
            future          = series.shape[1]-in_seq_length-1
            
            if j%args.logfreq == 0 and not args.extract_param: # <- print sub-epoch infos
                print(f'Passing example[{j}/{len_train_loader-1}] in epoch {epoch} with {future} f-frames')
            elif j%args.logfreq == 0:
                print(f'Passing example[{j}/{len_train_loader-1}] in epoch {epoch}')
            
            input_data  = clip_series(series, in_seq_length).to(args.device)
            
            if not args.extract_param:
                target_data = series[:,1:,:,:,:].to(args.device)
            
                if args.dual:
                    input_data  = withdual(input_data)
                    target_data = withdual(target_data)
                    if params is not None:
                        for pp, param in enumerate(params):
                            params[pp] = torch.cat([params[pp], params[pp]])
                        
            else:
                if args.dual:
                    input_data  = withdual(input_data)
                    if params is not None:
                        for pp, param in enumerate(params):
                            params[pp] = torch.cat([params[pp], params[pp]])
                            
                target_data = torch.cat([p.unsqueeze(1) for p in params], dim=1).to(args.device)
                target_data = target_data.float()
                
            if not args.extract_param:
                y_pred = model(input_data, future=future, params=params, noise_reg=args.noise_reg, approx_inference=False)
            else:
                y_pred = model(input_data, noise_reg=args.noise_reg, approx_inference=False)
                
            tot_loss, mse_loss, e_loss, grad_loss, massW_loss = loss_fn(y_pred, target_data)
            loss = tot_loss
            loss.backward()
            
            #loss = loss_fn(y_pred, target_data)
            #loss.backward()
            
            if j%args.superbatch == 0 or j==len_train_loader-1:
                optimizer.step()
                optimizer.zero_grad()
            
            loss4print = loss.item()
            
            epoch_train_losses.append( loss4print )
            
            if j%args.logfreq == 0:
                print(f'Loss: {loss4print:.4e} \t Running mean loss: {np.mean(epoch_train_losses):.4e}')
                
        train_losses.append( np.mean(epoch_train_losses) )
        with open( f'{args.paths["trainloss"]}', 'a+') as train_loss_file:
            train_loss_file.write(f'{train_losses[-1]}\n')
        # --- training loop ---
        
        # <<< validation loop <<<
        with torch.no_grad():
            
            model.eval()
            
            epoch_valid_losses =        []
            epoch_valid_mse_losses =    []
            epoch_valid_e_losses =      []
            epoch_valid_grad_losses =   []
            epoch_valid_massW_losses =  []
            
            y_preds = []
            y_trues = []
            
            for j, series_with_params in enumerate(valid_loader):
                
                if args.num_params != 0:
                    series = series_with_params[0]
                    params = series_with_params[1]
                else:
                    params = None
                
                if j >= 3 and args.debug:
                    print('Breaking because of DEBUG mode.')
                    break
                
                in_seq_length   = args.subseq_min # this should make validation always as hard as possible
                future          = series.shape[1]-in_seq_length-1
                    
                
                if not args.extract_param:
                    input_data  = clip_series(series, in_seq_length).to(args.device)
                    target_data = series[:,1:,:,:,:].to(args.device)
                
                    if args.dual:
                        input_data  = withdual(input_data)
                        target_data = withdual(target_data)
                        if params is not None:
                            for pp, param in enumerate(params):
                                params[pp] = torch.cat([params[pp], params[pp]])
                        
                else:
                    input_data  = series.to(args.device)
                    
                    if args.dual:
                        input_data  = withdual(input_data)
                        if params is not None:
                            for pp, param in enumerate(params):
                                params[pp] = torch.cat([params[pp], params[pp]])
                                
                    target_data = torch.cat([p.unsqueeze(1) for p in params], dim=1).to(args.device)
                    target_data = target_data.float()
                    
                    
                if not args.extract_param:
                    y_pred  = model(input_data, future=future, params=params)
                else:
                    y_pred  = model(input_data)
                    y_preds.append(y_pred.detach().cpu())
                    y_trues.append(target_data.detach().cpu())
                
                #loss = loss_fn(y_pred, target_data)
                #loss4print = loss.item()
                
                tot_loss, mse_loss, e_loss, grad_loss, massW_loss = loss_fn(y_pred, target_data)
                loss = tot_loss
                loss4print = loss.item()
                
                epoch_valid_losses.append( loss4print )
                epoch_valid_mse_losses.append(mse_loss.detach().cpu().item())
                epoch_valid_e_losses.append(e_loss.detach().cpu().item())
                epoch_valid_grad_losses.append(grad_loss.detach().cpu().item())
                epoch_valid_massW_losses.append(massW_loss.detach().cpu().item())
                
            valid_losses.append( np.mean(epoch_valid_losses) )
            valid_mse_losses.append(np.mean(epoch_valid_mse_losses))
            valid_e_losses.append(np.mean(epoch_valid_e_losses))
            valid_grad_losses.append(np.mean(epoch_valid_grad_losses))
            valid_massW_losses.append(np.mean(epoch_valid_massW_losses))
            
            with open( f'{args.paths["validloss"]}', 'a+') as valid_loss_file:
                valid_loss_file.write(f'{valid_losses[-1]}\n')
                
            validloss_dir = os.path.dirname(args.paths["validloss"])
            valid_terms_path = os.path.join(validloss_dir, 'valid_loss_terms.txt')

            with open(valid_terms_path, 'a+') as valid_terms_file:
                valid_terms_file.write(
                    f'{valid_losses[-1]}\t'
                    f'{valid_mse_losses[-1]}\t'
                    f'{valid_e_losses[-1]}\t'
                    f'{valid_massW_losses[-1]}\t'
                    f'{valid_grad_losses[-1]}\n'
                )
            
        optimizer.zero_grad() # <- better safe than sorry
        # --- validation loop ---
        
        # <<< graphic output <<<
        if args.graphics and not args.threeD and not args.extract_param:
            
            y_pred_cpu = y_pred.detach().cpu()
            target_data_cpu = target_data.detach().cpu()
            
            im_last_pred    = y_pred_cpu[0,-1,:,:,:].permute(1,2,0)
            im_last_target  = target_data_cpu[0,-1,:,:,:].permute(1,2,0)
            
            out_png(
                im_pred     = im_last_pred,
                im_target   = im_last_target,
                path        = f'{args.paths["png"]}/epoch_{epoch}.png',
                cmap        = args.cmap
                )
            
            make_lossplot(train_losses, valid_losses, args)
            
            if args.gifs:
                out_gifs(
                    y_pred      = y_pred_cpu,
                    path        = f'{args.paths["gif"]}/epoch_{epoch}.gif'
                    )
                
                out_gifs(
                    y_pred      = target_data_cpu,
                    path        = f'{args.paths["gif"]}/epoch_{epoch}_TRUE.gif'
                    )
                
            if args.vtk:
                
                epoch_path = f'{args.paths["vtk"]}/epoch_{epoch}'
                epoch_path_TRUE = f'{args.paths["vtk"]}/epoch_{epoch}_TRUE'
                
                os.mkdir(epoch_path)
                os.mkdir(epoch_path_TRUE)
                
                seq2vtk(
                    y_pred  = y_pred_cpu,
                    path    = epoch_path
                    )
                
                seq2vtk(
                    y_pred  = target_data_cpu,
                    path    = epoch_path_TRUE
                    )
                
                del epoch_path, epoch_path_TRUE
                
            if args.npy:
                
                epoch_path = f'{args.paths["npy"]}/epoch_{epoch}'
                epoch_path_TRUE = f'{args.paths["npy"]}/epoch_{epoch}_TRUE'
                
                os.mkdir(epoch_path)
                os.mkdir(epoch_path_TRUE)
                
                seq2npy(
                    y_pred  = y_pred_cpu,
                    path    = epoch_path
                    )
                
                seq2npy(
                    y_pred  = target_data_cpu,
                    path    = epoch_path_TRUE
                    )
                
                del epoch_path, epoch_path_TRUE
                
                
        elif args.graphics and args.threeD and args.vtk:
            
            make_lossplot(train_losses, valid_losses, args)
            
            y_pred_cpu = y_pred.detach().cpu()
            target_data_cpu = target_data.detach().cpu()
            
            epoch_path = f'{args.paths["gif"]}/epoch_{epoch}'
            epoch_path_TRUE = f'{args.paths["gif"]}/epoch_{epoch}_TRUE'
            
            os.mkdir( epoch_path )
            os.mkdir( epoch_path_TRUE )
            
            seq2vtk(
                y_pred      = y_pred_cpu,
                path        = epoch_path
                )
            
            seq2vtk(
                y_pred      = target_data_cpu,
                path        = epoch_path_TRUE
                )
            
        elif args.graphics and args.extract_param:
            
            make_lossplot(train_losses, valid_losses, args)
            
            for kk in range(args.num_params):
                
                preds = []
                trues = []
                
                for y_pred, target_data in zip(y_preds, y_trues):
                    for bb in range(y_pred.shape[0]):
                        preds.append( y_pred[bb,kk] )
                        trues.append( target_data[bb,kk] )
                        
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
        epoch_time = end_epoch-start_epoch
        
        if not args.extract_param:
            log_epoch_end_info(epoch, epoch_time, (y_pred, target_data), train_losses[-1], valid_losses[-1], args)
        
        save_model(
            model   = model,
            path    = f'{args.paths["model"]}/epoch_{epoch}.pt'
            )
        # --- epoch end logging ---


# <<< definition of loss functions <<<
# def compute_e(c, eps, dx):
#     w = (18.0 / eps) * c**2 * (1.0 - c)**2
#     gx = (c.roll(-1, dims=-1) - c.roll(1, dims=-1)) / (2.0*dx)
#     gy = (c.roll(-1, dims=-2) - c.roll(1, dims=-2)) / (2.0*dx)
#     grad_sq = gx**2 + gy**2
#     f = 0.5*eps*grad_sq + w
    
#     return f.sum(dim=(-1,-2)) * dx**2

# def compute_e_neumann(c, eps, dx):
#     w = (18.0 / eps) * c**2 * (1.0 - c)**2

#     #pbc along x
#     gx = (c.roll(-1, dims=-1) - c.roll(1, dims=-1)) / (2.0*dx)

#     #neumann bc along y
#     H = c.shape[-2]
#     gy = torch.zeros_like(c)

#     #internal points y
#     gy[..., 1:-1, :] = (c[..., 2:, :] - c[..., :-2, :]) / (2.0*dx)

#     #borders
#     gy[..., 0, :] = 0.0
#     gy[..., -1, :] = 0.0

#     grad_sq = gx**2 + gy**2
#     f = 0.5*eps*grad_sq + w

#     return f.sum(dim=(-1,-2)) * dx**2

# def bulk_grad_penalty(x, y, dx):
#     gx = (x.roll(-1, dims=-1) - x.roll(1, dims=-1)) / (2.0*dx)
#     gy = (x.roll(-1, dims=-2) - x.roll(1, dims=-2)) / (2.0*dx)
#     grad_sq = gx**2 + gy**2

#     # compute the penalty term respect to the bulk of the true frame y
#     bulk_mask = 1.0 - 16.0 * y**2 * (1.0 - y)**2
#     bulk_mask = torch.clamp(bulk_mask, 0.0, 1.0)

#     return (bulk_mask * grad_sq).mean()

# def bulk_grad_penalty_neumann(x, y, dx):
#     #pbc along x
#     gx = (x.roll(-1, dims=-1) - x.roll(1, dims=-1)) / (2.0*dx)

#     #neumann bc along y
#     H = y.shape[-2]
#     gy = torch.zeros_like(y)

#     #internal points y
#     gy[..., 1:-1, :] = (y[..., 2:, :] - y[..., :-2, :]) / (2.0*dx)

#     #borders
#     gy[..., 0, :] = 0.0
#     gy[..., -1, :] = 0.0

#     grad_sq = gx**2 + gy**2

#     # compute the penalty term respect to the bulk of the true frame y
#     bulk_mask = 1.0 - 16.0 * y**2 * (1.0 - y)**2
#     bulk_mask = torch.clamp(bulk_mask, 0.0, 1.0)

#     return (bulk_mask * grad_sq).mean()
    

# def e_penalty(x, y, eps, dx):
#     e_pred = compute_e_neumann(x, eps, dx)
#     e_true = compute_e_neumann(y, eps, dx)
    
#     return nn.MSELoss()(e_pred, e_true)


# def tot_loss_fn(x, y, eps, dx, coeffE, coeffG, massW):
#     grad_loss = bulk_grad_penalty_neumann(x, y, dx)
#     e_loss = e_penalty(x, y, eps, dx)
#     mse_loss = nn.MSELoss()(x,y)
    
#     mass_loss = massW*nn.MSELoss()(
#                 torch.mean( x, axis=(-1,-2) ),
#                 torch.mean( y, axis=(-1,-2) )
#                 )
#     tot_loss = mse_loss + coeffE*e_loss + coeffG*grad_loss + massW*mass_loss
    
#     return tot_loss, mse_loss, e_loss, grad_loss, mass_loss

class CahnHilliardCompositeLoss(nn.Module):
    """
    Loss composita per surrogate ConvGRU della Cahn-Hilliard equation.

    BC:
    - x: periodiche
    - y: Neumann omogenee

    Input:
    pred, target -> (B, T, C, H, W)

    Output:
    total, mse, energy, grad, mass, pde
    """

    def __init__(
        self,
        w_mse=1.0,
        w_grad=0.1,
        w_mass=1.0,
        w_pde=0.0,
        w_energy=0.05,
        gamma=1.0,
        D=1.0,
        dx=1.0,
        dy=1.0,
        dt=1.0,
    ):
        super().__init__()
        self.w_mse = w_mse
        self.w_grad = w_grad
        self.w_mass = w_mass
        self.w_pde = w_pde
        self.w_energy = w_energy
        self.gamma = gamma
        self.D = D
        self.dx = dx
        self.dy = dy
        self.dt = dt

    def _reshape_spatial(self, c):
        return c.reshape(-1, 1, c.shape[-2], c.shape[-1])

    def pad_mixed_bc(self, c):
        c_ = self._reshape_spatial(c)

        # periodic in x
        c_pad_x = F.pad(c_, (1, 1, 0, 0), mode="circular")

        # Neumann in y
        c_pad = F.pad(c_pad_x, (0, 0, 1, 1), mode="replicate")
        return c_pad

    def laplacian(self, c):
        c_pad = self.pad_mixed_bc(c)

        center = c_pad[:, :, 1:-1, 1:-1]
        left   = c_pad[:, :, 1:-1, 0:-2]
        right  = c_pad[:, :, 1:-1, 2:]
        up     = c_pad[:, :, 0:-2, 1:-1]
        down   = c_pad[:, :, 2:, 1:-1]

        lap = (left - 2.0 * center + right) / (self.dx ** 2) + \
              (up   - 2.0 * center + down)  / (self.dy ** 2)

        return lap.reshape(c.shape)

    def gradient(self, c):
        c_pad = self.pad_mixed_bc(c)

        gx = (c_pad[:, :, 1:-1, 2:] - c_pad[:, :, 1:-1, 0:-2]) / (2.0 * self.dx)
        gy = (c_pad[:, :, 2:, 1:-1] - c_pad[:, :, 0:-2, 1:-1]) / (2.0 * self.dy)

        return gx.reshape(c.shape), gy.reshape(c.shape)

    def mse_loss(self, pred, target):
        return F.mse_loss(pred, target)

    def gradient_loss(self, pred, target):
        gx_p, gy_p = self.gradient(pred)
        gx_t, gy_t = self.gradient(target)
        return F.l1_loss(gx_p, gx_t) + F.l1_loss(gy_p, gy_t)

    def mass_conservation_loss(self, pred, target):
        mass_pred = pred.sum(dim=(-1, -2)) * (self.dx * self.dy)
        mass_true = target.sum(dim=(-1, -2)) * (self.dx * self.dy)
        return F.mse_loss(mass_pred, mass_true)

    def pde_residual_loss(self, pred):
        dc_dt = (pred[:, 1:] - pred[:, :-1]) / self.dt
        mu = pred ** 3 - pred - self.gamma * self.laplacian(pred)
        rhs = self.D * self.laplacian(mu)
        residual = dc_dt - rhs[:, :-1]
        return torch.mean(residual ** 2)

    def free_energy(self, c):
        bulk = 0.25 * (c ** 2 - 1.0) ** 2
        gx, gy = self.gradient(c)
        grad_term = 0.5 * self.gamma * (gx ** 2 + gy ** 2)
        density = bulk + grad_term
        return density.sum(dim=(-1, -2)) * (self.dx * self.dy)

    def free_energy_loss(self, pred):
        F_t = self.free_energy(pred)
        dF = F_t[:, 1:] - F_t[:, :-1]
        violation = torch.relu(dF)
        return torch.mean(violation ** 2)

    def forward(self, pred, target):
        l_mse = self.mse_loss(pred, target)
        l_grad = self.gradient_loss(pred, target)
        l_mass = self.mass_conservation_loss(pred, target)
        l_pde = self.pde_residual_loss(pred)
        l_energy = self.free_energy_loss(pred)

        total = (
            self.w_mse * l_mse +
            self.w_grad * l_grad +
            self.w_mass * l_mass +
            self.w_pde * l_pde +
            self.w_energy * l_energy
        )

        return total, l_mse, l_energy, l_grad, l_mass, l_pde
    
    
# <<< main function <<<
def main():
    '''
    Main function: istantiation of models and dataloaders and launcing of training function
    '''
    
    #Parse arguments
    parser  = TrainingParser()
    args    = parser.parse_args()
    
    # crate folder structure
    args = build_train_logs_dir_tree(args)
    
    # Set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Instantiate dataloaders
    if not args.threeD and not args.extract_param:
        dataloaders = give_dataloaders(args)
        model_class = ConvGRU
    elif args.extract_param:
        dataloaders = give_dataloaders(args)
        model_class = ConvGRUClassifier
    elif args.threeD and not args.extract_param:
        dataloaders = give_3D_dataloaders(args)
        model_class = ConvGRU3D
    elif args.threeD:
        raise NotImplementedError('train.py was not able to recognize the training mode. Aborting.')
    
    train_loader = dataloaders["train_set"]
    valid_loader = dataloaders["valid_set"]
    
    # Define model and put to device
    model = model_class(
        hidden_units        = args.hidden,
        input_channels      = 1, # this is hardcoded for the moment... waiting for multidimensional data!
        output_channels     = None if not args.extract_param else args.num_params,
        hidden_channels     = args.channels,
        kernel_size         = args.kernel_size,
        padding_mode        = args.padding,
        separable           = False,
        bias                = args.bias,
        divergence          = args.divergence,
        conservative        = args.conservative,
        num_params          = args.num_params if not args.extract_param else 0,
        dropout             = args.dropout,
        dropout_prob        = args.dropout_prob
        )
    
    print_model_info(model)
    
    if args.divergence:
        model.make_div_filters( torch.zeros(1, device=args.device) )
    
    #model = torch.compile(model)
    
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
        lr              = args.lr,
        weight_decay    = args.weightd
        )
    
    # define loss function
    if args.extract_param:
        loss_fn = nn.MSELoss()
    elif not args.threeD:
        loss_fn = CahnHilliardCompositeLoss(
            w_mse=1.0,
            w_grad=args.coeffG,
            w_mass=args.massW,
            w_pde=args.coeffPDE,
            w_energy=args.coeffE,
            gamma=args.eps,
            D=args.mobility,
            dx=args.dx,
            dy=args.dy,
            dt=args.dt,
        )
    else:
        loss_fn = lambda x,y: \
            nn.MSELoss()(x,y) + args.massW*nn.MSELoss()(
                torch.mean( x, axis=(-1,-2,-3) ),
                torch.mean( y, axis=(-1,-2,-3) )
            )
                
    # training loop
    train(model, loss_fn, optimizer, (train_loader, valid_loader), args)
# --- main function ---

# <<< main calling <<<
if __name__ == '__main__':    
    main()
# --- main calling ---
