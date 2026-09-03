import torch
import torch.nn as nn
import torch.nn.functional as F


class CahnHilliarLoss(nn.Module):
    """
    Loss composita coerente con:
    - W(phi) = (18/epsilon) * phi^2 * (1-phi)^2
    - E = int [ W(phi) + epsilon^2 * |grad phi|^2 ] dx
    - M(phi) = M0 * (36/epsilon) * phi^2 * (1-phi)^2
    - BC: periodiche in x, Neumann omogenee in y

    pred, target: (B, T, C, H, W)
    """

    def __init__(
        self,
        w_mse=1.0,
        w_energy=0.05,
        w_grad=0.1,
        w_pde=0.0,
        epsilon=1.0,
        M0=1.0,
        dx=1.0,
        dy=1.0,
        dt=1.0,
    ):
        super().__init__()
        self.w_mse = w_mse
        self.w_energy = w_energy
        self.w_grad = w_grad
        self.w_pde = w_pde
        self.epsilon = epsilon
        self.M0 = M0
        self.dx = dx
        self.dy = dy
        self.dt = dt

    def _reshape_spatial(self, c):
        return c.reshape(-1, 1, c.shape[-2], c.shape[-1])

    def pad_mixed_bc(self, c):
        c_ = self._reshape_spatial(c)
        c_pad_x = F.pad(c_, (1, 1, 0, 0), mode="circular")
        c_pad = F.pad(c_pad_x, (0, 0, 1, 1), mode="replicate")
        return c_pad

    def gradient(self, c):
        c_pad = self.pad_mixed_bc(c)
        gx = (c_pad[:, :, 1:-1, 2:] - c_pad[:, :, 1:-1, 0:-2]) / (2.0 * self.dx)
        gy = (c_pad[:, :, 2:, 1:-1] - c_pad[:, :, 0:-2, 1:-1]) / (2.0 * self.dy)
        return gx.reshape(c.shape), gy.reshape(c.shape)

    def divergence(self, jx, jy):
        jx_pad = self.pad_mixed_bc(jx)
        jy_pad = self.pad_mixed_bc(jy)

        djx_dx = (jx_pad[:, :, 1:-1, 2:] - jx_pad[:, :, 1:-1, 0:-2]) / (2.0 * self.dx)
        djy_dy = (jy_pad[:, :, 2:, 1:-1] - jy_pad[:, :, 0:-2, 1:-1]) / (2.0 * self.dy)

        div = djx_dx + djy_dy
        return div.reshape(jx.shape)

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

    def W(self, phi):
        return (18.0 / self.epsilon) * phi**2 * (1.0 - phi)**2

    def dW_dphi(self, phi):
        # derivata di (18/eps) * phi^2 * (1-phi)^2
        return (36.0 / self.epsilon) * phi * (1.0 - phi) * (1.0 - 2.0 * phi)

    def M(self, phi):
        return self.M0 * (36.0 / self.epsilon) * phi**2 * (1.0 - phi)**2

    def free_energy(self, phi):
        w_local = self.W(phi)
        gx, gy = self.gradient(phi)
        grad2 = gx**2 + gy**2
        density = w_local + (self.epsilon ** 2) * grad2
        return density.sum(dim=(-1, -2)) * (self.dx * self.dy)

    def chemical_potential(self, phi):
        # Coerente con E = int [W(phi) + eps^2 |grad phi|^2] dx
        # delta/delta phi [eps^2 |grad phi|^2] = -2 eps^2 Delta phi
        return self.dW_dphi(phi) - 2.0 * (self.epsilon ** 2) * self.laplacian(phi)

    def pde_residual_loss(self, pred):
        dphi_dt = (pred[:, 1:] - pred[:, :-1]) / self.dt

        phi_t = pred[:, :-1]
        mu = self.chemical_potential(phi_t)

        gx_mu, gy_mu = self.gradient(mu)
        mobility = self.M(phi_t)

        jx = mobility * gx_mu
        jy = mobility * gy_mu

        rhs = self.divergence(jx, jy)

        residual = dphi_dt - rhs
        return torch.mean(residual ** 2)

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

    def free_energy_loss(self, pred):
        E_t = self.free_energy(pred)
        dE = E_t[:, 1:] - E_t[:, :-1]
        violation = torch.relu(dE)
        return torch.mean(violation ** 2)

    def forward(self, pred, target):
        l_mse = self.mse_loss(pred, target)
        l_energy = self.free_energy_loss(pred)
        l_grad = self.gradient_loss(pred, target)
        l_pde = self.pde_residual_loss(pred)

        total = (
            self.w_mse * l_mse +
            self.w_energy * l_energy +
            self.w_grad * l_grad +
            self.w_pde * l_pde
        )

        return total, l_mse, l_energy, l_grad, l_mass
