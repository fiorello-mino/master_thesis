import torch
import torch.nn as nn
import torch.nn.functional as F


class CahnHilliardLoss(nn.Module):
    """
    Quantità fisiche:
    - W(phi) = (18/epsilon) * phi^2 * (1-phi)^2
    - E = int [ W(phi) + epsilon^2 * |grad phi|^2 ] dx
    - M(phi) = M0 * (36/epsilon) * phi^2 * (1-phi)^2
    - BC: Neumann in x, y, z

    pred, target: (B, T, C, x, y, z)
    """

    def __init__(
        self,
        w_mse=1.0,
        w_energy=1e-3,
        w_grad=0.0,
        w_pde=0.0,
        epsilon=0.1,
        M0=1.0,
        dx=0.025,
        dy=0.025,
        dz=0.025,
        dt=5e-3,
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
        self.dz = dz
        self.dt = dt

    def _reshape_spatial_3d(self, c):
        # c: (B, T, C, x, y, z)
        return c.reshape(-1, 1, c.shape[-3], c.shape[-2], c.shape[-1])

    def pad_neumann(self, c):
        c_ = self._reshape_spatial_3d(c)  # -> (N, 1, x, y, z)
        c_pad = F.pad(c_, (1, 1, 1, 1, 1, 1), mode="replicate")
        return c_pad

    def gradient(self, c):
        c_ = self._reshape_spatial_3d(c)  # (N, 1, x, y, z)
        c_pad = F.pad(c_, (1, 1, 1, 1, 1, 1), mode="replicate")

        gx = (
            c_pad[:, :, 2:, 1:-1, 1:-1]
            - c_pad[:, :, :-2, 1:-1, 1:-1]
        ) / (2.0 * self.dx)

        gy = (
            c_pad[:, :, 1:-1, 2:, 1:-1]
            - c_pad[:, :, 1:-1, :-2, 1:-1]
        ) / (2.0 * self.dy)

        gz = (
            c_pad[:, :, 1:-1, 1:-1, 2:]
            - c_pad[:, :, 1:-1, 1:-1, :-2]
        ) / (2.0 * self.dz)

        return (
            gx.reshape(c.shape),
            gy.reshape(c.shape),
            gz.reshape(c.shape),
        )

    def divergence(self, jx, jy, jz):
        jx_pad = F.pad(jx, (1, 1, 1, 1, 1, 1), mode="replicate")
        jy_pad = F.pad(jy, (1, 1, 1, 1, 1, 1), mode="replicate")
        jz_pad = F.pad(jz, (1, 1, 1, 1, 1, 1), mode="replicate")

        djx_dx = (jx_pad[:, :, 2:, 1:-1, 1:-1] - jx_pad[:, :, :-2, 1:-1, 1:-1]) / (2.0 * self.dx)
        djy_dy = (jy_pad[:, :, 1:-1, 2:, 1:-1] - jy_pad[:, :, 1:-1, :-2, 1:-1]) / (2.0 * self.dy)
        djz_dz = (jz_pad[:, :, 1:-1, 1:-1, 2:] - jz_pad[:, :, 1:-1, 1:-1, :-2]) / (2.0 * self.dz)

        div = djx_dx + djy_dy + djz_dz
        return div.reshape(jx.shape)

    def laplacian(self, c):
        c_ = self._reshape_spatial_3d(c)
        c_pad = F.pad(c_, (1, 1, 1, 1, 1, 1), mode="replicate")

        center  = c_pad[:, :, 1:-1, 1:-1, 1:-1]
        x_plus  = c_pad[:, :, 2:, 1:-1, 1:-1]
        x_minus = c_pad[:, :, :-2, 1:-1, 1:-1]
        y_plus  = c_pad[:, :, 1:-1, 2:, 1:-1]
        y_minus = c_pad[:, :, 1:-1, :-2, 1:-1]
        z_plus  = c_pad[:, :, 1:-1, 1:-1, 2:]
        z_minus = c_pad[:, :, 1:-1, 1:-1, :-2]

        lap = (
            (x_plus - 2.0 * center + x_minus) / (self.dx ** 2) +
            (y_plus - 2.0 * center + y_minus) / (self.dy ** 2) +
            (z_plus - 2.0 * center + z_minus) / (self.dz ** 2)
        )
        return lap.reshape(c.shape)

    def W(self, phi):
        return (18.0 / self.epsilon) * phi**2 * (1.0 - phi)**2

    def dW_dphi(self, phi):
        return (36.0 / self.epsilon) * phi * (1.0 - phi) * (1.0 - 2.0 * phi)

    def M(self, phi):
        return self.M0 * (36.0 / self.epsilon) * phi**2 * (1.0 - phi)**2

    def free_energy(self, phi):
        w_local = self.W(phi)
        gx, gy, gz = self.gradient(phi)
        grad2 = gx**2 + gy**2 + gz**2
        density = w_local + (self.epsilon ** 2) * grad2
        return density.sum(dim=(-3, -2, -1)) * (self.dx * self.dy * self.dz)

    def chemical_potential(self, phi):
        return self.dW_dphi(phi) - 2.0 * (self.epsilon ** 2) * self.laplacian(phi)

    def pde_residual_loss(self, pred):
        dphi_dt = (pred[:, 1:] - pred[:, :-1]) / self.dt

        phi_t = pred[:, :-1]
        mu = self.chemical_potential(phi_t)

        gx_mu, gy_mu, gz_mu = self.gradient(mu)
        mobility = self.M(phi_t)

        jx = mobility * gx_mu
        jy = mobility * gy_mu
        jz = mobility * gz_mu

        rhs = self.divergence(jx, jy, jz)

        residual = dphi_dt - rhs
        return torch.mean(residual ** 2)

    def mse_loss(self, pred, target):
        return F.mse_loss(pred, target)

    def gradient_loss(self, pred, target):
        gx_p, gy_p, gz_p = self.gradient(pred)
        gx_t, gy_t, gz_t = self.gradient(target)
        return F.l1_loss(gx_p, gx_t) + F.l1_loss(gy_p, gy_t) + F.l1_loss(gz_p, gz_t)

    def mass_conservation_loss(self, pred, target):
        mass_pred = pred.sum(dim=(-3, -2, -1)) * (self.dx * self.dy * self.dz)
        mass_true = target.sum(dim=(-3, -2, -1)) * (self.dx * self.dy * self.dz)
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
        l_mass = self.mass_conservation_loss(pred, target)

        total = (
            self.w_mse * l_mse +
            self.w_energy * l_energy +
            self.w_grad * l_grad +
            self.w_pde * l_pde
        )

        metrics = {
            "loss_total": total.detach(),
            "loss_mse": l_mse.detach(),
            "loss_energy": l_energy.detach(),
            "loss_grad": l_grad.detach(),
            "loss_pde": l_pde.detach(),
            "loss_mass": l_mass.detach(),
        }

        return total, metrics