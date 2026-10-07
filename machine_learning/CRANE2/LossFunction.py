import torch
import torch.nn as nn
import torch.nn.functional as F

class CahnHilliardLoss(nn.Module):
    """
    Loss per un campo di Cahn-Hilliard 3D.

    Input:
        pred, target: (B, T, C, Nx, Ny, Nz)

    Coordinate fisiche:
        asse -3 -> x
        asse -2 -> y
        asse -1 -> z

    Energia libera:
        E(phi) = integral [W(phi) + epsilon/2 |grad phi|^2] dV

    Potenziale chimico:
        mu(phi) = dW/dphi - epsilon * laplacian(phi)
        
    Termini aggiuntivi:
        - energy_matching_loss: MSE tra E_pred ed E_true
        - gradient_matching_loss: MSE tra i gradienti spaziali (Norma H1)
        - mu_matching_loss: MSE tra il potenziale chimico predetto e target
    """

    def __init__(
        self,
        w_mse=1.0,
        w_grad=0.0,      # Peso per il Gradient Matching
        w_energy=0.0,    # Peso per l'Energy Matching
        w_mu=0.0,        # Peso per il Chemical Potential Matching
        w_bounds=0.0,    # Lasciato a 0 se l'architettura gestisce già i limiti
        bounds_peak_weight=0.05,
        epsilon=0.1,
        dx=0.025,
        dy=0.025,
        dz=0.025
    ):
        super().__init__()

        self.w_mse = w_mse
        self.w_grad = w_grad
        self.w_energy = w_energy
        self.w_mu = w_mu
        self.w_bounds = w_bounds

        self.bounds_peak_weight = bounds_peak_weight
        self.epsilon = epsilon
        self.dx = dx
        self.dy = dy
        self.dz = dz

    def _reshape_spatial_3d(self, c):
        if c.ndim != 6:
            raise ValueError(f"Expected 6D tensor, got {tuple(c.shape)}.")
        B, T, C, Nx, Ny, Nz = c.shape
        return c.reshape(B * T, C, Nx, Ny, Nz)

    @staticmethod
    def _restore_spatial_3d(c_reshaped, original_shape):
        return c_reshaped.reshape(original_shape)

    def pad_neumann(self, c):
        c_reshaped = self._reshape_spatial_3d(c)
        return F.pad(c_reshaped, (1, 1, 1, 1, 1, 1), mode="replicate")

    def gradient(self, c):
        original_shape = c.shape
        c_pad = self.pad_neumann(c)

        gx = (c_pad[:, :, 2:, 1:-1, 1:-1] - c_pad[:, :, :-2, 1:-1, 1:-1]) / (2.0 * self.dx)
        gy = (c_pad[:, :, 1:-1, 2:, 1:-1] - c_pad[:, :, 1:-1, :-2, 1:-1]) / (2.0 * self.dy)
        gz = (c_pad[:, :, 1:-1, 1:-1, 2:] - c_pad[:, :, 1:-1, 1:-1, :-2]) / (2.0 * self.dz)

        return (
            self._restore_spatial_3d(gx, original_shape),
            self._restore_spatial_3d(gy, original_shape),
            self._restore_spatial_3d(gz, original_shape),
        )

    def laplacian(self, c):
        original_shape = c.shape
        c_pad = self.pad_neumann(c)

        center = c_pad[:, :, 1:-1, 1:-1, 1:-1]
        x_plus = c_pad[:, :, 2:, 1:-1, 1:-1]
        x_minus = c_pad[:, :, :-2, 1:-1, 1:-1]
        y_plus = c_pad[:, :, 1:-1, 2:, 1:-1]
        y_minus = c_pad[:, :, 1:-1, :-2, 1:-1]
        z_plus = c_pad[:, :, 1:-1, 1:-1, 2:]
        z_minus = c_pad[:, :, 1:-1, 1:-1, :-2]

        lap = (
            (x_plus - 2.0 * center + x_minus) / (self.dx ** 2)
            + (y_plus - 2.0 * center + y_minus) / (self.dy ** 2)
            + (z_plus - 2.0 * center + z_minus) / (self.dz ** 2)
        )

        return self._restore_spatial_3d(lap, original_shape)

    def W(self, phi):
        return (18.0 / self.epsilon) * phi.square() * (1.0 - phi).square()

    def W_prime(self, phi):
        """Derivata prima del potenziale a doppia buca dW/dphi"""
        return (36.0 / self.epsilon) * phi * (1.0 - phi) * (1.0 - 2.0 * phi)

    def free_energy(self, phi):
        w_local = self.W(phi)
        gx, gy, gz = self.gradient(phi)
        grad_sq = gx.square() + gy.square() + gz.square()
        density = w_local + (self.epsilon/2) * grad_sq
        dV = self.dx * self.dy * self.dz
        return density.sum(dim=(-3, -2, -1)) * dV

    def chemical_potential(self, phi):
        """mu = dW/dphi - epsilon^2 * laplacian(phi)"""
        w_p = self.W_prime(phi)
        lap = self.laplacian(phi)
        return w_p - self.epsilon * lap

    def mse_loss(self, pred, target):
        return F.mse_loss(pred, target)

    def gradient_matching_loss(self, pred, target):
        gx_p, gy_p, gz_p = self.gradient(pred)
        gx_t, gy_t, gz_t = self.gradient(target)
        
        l_gx = F.mse_loss(gx_p, gx_t)
        l_gy = F.mse_loss(gy_p, gy_t)
        l_gz = F.mse_loss(gz_p, gz_t)
        
        return l_gx + l_gy + l_gz

    def energy_matching_loss(self, pred, target):
        energy_pred = self.free_energy(pred)
        energy_true = self.free_energy(target)
        return F.mse_loss(energy_pred, energy_true)

    def mu_matching_loss(self, pred, target):
        mu_pred = self.chemical_potential(pred)
        mu_target = self.chemical_potential(target)
        return F.mse_loss(mu_pred, mu_target)

    def bounds_loss(self, pred):
        below = F.relu(-pred)
        above = F.relu(pred - 1.0)
        violation_sq = below.square() + above.square()
        mean_term = violation_sq.mean()
        peak_term = violation_sq.amax(dim=(-3, -2, -1)).mean()
        return mean_term + self.bounds_peak_weight * peak_term

    def forward(self, pred, target):
        l_mse = self.mse_loss(pred, target)
        
        # Calcola i termini solo se richiesti (per risparmiare computazione)
        l_grad = self.gradient_matching_loss(pred, target)
        l_energy = self.energy_matching_loss(pred, target)
        l_mu = self.mu_matching_loss(pred, target)
        l_bounds = self.bounds_loss(pred)

        total = (
            self.w_mse * l_mse
            + self.w_grad * l_grad
            + self.w_energy * l_energy
            + self.w_mu * l_mu
            + self.w_bounds * l_bounds
        )

        metrics = {
            "loss_total": total.detach(),
            "loss_mse": l_mse.detach(),
            "loss_grad": l_grad.detach(),
            "loss_energy": l_energy.detach(),
            "loss_mu": l_mu.detach(),
            "loss_bounds": l_bounds.detach(),
        }

        return total, metrics
