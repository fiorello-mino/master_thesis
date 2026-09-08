import torch
import torch.nn as nn
import torch.nn.functional as F


class CahnHilliardLoss(nn.Module):
    """
    Loss per un campo di Cahn-Hilliard 3D.

    Convenzione degli input:
        pred, target: (B, T, C, Nx, Ny, Nz)

    Le coordinate fisiche sono le ultime tre dimensioni:
        asse -3 -> x
        asse -2 -> y
        asse -1 -> z

    Il reshape per gli operatori 3D conserva i canali:
        (B, T, C, Nx, Ny, Nz) -> (B*T, C, Nx, Ny, Nz)

    Energia:
        E(phi) = integral [W(phi) + epsilon^2 |grad phi|^2] dV

    Potenziale:
        W(phi) = (18 / epsilon) phi^2 (1 - phi)^2

    Mobilità:
        M(phi) = M0 (36 / epsilon) phi^2 (1 - phi)^2

    Boundary conditions:
        Neumann omogenee in x, y, z, implementate con padding replicate.
    """

    def __init__(
        self,
        w_mse=1.0,
        w_energy=1e-3,
        w_grad=0.0,
        w_pde=0.0,
        w_mass=0.0,
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
        self.w_mass = w_mass

        self.epsilon = epsilon
        self.M0 = M0
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.dt = dt

    def _reshape_spatial_3d(self, c):
        """
        (B, T, C, Nx, Ny, Nz) -> (B*T, C, Nx, Ny, Nz).

        Non fonde i canali: ogni canale resta un canale separato per F.pad
        e per gli operatori differenziali.
        """
        if c.ndim != 6:
            raise ValueError(
                "Expected a 6D tensor with shape (B, T, C, Nx, Ny, Nz), "
                f"but received shape {tuple(c.shape)}."
            )

        B, T, C, Nx, Ny, Nz = c.shape
        return c.reshape(B * T, C, Nx, Ny, Nz)

    @staticmethod
    def _restore_spatial_3d(c_reshaped, original_shape):
        """(B*T, C, Nx, Ny, Nz) -> (B, T, C, Nx, Ny, Nz)."""
        return c_reshaped.reshape(original_shape)

    def pad_neumann(self, c):
        """
        Applica un ghost layer per lato con estensione replicate.

        Input:  (B, T, C, Nx, Ny, Nz)
        Output: (B*T, C, Nx+2, Ny+2, Nz+2)
        """
        c_ = self._reshape_spatial_3d(c)
        return F.pad(c_, (1, 1, 1, 1, 1, 1), mode="replicate")

    def gradient(self, c):
        """
        Gradiente centrale del secondo ordine con BC di Neumann.

        Input:
            c: (B, T, C, Nx, Ny, Nz)

        Output:
            gx, gy, gz: ciascuno con shape (B, T, C, Nx, Ny, Nz)
        """
        original_shape = c.shape
        c_pad = self.pad_neumann(c)

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
            self._restore_spatial_3d(gx, original_shape),
            self._restore_spatial_3d(gy, original_shape),
            self._restore_spatial_3d(gz, original_shape),
        )

    def divergence(self, jx, jy, jz):
        """
        Divergenza centrale del secondo ordine con BC di Neumann.

        Input:
            jx, jy, jz: (B, T, C, Nx, Ny, Nz)

        Output:
            div(j): (B, T, C, Nx, Ny, Nz)
        """
        if jx.shape != jy.shape or jx.shape != jz.shape:
            raise ValueError(
                "jx, jy and jz must have identical shapes; received "
                f"{tuple(jx.shape)}, {tuple(jy.shape)}, {tuple(jz.shape)}."
            )

        original_shape = jx.shape

        jx_pad = self.pad_neumann(jx)
        jy_pad = self.pad_neumann(jy)
        jz_pad = self.pad_neumann(jz)

        djx_dx = (
            jx_pad[:, :, 2:, 1:-1, 1:-1]
            - jx_pad[:, :, :-2, 1:-1, 1:-1]
        ) / (2.0 * self.dx)

        djy_dy = (
            jy_pad[:, :, 1:-1, 2:, 1:-1]
            - jy_pad[:, :, 1:-1, :-2, 1:-1]
        ) / (2.0 * self.dy)

        djz_dz = (
            jz_pad[:, :, 1:-1, 1:-1, 2:]
            - jz_pad[:, :, 1:-1, 1:-1, :-2]
        ) / (2.0 * self.dz)

        div = djx_dx + djy_dy + djz_dz
        return self._restore_spatial_3d(div, original_shape)

    def laplacian(self, c):
        """
        Laplaciano 3D con stencil centrale a 7 punti e BC di Neumann.

        Input/output:
            (B, T, C, Nx, Ny, Nz)
        """
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
        """Densità di energia libera locale W(phi)."""
        return (18.0 / self.epsilon) * phi.square() * (1.0 - phi).square()

    def dW_dphi(self, phi):
        """Derivata dW/dphi."""
        return (
            (36.0 / self.epsilon)
            * phi
            * (1.0 - phi)
            * (1.0 - 2.0 * phi)
        )

    def M(self, phi):
        """Mobilità degenere M(phi)."""
        return self.M0 * (36.0 / self.epsilon) * phi.square() * (1.0 - phi).square()

    def free_energy(self, phi):
        """
        Energia libera integrata per batch, timestep e canale.

        Output shape: (B, T, C)
        """
        w_local = self.W(phi)
        gx, gy, gz = self.gradient(phi)

        grad_sq = gx.square() + gy.square() + gz.square()
        density = w_local + (self.epsilon ** 2) * grad_sq

        dV = self.dx * self.dy * self.dz
        return density.sum(dim=(-3, -2, -1)) * dV

    def chemical_potential(self, phi):
        """
        mu = dW/dphi - 2 epsilon^2 Laplacian(phi).

        Il fattore 2 è coerente con una densità energetica
        epsilon^2 |grad phi|^2, senza il prefattore 1/2.
        """
        return self.dW_dphi(phi) - 2.0 * (self.epsilon ** 2) * self.laplacian(phi)

    def mse_loss(self, pred, target):
        return F.mse_loss(pred, target)

    def gradient_loss(self, pred, target):
        gx_pred, gy_pred, gz_pred = self.gradient(pred)
        gx_true, gy_true, gz_true = self.gradient(target)

        return (
            F.l1_loss(gx_pred, gx_true)
            + F.l1_loss(gy_pred, gy_true)
            + F.l1_loss(gz_pred, gz_true)
        )

    def mass_conservation_loss(self, pred, target):
        """
        Errore nella massa totale predetta rispetto al target.

        Non è direttamente il drift temporale della sola predizione:
        confronta M_pred(t) con M_target(t), per ogni B, T, C.
        """
        dV = self.dx * self.dy * self.dz

        mass_pred = pred.sum(dim=(-3, -2, -1)) * dV
        mass_true = target.sum(dim=(-3, -2, -1)) * dV

        return F.mse_loss(mass_pred, mass_true)

    def free_energy_loss(self, pred):
        """
        Penalizza esclusivamente gli incrementi temporali dell'energia:
        max(E(t + dt) - E(t), 0)^2.
        """
        energy = self.free_energy(pred)
        delta_energy = energy[:, 1:] - energy[:, :-1]
        return torch.mean(torch.relu(delta_energy).square())

    def pde_residual_loss(self, pred):
        """
        Residuo della PDE:
            dphi/dt - div(M(phi) grad(mu)) = 0.

        Usa una differenza temporale in avanti tra frame consecutivi.
        """
        if pred.shape[1] < 2:
            raise ValueError(
                "pde_residual_loss requires at least T=2 time frames, "
                f"but received T={pred.shape[1]}."
            )

        dphi_dt = (pred[:, 1:] - pred[:, :-1]) / self.dt

        phi_t = pred[:, :-1]
        mu_t = self.chemical_potential(phi_t)

        dmu_dx, dmu_dy, dmu_dz = self.gradient(mu_t)
        mobility = self.M(phi_t)

        jx = mobility * dmu_dx
        jy = mobility * dmu_dy
        jz = mobility * dmu_dz

        rhs = self.divergence(jx, jy, jz)
        residual = dphi_dt - rhs

        return torch.mean(residual.square())

    def forward(self, pred, target):
        """
        Returns:
            total: scalare differenziabile su cui chiamare backward().
            metrics: dizionario di scalari detached, per logging.
        """
        if pred.shape != target.shape:
            raise ValueError(
                "pred and target must have identical shapes; received "
                f"{tuple(pred.shape)} and {tuple(target.shape)}."
            )

        l_mse = self.mse_loss(pred, target)
        l_energy = self.free_energy_loss(pred)
        l_grad = self.gradient_loss(pred, target)
        l_pde = self.pde_residual_loss(pred)
        l_mass = self.mass_conservation_loss(pred, target)

        total = (
            self.w_mse * l_mse
            + self.w_energy * l_energy
            + self.w_grad * l_grad
            + self.w_pde * l_pde
            + self.w_mass * l_mass
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
