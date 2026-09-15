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

    Gli operatori spaziali ricevono un tensore 5D:
        (B, T, C, Nx, Ny, Nz) -> (B*T, C, Nx, Ny, Nz)

    Energia libera:
        E(phi) = integral [W(phi) + epsilon^2 |grad phi|^2] dV

    Potenziale:
        W(phi) = (18 / epsilon) phi^2 (1 - phi)^2

    Mobilità:
        M(phi) = M0 (36 / epsilon) phi^2 (1 - phi)^2

    Boundary conditions:
        Neumann omogenee in x, y, z, approssimate con padding replicate.

    Termini aggiuntivi:
        - energy_matching_loss: MSE tra E_pred(B,T,C) ed E_true(B,T,C)
        - bounds_loss: mean violation + peak violation per frame, pensata per
          artefatti localizzati fuori dall'intervallo [0, 1].
    """

    def __init__(
        self,
        w_mse=1.0,
        w_energy=0.0,
        w_grad=0.0,
        w_bounds=0.0,
        bounds_peak_weight=0.05,
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
        self.w_bounds = w_bounds

        # Peso interno della penalità sui picchi locali fuori [0, 1].
        self.bounds_peak_weight = bounds_peak_weight

        self.epsilon = epsilon
        self.M0 = M0
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.dt = dt

    def _reshape_spatial_3d(self, c):
        """(B, T, C, Nx, Ny, Nz) -> (B*T, C, Nx, Ny, Nz)."""
        if c.ndim != 6:
            raise ValueError(
                "Expected c to have shape (B, T, C, Nx, Ny, Nz), "
                f"but got {tuple(c.shape)}."
            )

        B, T, C, Nx, Ny, Nz = c.shape
        return c.reshape(B * T, C, Nx, Ny, Nz)

    @staticmethod
    def _restore_spatial_3d(c_reshaped, original_shape):
        """(B*T, C, Nx, Ny, Nz) -> (B, T, C, Nx, Ny, Nz)."""
        return c_reshaped.reshape(original_shape)

    def pad_neumann(self, c):
        """Adds one replicate ghost layer on both sides of x, y, z."""
        c_reshaped = self._reshape_spatial_3d(c)
        return F.pad(c_reshaped, (1, 1, 1, 1, 1, 1), mode="replicate")

    def gradient(self, c):
        """Second-order central gradient with homogeneous Neumann padding."""
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
        """Second-order central divergence with homogeneous Neumann padding."""
        if jx.shape != jy.shape or jx.shape != jz.shape:
            raise ValueError(
                "jx, jy and jz must have identical shapes; got "
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
        """3D seven-point Laplacian with homogeneous Neumann padding."""
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
        """Local free-energy density W(phi)."""
        return (18.0 / self.epsilon) * phi.square() * (1.0 - phi).square()

    def free_energy(self, phi):
        """
        Calculates the free energy for each sample, time and channel.

        Output shape:
            (B, T, C)
        """
        w_local = self.W(phi)
        gx, gy, gz = self.gradient(phi)

        grad_sq = gx.square() + gy.square() + gz.square()
        density = w_local + (self.epsilon ** 2) * grad_sq

        dV = self.dx * self.dy * self.dz
        return density.sum(dim=(-3, -2, -1)) * dV

    def mse_loss(self, pred, target):
        return F.mse_loss(pred, target)

    def energy_matching_loss(self, pred, target):
        """
        MSE between predicted and target free energy at every B, T, C.

        This replaces the previous energy-dissipation-only penalty.
        It forces the predicted trajectory to have the same energy curve
        as the target trajectory, but it does not alone force the spatial
        morphology to match; the voxel MSE and/or gradient loss do that.
        """
        energy_pred = self.free_energy(pred)
        energy_true = self.free_energy(target)
        return F.mse_loss(energy_pred, energy_true)

    def bounds_loss(self, pred):
        """
        Soft range constraint for phi in [0, 1], designed for local artifacts.

        mean_term penalizes the total spatial extent/severity of violations.
        peak_term computes one maximum violation per (sample, time, channel),
        then averages those maxima. Therefore a small local spike cannot be
        fully diluted by the 3D volume.

        The returned quantity is:
            mean(v^2) + bounds_peak_weight * mean(max_xyz(v^2))
        where v is the distance outside [0, 1].
        """
        below = F.relu(-pred)
        above = F.relu(pred - 1.0)

        violation_sq = below.square() + above.square()

        mean_term = violation_sq.mean()
        peak_per_frame = violation_sq.amax(dim=(-3, -2, -1))
        peak_term = peak_per_frame.mean()

        return mean_term + self.bounds_peak_weight * peak_term

    def free_energy_dissipation_loss(self, pred):
        """
        Optional diagnostic/regularizer: penalizes E(t+dt) > E(t).

        It is not used in total by default. Use it only if you later add a
        separate weight and explicitly want energy monotonicity in addition
        to matching the reference energy curve.
        """
        if pred.shape[1] < 2:
            raise ValueError(
                "free_energy_dissipation_loss requires at least T=2 time frames, "
                f"but got T={pred.shape[1]}."
            )

        energy = self.free_energy(pred)
        delta_energy = energy[:, 1:] - energy[:, :-1]
        return F.relu(delta_energy).square().mean()

    def forward(self, pred, target):
        """
        Returns:
            total: differentiable scalar for backward().
            metrics: detached scalar terms for logging.
        """
        if pred.shape != target.shape:
            raise ValueError(
                "pred and target must have identical shapes; got "
                f"{tuple(pred.shape)} and {tuple(target.shape)}."
            )

        l_mse = self.mse_loss(pred, target)
        l_energy = self.energy_matching_loss(pred, target)
        l_bounds = self.bounds_loss(pred)

        total = (
            self.w_mse * l_mse
            + self.w_energy * l_energy
            + self.w_bounds * l_bounds
        )

        metrics = {
            "loss_total": total.detach(),
            "loss_mse": l_mse.detach(),
            "loss_energy": l_energy.detach(),
            "loss_bounds": l_bounds.detach(),
        }

        return total, metrics

