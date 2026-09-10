import torch
import torch.nn as nn
import torch.nn.functional as F


class CahnHilliardLoss(nn.Module):
    """
    Loss per un campo di Cahn-Hilliard 3D.

    Convenzione degli input:
        pred, target: (B, T, C, Nx, Ny, Nz)

    Coordinate fisiche:
        asse -3 -> x
        asse -2 -> y
        asse -1 -> z

    Gli operatori spaziali usano il layout PyTorch 3D:
        (B, T, C, Nx, Ny, Nz) -> (B*T, C, Nx, Ny, Nz)

    Energia:
        E(phi) = integral [W(phi) + epsilon^2 |grad phi|^2] dV

    Potenziale:
        W(phi) = (18 / epsilon) phi^2 (1 - phi)^2

    Mobilità:
        M(phi) = M0 (36 / epsilon) phi^2 (1 - phi)^2

    Boundary conditions:
        Neumann omogenee in x, y, z, approssimate con padding replicate.
    """

    def __init__(
        self,
        w_mse=1.0,
        w_energy=1e-3,
        w_grad=0.0,
        w_pde=0.0,
        w_mass=0.0,
        w_bounds=0.0,
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
        self.w_bounds = w_bounds

        self.epsilon = epsilon
        self.M0 = M0
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.dt = dt

    def _reshape_spatial_3d(self, c):
        """
        Trasforma:
            (B, T, C, Nx, Ny, Nz) -> (B*T, C, Nx, Ny, Nz)

        La dimensione C viene conservata e non fusa nel batch.
        """
        if c.ndim != 6:
            raise ValueError(
                "Expected c to have shape (B, T, C, Nx, Ny, Nz), "
                f"but got {tuple(c.shape)}."
            )

        B, T, C, Nx, Ny, Nz = c.shape
        return c.reshape(B * T, C, Nx, Ny, Nz)

    @staticmethod
    def _restore_spatial_3d(c_reshaped, original_shape):
        """Trasforma (B*T, C, Nx, Ny, Nz) -> (B, T, C, Nx, Ny, Nz)."""
        return c_reshaped.reshape(original_shape)

    def pad_neumann(self, c):
        """
        Aggiunge un ghost layer per lato lungo x, y e z.

        Input:
            c: (B, T, C, Nx, Ny, Nz)

        Output:
            (B*T, C, Nx+2, Ny+2, Nz+2)
        """
        c_reshaped = self._reshape_spatial_3d(c)
        return F.pad(c_reshaped, (1, 1, 1, 1, 1, 1), mode="replicate")

    def gradient(self, c):
        """
        Calcola gradiente 3D con differenze centrali del secondo ordine.

        Output:
            gx, gy, gz con shape uguale a c.
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
        Calcola div(j) = d(jx)/dx + d(jy)/dy + d(jz)/dz
        con differenze centrali del secondo ordine e BC di Neumann.

        Input/output:
            jx, jy, jz e div hanno shape (B, T, C, Nx, Ny, Nz).
        """
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
        """
        Calcola Laplaciano 3D con stencil a 7 punti e BC di Neumann.

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
        """Derivata del potenziale: dW/dphi."""
        return (
            (36.0 / self.epsilon)
            * phi
            * (1.0 - phi)
            * (1.0 - 2.0 * phi)
        )

    def M(self, phi):
        """Mobilità degenere M(phi)."""
        return (
            self.M0
            * (36.0 / self.epsilon)
            * phi.square()
            * (1.0 - phi).square()
        )

    def free_energy(self, phi):
        """
        Calcola E(phi) per ogni batch, frame temporale e canale.

        Output:
            (B, T, C)
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

        Il fattore 2 è coerente con:
            E = integral [W(phi) + epsilon^2 |grad phi|^2] dV
        """
        return self.dW_dphi(phi) - 2.0 * (self.epsilon ** 2) * self.laplacian(phi)

    def mse_loss(self, pred, target):
        return F.mse_loss(pred, target)

    def gradient_loss(self, pred, target):
        """Errore L1 tra i gradienti predetti e target."""
        gx_pred, gy_pred, gz_pred = self.gradient(pred)
        gx_true, gy_true, gz_true = self.gradient(target)

        return (
            F.l1_loss(gx_pred, gx_true)
            + F.l1_loss(gy_pred, gy_true)
            + F.l1_loss(gz_pred, gz_true)
        )

    def mass_conservation_loss(self, pred, target):
        """
        Confronta la massa totale di predizione e target per ogni B, T, C.
        """
        dV = self.dx * self.dy * self.dz

        mass_pred = pred.sum(dim=(-3, -2, -1)) * dV
        mass_true = target.sum(dim=(-3, -2, -1)) * dV

        return F.mse_loss(mass_pred, mass_true)

    def bounds_loss(self, pred):
        """
        Penalizza solamente i valori esterni al range fisico [0, 1].

        Per ogni voxel:
            phi < 0: penalità phi^2
            0 <= phi <= 1: penalità 0
            phi > 1: penalità (phi - 1)^2
        """
        below_zero = F.relu(-pred)
        above_one = F.relu(pred - 1.0)

        return torch.mean(below_zero.square() + above_one.square())

    def free_energy_loss(self, pred):
        """
        Penalizza aumenti di energia libera tra frame consecutivi:
            mean(max(E(t+dt) - E(t), 0)^2).
        """
        if pred.shape[1] < 2:
            raise ValueError(
                "free_energy_loss requires at least T=2 time frames, "
                f"but got T={pred.shape[1]}."
            )

        energy = self.free_energy(pred)
        delta_energy = energy[:, 1:] - energy[:, :-1]
        return torch.mean(F.relu(delta_energy).square())

    def pde_residual_loss(self, pred):
        """
        Residuo della PDE di Cahn-Hilliard:
            dphi/dt - div(M(phi) grad(mu)) = 0.

        La derivata temporale è una differenza in avanti tra frame consecutivi.
        """
        if pred.shape[1] < 2:
            raise ValueError(
                "pde_residual_loss requires at least T=2 time frames, "
                f"but got T={pred.shape[1]}."
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
            total: loss scalare differenziabile su cui chiamare backward().
            metrics: dizionario detached per logging.
        """
        if pred.shape != target.shape:
            raise ValueError(
                "pred and target must have identical shapes; got "
                f"{tuple(pred.shape)} and {tuple(target.shape)}."
            )

        l_mse = self.mse_loss(pred, target)
        l_energy = self.free_energy_loss(pred)
        l_grad = self.gradient_loss(pred, target)
        l_pde = self.pde_residual_loss(pred)
        l_mass = self.mass_conservation_loss(pred, target)
        l_bounds = self.bounds_loss(pred)

        total = (
            self.w_mse * l_mse
            + self.w_energy * l_energy
            + self.w_grad * l_grad
            + self.w_pde * l_pde
            + self.w_mass * l_mass
            + self.w_bounds * l_bounds
        )

        metrics = {
            "loss_total": total.detach(),
            "loss_mse": l_mse.detach(),
            "loss_energy": l_energy.detach(),
            "loss_grad": l_grad.detach(),
            "loss_pde": l_pde.detach(),
            "loss_mass": l_mass.detach(),
            "loss_bounds": l_bounds.detach(),
        }

        return total, metrics
