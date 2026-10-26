import numpy as np
import enum
import torch

from config_loader import TargetConfig, FeatureConfig

class LatticeSystem(enum.Enum):
    triclinic = 0
    monoclinic = 1
    orthorhombic = 2
    tetragonal = 3
    rhombohedral = 4
    hexagonal = 5
    cubic = 6


class GixdUtility:
    config = TargetConfig()
    feature_config = FeatureConfig()

    @staticmethod
    def calc_laue_index():
        max_laue_index = GixdUtility.feature_config.max_laue_index
        help_list1 = [(0, 0, i) for i in range(0, max_laue_index+1)]

        help_list2 = [(0, i, j) for i in range(1, max_laue_index+1)
                      for j in range(-max_laue_index, max_laue_index+1)]
        help_list3 = [(h, k, l) for h in range(1, max_laue_index+1)
                      for k in range(-max_laue_index, max_laue_index+1)
                      for l in range(-max_laue_index, max_laue_index+1)]
        full_list = help_list1 + help_list2 + help_list3
        return full_list

    @staticmethod
    def _two_theta_range_from_q_limits():
        """
        Compute (2θ_min, 2θ_max) from feature_config.{q_xy_max, q_z_max} and the
        wavelength used in XRDCalculator.

        Returns a tuple (0.0, 2θ_max_degrees) suitable for get_pattern(... two_theta_range=...).
        """
        target_config = GixdUtility.config
        feature_config = GixdUtility.feature_config
        lam = float(getattr(target_config, "xray_wavelength", 1.5406))  # Å; fallback Cu Kα
        qxy_max = float(feature_config.max_q_xy)  # Å^{-1}
        qz_max  = float(feature_config.max_q_z)   # Å^{-1}

        q_max = float(np.hypot(qxy_max, qz_max))       # √(qxy^2 + qz^2)
        # q = (4π/λ) sin θ  ->  θ = arcsin(q λ / 4π)
        arg = q_max * lam / (4.0 * np.pi)
        # numerical safety: clip into [0,1]
        arg = float(np.clip(arg, 0.0, 1.0))
        two_theta_max = 2.0 * np.degrees(np.arcsin(arg))

        # keep within (0, 180]; add a tiny floor so the range is not empty
        two_theta_max = float(np.clip(two_theta_max, 0.1, 179.9))
        return (0.0, two_theta_max)

    @staticmethod
    def normalize_cell(cell):
        """
        Normalize cell parameters (lengths in Å, angles in radians)
        using config values.
        If the input is a torch.Tensor, the result is on the same device and dtype.
        """
        cfg = GixdUtility.config
        if isinstance(cell, torch.Tensor):
            min_val = torch.tensor([cfg.min_distance]*3 + [np.deg2rad(cfg.min_angle)]*3,
                                   dtype=cell.dtype, device=cell.device)
            max_val = torch.tensor([cfg.max_distance]*3 + [np.deg2rad(cfg.max_angle)]*3,
                                   dtype=cell.dtype, device=cell.device)
            return (cell - min_val) / (max_val - min_val)
        else:
            min_val = np.array([cfg.min_distance]*3 + [np.deg2rad(cfg.min_angle)]*3)
            max_val = np.array([cfg.max_distance]*3 + [np.deg2rad(cfg.max_angle)]*3)
            return (cell - min_val) / (max_val - min_val)

    @staticmethod
    def normalize_cell_deg(cell):
        """
        Normalize cell parameters (lengths in Å, angles in degrees)
        using config values.
        """
        cfg = GixdUtility.config
        if isinstance(cell, torch.Tensor):
            min_val = torch.tensor([cfg.min_distance]*3 + [cfg.min_angle]*3,
                                   dtype=cell.dtype, device=cell.device)
            max_val = torch.tensor([cfg.max_distance]*3 + [cfg.max_angle]*3,
                                   dtype=cell.dtype, device=cell.device)
            return (cell - min_val) / (max_val - min_val)
        else:
            min_val = np.array([cfg.min_distance]*3 + [cfg.min_angle]*3)
            max_val = np.array([cfg.max_distance]*3 + [cfg.max_angle]*3)
            return (cell - min_val) / (max_val - min_val)

    @staticmethod
    def denormalize_cell(norm_cell):
        """
        Denormalize cell parameters (lengths in Å, angles in radians)
        using config values.
        """
        cfg = GixdUtility.config
        if isinstance(norm_cell, torch.Tensor):
            min_val = torch.tensor([cfg.min_distance]*3 + [np.deg2rad(cfg.min_angle)]*3,
                                   dtype=norm_cell.dtype, device=norm_cell.device)
            max_val = torch.tensor([cfg.max_distance]*3 + [np.deg2rad(cfg.max_angle)]*3,
                                   dtype=norm_cell.dtype, device=norm_cell.device)
            return min_val + norm_cell * (max_val - min_val)
        else:
            min_val = np.array([cfg.min_distance]*3 + [np.deg2rad(cfg.min_angle)]*3)
            max_val = np.array([cfg.max_distance]*3 + [np.deg2rad(cfg.max_angle)]*3)
            return min_val + norm_cell * (max_val - min_val)

    @staticmethod
    def denormalize_cell_torch(norm_cell):
        """
        Denormalize cell parameters for torch tensors (lengths in Å, angles in radians)
        without changing dtype or device.
        """
        cfg = GixdUtility.config
        if not isinstance(norm_cell, torch.Tensor):
            raise TypeError("Expected a torch.Tensor")
        min_val = torch.tensor([cfg.min_distance]*3 + [np.deg2rad(cfg.min_angle)]*3,
                               dtype=norm_cell.dtype, device=norm_cell.device)
        max_val = torch.tensor([cfg.max_distance]*3 + [np.deg2rad(cfg.max_angle)]*3,
                               dtype=norm_cell.dtype, device=norm_cell.device)
        return min_val + norm_cell * (max_val - min_val)

    @staticmethod
    def denormalize_cell_deg(norm_cell):
        """
        Denormalize cell parameters (lengths in Å, angles in degrees)
        using config values.
        """
        cfg = GixdUtility.config
        if isinstance(norm_cell, torch.Tensor):
            min_val = torch.tensor([cfg.min_distance]*3 + [cfg.min_angle]*3,
                                   dtype=norm_cell.dtype, device=norm_cell.device)
            max_val = torch.tensor([cfg.max_distance]*3 + [cfg.max_angle]*3,
                                   dtype=norm_cell.dtype, device=norm_cell.device)
            return min_val + norm_cell * (max_val - min_val)
        else:
            min_val = np.array([cfg.min_distance]*3 + [cfg.min_angle]*3)
            max_val = np.array([cfg.max_distance]*3 + [cfg.max_angle]*3)
            return min_val + norm_cell * (max_val - min_val)

    @staticmethod
    def reciprocal_lattice_to_unit_cell_deg(k1, k2, k3):
        """
        Convert a reciprocal lattice (k1, k2, k3) to unit cell parameters 
        (a, b, c, α, β, γ) in degrees.
        
        Works for both numpy arrays and torch tensors and handles both
        the single-sample case (1D vectors) and batched case (2D arrays/tensors 
        with shape (B, 3)).
        """
        # If torch tensors:
        if isinstance(k1, torch.Tensor):
            # Single sample: assume shape (3,)
            if k1.dim() == 1:
                signed_rec_volume = torch.abs(torch.dot(torch.cross(k1, k2), k3))
                v1 = torch.cross(k2, k3) / signed_rec_volume
                v2 = torch.cross(k1, k3) / signed_rec_volume
                v3 = torch.cross(k1, k2) / signed_rec_volume
                a = torch.norm(v1)
                b = torch.norm(v2)
                c = torch.norm(v3)
                alpha = torch.acos(torch.dot(v2, v3) / (b * c))
                beta = torch.acos(torch.dot(v1, v3) / (a * c))
                gamma = torch.acos(torch.dot(v1, v2) / (a * b))
                return a, b, c, torch.rad2deg(alpha), torch.rad2deg(beta), torch.rad2deg(gamma)
            # Batched case: assume shape (B,3)
            elif k1.dim() == 2:
                # Compute signed reciprocal volume per sample.
                cross12 = torch.cross(k1, k2, dim=1)  # (B,3)
                signed_rec_volume = torch.abs((cross12 * k3).sum(dim=1))  # (B,)
                signed_rec_volume = signed_rec_volume.unsqueeze(1)  # (B,1) for division
                v1 = torch.cross(k2, k3, dim=1) / signed_rec_volume  # (B,3)
                v2 = torch.cross(k1, k3, dim=1) / signed_rec_volume  # (B,3)
                v3 = torch.cross(k1, k2, dim=1) / signed_rec_volume  # (B,3)
                a = torch.norm(v1, dim=1)
                b = torch.norm(v2, dim=1)
                c = torch.norm(v3, dim=1)
                alpha = torch.acos((v2 * v3).sum(dim=1) / (b * c))
                beta = torch.acos((v1 * v3).sum(dim=1) / (a * c))
                gamma = torch.acos((v1 * v2).sum(dim=1) / (a * b))
                return a, b, c, torch.rad2deg(alpha), torch.rad2deg(beta), torch.rad2deg(gamma)
            else:
                raise ValueError("Unsupported tensor dimension for k1; expected 1 or 2.")
        else:
            # Assume numpy arrays.
            k1 = np.asarray(k1)
            k2 = np.asarray(k2)
            k3 = np.asarray(k3)
            if k1.ndim == 1:
                signed_rec_volume = np.abs(np.dot(np.cross(k1, k2), k3))
                v1 = np.cross(k2, k3) / signed_rec_volume
                v2 = np.cross(k1, k3) / signed_rec_volume
                v3 = np.cross(k1, k2) / signed_rec_volume
                a = np.linalg.norm(v1)
                b = np.linalg.norm(v2)
                c = np.linalg.norm(v3)
                alpha = np.arccos(np.dot(v2, v3) / (b * c))
                beta = np.arccos(np.dot(v1, v3) / (a * c))
                gamma = np.arccos(np.dot(v1, v2) / (a * b))
                return a, b, c, np.rad2deg(alpha), np.rad2deg(beta), np.rad2deg(gamma)
            elif k1.ndim == 2:
                signed_rec_volume = np.abs(np.sum(np.cross(k1, k2) * k3, axis=1))
                signed_rec_volume = signed_rec_volume[:, np.newaxis]
                v1 = np.cross(k2, k3) / signed_rec_volume
                v2 = np.cross(k1, k3) / signed_rec_volume
                v3 = np.cross(k1, k2) / signed_rec_volume
                a = np.linalg.norm(v1, axis=1)
                b = np.linalg.norm(v2, axis=1)
                c = np.linalg.norm(v3, axis=1)
                alpha = np.arccos(np.sum(v2 * v3, axis=1) / (b * c))
                beta = np.arccos(np.sum(v1 * v3, axis=1) / (a * c))
                gamma = np.arccos(np.sum(v1 * v2, axis=1) / (a * b))
                return a, b, c, np.rad2deg(alpha), np.rad2deg(beta), np.rad2deg(gamma)
            else:
                raise ValueError("Unsupported numpy array dimension for k1; expected 1 or 2.")
