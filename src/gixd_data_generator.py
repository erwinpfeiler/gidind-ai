import numpy as np

from gixd_utility import GixdUtility, LatticeSystem
from config_loader import FeatureConfig, MachineLearningHyperparameter, PathConfig, TargetConfig
import pandas as pd
from matplotlib import pyplot as plt
import pickle
from timeit import default_timer as timer

from tqdm.auto import trange
from scipy.spatial import KDTree
import configparser

import os

import torch
from torch import Tensor
import torch.nn as nn
import time
import torch.nn.functional as F
import torch.optim as optim
#from torchsummary import summary
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
#from captum.attr import IntegratedGradients
#torch.autograd.detect_anomaly(True)

def create_data_generators(top_layer_mode="uniform",
                           target_mode="cell",
                           num_phases=1, use_gmm=False,
                           split_orientations=False,
                           use_filtered_data_stream: bool = False,
                           calc_intensities: bool = False,
                           unit_cell_generator_path: str | None = None):
    machine_learning_config = MachineLearningHyperparameter()
    seed = machine_learning_config.training_seed
    training_data_generators = dict()
    validation_data_generators = dict()
    test_data_generators = dict()


    training_data_generators["uniform"] = GixdDataGenerator("uniform", "training",
                                                            target_mode=target_mode,
                                                            num_phases=num_phases,
                                                            use_gmm=use_gmm,
                                                            split_orientations=split_orientations,
                                                            calc_intensities=calc_intensities,
                                                            unit_cell_generator_path=unit_cell_generator_path,
                                                            rng_seed=seed)
    validation_data_generators["uniform"] = GixdDataGenerator("uniform", "validation",
                                                            target_mode=target_mode,
                                                            num_phases=num_phases,
                                                            use_gmm=use_gmm,
                                                            split_orientations=split_orientations,
                                                            calc_intensities=calc_intensities,
                                                            unit_cell_generator_path=unit_cell_generator_path,
                                                            rng_seed=seed + 1)
    test_data_generators["uniform"] = GixdDataGenerator("uniform", "testing",
                                                            target_mode=target_mode,
                                                            num_phases=num_phases,
                                                            use_gmm=use_gmm,
                                                            split_orientations=split_orientations,
                                                            calc_intensities=calc_intensities,
                                                            unit_cell_generator_path=unit_cell_generator_path,
                                                            rng_seed=seed + 2)
    
    if use_filtered_data_stream:
        from gixd_data_stream import FilteredBatchStream
        training_stream = FilteredBatchStream(
            training_data_generators['uniform'],
            batch_size=machine_learning_config.training_batch_size,
            samples_per_epoch=machine_learning_config.training_epoch_size,
            chunk_size=machine_learning_config.simulation_batch_size,
            min_peaks=5,
            max_peaks=500)
        training_data_generators['stream'] = training_stream

        validation_stream = FilteredBatchStream(
            validation_data_generators['uniform'],
            batch_size=machine_learning_config.validation_batch_size,
            samples_per_epoch=machine_learning_config.validation_epoch_size,
            chunk_size=machine_learning_config.simulation_batch_size,
            min_peaks=5,
            max_peaks=500)
        validation_data_generators['stream'] = validation_stream

        testing_stream = FilteredBatchStream(
            test_data_generators['uniform'],
            batch_size=machine_learning_config.testing_batch_size,
            samples_per_epoch=machine_learning_config.testing_epoch_size,
            chunk_size=machine_learning_config.simulation_batch_size,
            min_peaks=5,
            max_peaks=500)
        test_data_generators['stream'] = testing_stream
        
    return training_data_generators, validation_data_generators, test_data_generators

class GixdDataGenerator(Dataset):
    def __init__(self, top_layer_index, data_mode:str,
                 target_mode="cell", num_phases=1, use_gmm=False, split_orientations=False,
                 calc_intensities=False, load_unit_cell_generator=True, rng_seed=None,
                 unit_cell_generator_path: str | None = None, **kwargs):
        """top_layer_idx: can be 3-tuple of integers to only get samples with that orientation
        or can be 'uniform' to get uniform distributed orientations"""
        super().__init__(**kwargs)
        self.feature_config = FeatureConfig()
        self.target_config = TargetConfig()
        self.ml_hyperparameter = MachineLearningHyperparameter()
        self.path_config = PathConfig()
        self.unit_cell_generator_path = unit_cell_generator_path
        self.rng = np.random.default_rng(rng_seed)
        self.target_mode = target_mode
        self.num_phases=num_phases
        self.calc_intensities = calc_intensities
        #assert self.target_mode in ["cell", "lattice_system", "top_layer"]
        if data_mode == "test": data_mode = "testing"
        assert data_mode in ["training", "validation", "testing"]
        self.data_mode = data_mode
        if self.data_mode == "testing":
            self.epoch_size = self.ml_hyperparameter.testing_epoch_size
            self.batch_size = self.ml_hyperparameter.testing_batch_size
        elif self.data_mode == "training":
            self.epoch_size = self.ml_hyperparameter.training_epoch_size
            self.batch_size = self.ml_hyperparameter.training_batch_size
        elif self.data_mode == "validation":
            self.epoch_size = self.ml_hyperparameter.validation_epoch_size
            self.batch_size = self.ml_hyperparameter.validation_batch_size
        self.max_intensity = self.feature_config.max_intensity
        self.q_xy_max = self.feature_config.max_q_xy
        self.q_z_max = self.feature_config.max_q_z
        self.image_size = self.feature_config.image_size
        self.num_channels = self.feature_config.number_channels
        self.use_specular_scan = self.feature_config.use_specular_scan
        self.sort_lattices_unique = False
        self.use_gmm = use_gmm
        self.top_layer_index = "uniform"

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[DataGenerator] ({self.data_mode}): Using device {self.device}")
        
        # Real-case inference only needs the scattering geometry below.  Avoid
        # loading or generating the synthetic-data sampler in that workflow.
        self.unit_cell_generator = (
            self.load_unit_cell_generator() if load_unit_cell_generator else None
        )
        self.laue_indices = GixdUtility.calc_laue_index()
        self.laue_indices_tensor = torch.tensor(
            self.laue_indices,
            dtype=torch.float64,
            device=self.device
        )

    def __len__(self):
        return int(np.ceil(self.epoch_size / self.batch_size))

    def _resolve_batch_size(self, index: int) -> int:
        remaining = self.epoch_size - index * self.batch_size
        if remaining <= 0:
            raise IndexError(index)
        return min(self.batch_size, remaining)
    
    def __getitem__(self, index=0):
        effective_batch_size = self._resolve_batch_size(index)
        lattice_system_mode_1 = "uniform"  # LatticeSystem.cubic
        lattice_system_mode_2 = "uniform"
        if self.num_phases == 1:
            cell_targets, images, lattice_system_targets, q_xy, q_z, top_layer_indices = \
                self.generate_data_batch(effective_batch_size, "uniform")
            combined_images = images

        elif self.num_phases == 2:
            cell_targets, images_1, lattice_system_targets, q_xy_1, q_z_1, top_layer_indices = self.generate_data_batch(
                np.int32(effective_batch_size*1.5) if self.sort_lattices_unique else effective_batch_size, lattice_system_mode_1)
            cell_targets_2, images_2, lattice_system_targets_2, q_xy_1, q_z_2, top_layer_indices_2 = self.generate_data_batch(
                np.int32(effective_batch_size*1.5) if self.sort_lattices_unique else effective_batch_size, lattice_system_mode_2, fixed_miller_indices=None)
            combined_images = images_1 + images_2
        return (combined_images, torch.Tensor(top_layer_indices).to(torch.float32)),\
            GixdUtility.normalize_cell(cell_targets), lattice_system_targets,\
            q_xy.to(torch.float64), q_z.to(torch.float64)

                    

    def load_unit_cell_generator(self):
        crysal_generator_papth = (
            self.unit_cell_generator_path or self.path_config.unit_cell_generator_path
        )
        if os.path.exists(crysal_generator_papth):
            print("Found learned GMM models")
            # First try normal pickle load
            try:
                with open(crysal_generator_papth, "rb") as fh:
                    unit_cell_generator = pickle.load(fh)
            except RuntimeError as e:
                # Never delete a frozen sampler or silently regenerate its split on a device error
                if "CUDA device" in str(e) or "cuda" in str(e).lower():
                    raise RuntimeError(
                        f"Cannot load sampler {crysal_generator_papth} on this device. "
                        "The original file is preserved; use a CPU-portable sampler or explicitly "
                        "prepare a new dataset in a separate directory."
                    ) from e
                else:
                    raise
        else:
            print("No learned GMM found, so learning them now...")
            from unit_cell_generator import UnitCellGenerator

            unit_cell_generator = UnitCellGenerator(lattice_types=LatticeSystem._member_names_,
                                                    calc_intensities=False, # set True, if not used will be ignored later
                                                    return_niggli=True, seed=42,
                                                    optimize_gmm=False)
            os.makedirs(os.path.dirname(crysal_generator_papth) or ".", exist_ok=True)
            with open(crysal_generator_papth, "wb") as fh:
                pickle.dump(unit_cell_generator, fh)
            print("Saved learned GMM models")

        if False:
            for lt in LatticeSystem._member_names_:
                for data_mode in ["training", "validation", "testing"]:
                    unit_cell_generator.plot_corner_unit_cells(
                        10000, lt, data_mode,
                        save_path=f"{self.path_config.test_data_path}/{data_mode}_{lt}_corner.png")
                    unit_cell_generator.export_unit_cell_samples(100, lt, data_mode=data_mode,
                        file_path=f"{self.path_config.test_data_path}/{data_mode}_{lt}_samples.csv")
                    
        unit_cell_generator.load_needed_models()
        unit_cell_generator.change_device(use_gpu=True if self.calc_intensities else False)
        unit_cell_generator.calc_intensities = self.calc_intensities
        return unit_cell_generator
        
    @staticmethod
    def filter_q_points(q_xy, q_z, tolerance=0.01):
        """
        Removes duplicate points in q-space that are closer than 'tolerance'.
        This version is implemented entirely in PyTorch (without changing device or dtype).
        
        Parameters:
            q_xy: torch.Tensor of shape (n,)
            q_z: torch.Tensor of shape (n,)
            tolerance: float, minimum distance below which points are considered duplicates.
            
        Returns:
            unique_q_xy, unique_q_z: torch.Tensors containing the filtered points,
                                    sorted by their distance from the origin.
        """
        # Stack q_xy and q_z to create an (n, 2) tensor.
        q_points = torch.stack([q_xy, q_z], dim=1)
        n = q_points.size(0)
        unique_indices = []
        
        # Greedy selection of unique points.
        for i in range(n):
            current_point = q_points[i]
            if not unique_indices:
                unique_indices.append(i)
            else:
                accepted_points = q_points[unique_indices]
                dists = torch.norm(accepted_points - current_point, dim=1)
                if torch.all(dists >= tolerance):
                    unique_indices.append(i)
        
        # Get the unique points.
        unique_points = q_points[unique_indices]
        # Compute distances from the origin.
        distances = torch.norm(unique_points, dim=1)
        # Sort the unique points by their distance.
        sorted_indices = torch.argsort(distances)
        unique_points_sorted = unique_points[sorted_indices]
        
        # Split back into q_xy and q_z components.
        unique_q_xy = unique_points_sorted[:, 0]
        unique_q_z = unique_points_sorted[:, 1]
        return unique_q_xy, unique_q_z


    def generate_random_top_layers(self, batch_size):
        used_top_layers = self.feature_config.miller_indices
        indices_of_top_layer = self.rng.choice(len(used_top_layers),batch_size)
        selected_top_layers = [used_top_layers[idx] for idx in indices_of_top_layer]
        return indices_of_top_layer, selected_top_layers
    
    def generate_random_top_layers_2(self, batch_size, fixed_miller_indices):
        used_top_layers = self.feature_config.miller_indices
        indices_of_top_layer = self.rng.choice(len(used_top_layers),batch_size)
        
        # fixed_miller_indices vergleichen mit indices_of_top_layers
        temporary_indices_of_top_layer = list()
        for idx1, item1 in enumerate(indices_of_top_layer):
            while True:
                if item1 != fixed_miller_indices[idx1]:
                    temporary_indices_of_top_layer.append(item1)
                    break
                else:
                    item1 = self.rng.choice(len(used_top_layers),1)
        selected_top_layers = [used_top_layers[idx] for idx in temporary_indices_of_top_layer]

        return temporary_indices_of_top_layer, selected_top_layers
    
    def get_miller_indices_for_top_layer_inidices(self, indices_of_top_layer):
        return [self.feature_config.miller_indices[idx] for idx in indices_of_top_layer]
    
    def generate_image_based_on_celltargets(self,ct, batch_size, lattice_system_mode="uniform", testing_number_missing_peaks=0,
                            testing_number_extra_peaks=0, fixed_miller_indices=None, different_miller_indices = False):
        raise NotImplementedError(f"Does not have intensities implemented, and will be deprecated soon")
        ao, bo, co, alphao, betao, gammao, lattice_system_list = self.create_cell_targets(batch_size,
            lattice_system_mode=lattice_system_mode,
            data_mode=self.data_mode)
        a, b, c, alpha, beta, gamma = ct[0], ct[1], ct[2], ct[3], ct[4], ct[5]
        
        #actual_batch_size = sum(batch_dict.values()) if self.data_mode != "training" else batch_size
        lattice_system_targets = torch.Tensor(lattice_system_list)
        v1, v2, v3, = self.calc_real_unit_cell(
            torch.as_tensor(a, dtype=torch.float64, device=self.device),
            torch.as_tensor(b, dtype=torch.float64, device=self.device),
            torch.as_tensor(c, dtype=torch.float64, device=self.device),
            torch.as_tensor(alpha, dtype=torch.float64, device=self.device),
            torch.as_tensor(beta, dtype=torch.float64, device=self.device),
            torch.as_tensor(gamma, dtype=torch.float64, device=self.device))
        top_layer_indices = None
        if self.top_layer_index=="uniform" and fixed_miller_indices is None:
            top_layer_indices, top_layers = self.generate_random_top_layers(batch_size)
        elif fixed_miller_indices is not None and different_miller_indices==False:
            top_layers = self.get_miller_indices_for_top_layer_inidices(fixed_miller_indices)
            top_layer_indices = fixed_miller_indices
        elif different_miller_indices:
            top_layer_indices, top_layers = self.generate_random_top_layers_2(batch_size, fixed_miller_indices)
        else:
            top_layers = batch_size*[self.top_layer_index]
        top_layers = torch.as_tensor(top_layers, dtype=torch.int64, device=self.device)
        rotation_matrices = self.calc_rotation_matrix(v1, v2, v3, top_layers)
        v1_rot, v2_rot, v3_rot = self.rotate_cell_by_matrix(
            v1, v2, v3, rotation_matrices)
        
        reciprocal_mtx = torch.stack(self.calc_reciprocal_cell(v1_rot, v2_rot, v3_rot), dim=-1)
        #qs = _get_scattering_vector(reciprocal_mtx, laue_indices.to(reciprocal_mtx))
        q_xy, q_z = self.calc_q_points(reciprocal_mtx, testing_number_missing_peaks,
                                       testing_number_extra_peaks)
        images = self.calc_images_from_q_points(q_xy, q_z)
        return images, lattice_system_targets, q_xy, q_z, top_layer_indices
    
    def generate_data_batch(self, batch_size, lattice_system_mode="uniform", 
                            top_layer_index="uniform", testing_number_missing_peaks=0,
                            testing_number_extra_peaks=0):
        a, b, c, alpha, beta, gamma, lattice_system_list, intensity_masks = self.create_cell_targets(batch_size,
            lattice_system_mode=lattice_system_mode,
            data_mode=self.data_mode)
        cell_targets = self.calc_target_tensor(a, b, c, alpha, beta, gamma)
        
        lattice_system_targets = torch.Tensor(lattice_system_list)
        v1, v2, v3, = self.calc_real_unit_cell(
            torch.as_tensor(a, dtype=torch.float64, device=self.device),
            torch.as_tensor(b, dtype=torch.float64, device=self.device),
            torch.as_tensor(c, dtype=torch.float64, device=self.device),
            torch.as_tensor(alpha, dtype=torch.float64, device=self.device),
            torch.as_tensor(beta, dtype=torch.float64, device=self.device),
            torch.as_tensor(gamma, dtype=torch.float64, device=self.device))
        top_layer_indices = None
        if top_layer_index=="uniform":
            top_layer_indices, top_layers = self.generate_random_top_layers(batch_size)
        else:
            top_layers = batch_size*[top_layer_index]
        top_layers = torch.as_tensor(top_layers, dtype=torch.int64, device=self.device)
        rotation_matrices = self.calc_rotation_matrix(v1, v2, v3, top_layers)
        v1_rot, v2_rot, v3_rot = self.rotate_cell_by_matrix(
            v1, v2, v3, rotation_matrices)
        
        reciprocal_mtx = torch.stack(self.calc_reciprocal_cell(v1_rot, v2_rot, v3_rot), dim=-1)
        #qs = _get_scattering_vector(reciprocal_mtx, laue_indices.to(reciprocal_mtx))
        q_xy_full, q_z_full = self.calc_q_points(reciprocal_mtx, testing_number_missing_peaks,
                                       testing_number_extra_peaks)
        
        if intensity_masks is not None and self.calc_intensities:
            if not torch.is_tensor(intensity_masks):
                intensity_masks = torch.tensor(intensity_masks, dtype=torch.bool, device=self.device)
            else:
                intensity_masks = intensity_masks.to(self.device).to(torch.bool)

            # ensure shape match
            if intensity_masks.shape != q_xy_full.shape:
                raise ValueError(f"intensity_masks shape {intensity_masks.shape} != q shape {q_xy_full.shape}")

            q_xy = q_xy_full * intensity_masks
            q_z  = q_z_full  * intensity_masks
        else:
            q_xy = q_xy_full
            q_z = q_z_full
        
        images = self.calc_images_from_q_points(q_xy, q_z)
        return cell_targets, images, lattice_system_targets, q_xy, q_z, top_layer_indices

    def reconstruct_images(self, targets, MI_indices):
        cell_targets = GixdUtility.denormalize_cell_torch(targets.cpu())
        a, b, c = cell_targets[:,0], cell_targets[:,1], cell_targets[:,2]
        alpha, beta, gamma = cell_targets[:,3], cell_targets[:,4], cell_targets[:,5]
        
        v1, v2, v3, = self.calc_real_unit_cell(
            torch.as_tensor(a, dtype=torch.float64),
            torch.as_tensor(b, dtype=torch.float64),
            torch.as_tensor(c, dtype=torch.float64),
            torch.as_tensor(alpha, dtype=torch.float64),
            torch.as_tensor(beta, dtype=torch.float64),
            torch.as_tensor(gamma, dtype=torch.float64))
    
        top_layers = self.get_miller_indices_for_top_layer_inidices(MI_indices)
        top_layer_indices = MI_indices

        top_layers = torch.as_tensor(top_layers, dtype=torch.int64)
        rotation_matrices = self.calc_rotation_matrix(v1, v2, v3, top_layers)
        v1_rot, v2_rot, v3_rot = self.rotate_cell_by_matrix(
            v1, v2, v3, rotation_matrices)
        
        reciprocal_mtx = torch.stack(self.calc_reciprocal_cell(v1_rot, v2_rot, v3_rot), dim=-1)
        #qs = _get_scattering_vector(reciprocal_mtx, laue_indices.to(reciprocal_mtx))
        q_xy, q_z = self.calc_q_points(reciprocal_mtx, 0,
                                       0)

        images = self.calc_images_from_q_points(q_xy, q_z)
        return images


    def create_cell_targets(self, batch_size, lattice_system_mode, data_mode):
        bravais_distribution = self.get_bravais_distribution(lattice_system_mode)
        unit_cells_list, batch_size_list, lattice_system_list = [], {}, []
        intensity_mask_list = []

        for lattice_system, proportion in bravais_distribution.items():
            if proportion > 0.0:
                lattice_system_batch_size = int(np.ceil(batch_size * proportion))
                new_unit_cells, new_masks = self.sample_crystals(lattice_system, lattice_system_batch_size, data_mode)
                unit_cells_list.append(new_unit_cells)
                intensity_mask_list.append(new_masks.detach().cpu().numpy())
                #lattice_system_list += new_unit_cells.shape[0]*[LatticeSystem._member_map_[lattice_system].value]
                batch_size_list[lattice_system] = new_unit_cells.shape[0]
                lattice_system_list.append(LatticeSystem[lattice_system].value*np.ones(
                    new_unit_cells.shape[0], np.int16
                ))
            else:
                batch_size_list[lattice_system] = 0
        lattice_system_arr = np.array(lattice_system_list).flatten()
        unit_cells = np.vstack(unit_cells_list, dtype=np.float32)
        intensity_masks = np.vstack(intensity_mask_list, dtype=bool)
       
        indices = np.arange(unit_cells.shape[0])
        shuffled_indices = self.rng.permutation(indices)
        unit_cells = unit_cells[shuffled_indices,:]
        intensity_masks = intensity_masks[shuffled_indices,:]
        intensity_masks = intensity_masks[:batch_size]
        lattice_system_arr = lattice_system_arr[shuffled_indices]
        
        lattice_system_arr = lattice_system_arr[:batch_size]
        a, b, c, alpha, beta, gamma = self.extract_cell_parameters(unit_cells, batch_size)
        return a, b, c, alpha, beta, gamma, lattice_system_arr, intensity_masks

    def get_bravais_distribution(self, lattice_system_mode):
        if lattice_system_mode == "uniform":
            return {system: 1/len(LatticeSystem) for system in LatticeSystem._member_names_}
        elif lattice_system_mode == "uniform_but_cubic":
            # Set the probability of all systems except cubic to 1/6
            distribution = {system: 0.0 for system in LatticeSystem._member_names_}
            num_non_cubic = len(LatticeSystem) - 1  # Exclude cubic
            for system in LatticeSystem:
                if system != LatticeSystem.cubic:
                    distribution[system.name] = 1 / num_non_cubic  # Uniform distribution for non-cubic systems
            return distribution
        else:
            distribution =  {system: 0.0 for system in LatticeSystem._member_names_}
            distribution.update({lattice_system_mode: 1.0})
            return distribution
        
    def sample_crystals(self, lattice_system, batch_size, data_mode):
        if data_mode == "training":
            return self.unit_cell_generator.get_training_unit_cells(batch_size, lattice_system)
        elif data_mode == "validation":
            return self.unit_cell_generator.get_validation_unit_cells(batch_size, lattice_system)
        elif data_mode == "testing":
            return self.unit_cell_generator.get_testing_unit_cells(batch_size, lattice_system)
        else:
            raise ValueError("Invalid data_mode")

    def extract_cell_parameters(self, unit_cells, batch_size):
        a, b, c = unit_cells[:batch_size, 0], unit_cells[:batch_size, 1], unit_cells[:batch_size, 2]
        alpha, beta, gamma = np.deg2rad(unit_cells[:batch_size, 3]), np.deg2rad(unit_cells[:batch_size, 4]), np.deg2rad(unit_cells[:batch_size, 5])
        return a, b, c, alpha, beta, gamma

    @staticmethod
    def calc_target_tensor(a, b, c, alpha, beta, gamma):
        return np.stack([a, b, c, alpha, beta, gamma], axis=1)
    
    @staticmethod
    def calc_real_unit_cell(
        a: Tensor, 
        b: Tensor,
        c: Tensor, 
        alpha: Tensor, 
        beta: Tensor,
        gamma: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Calculate the real-space unit cell vectors from the lattice parameters.
        Converted to pytorch by Vladimir Starostin
        Parameters
        ----------
        a : Tensor
            The lattice parameter a of shape (batch_size,).
        b : Tensor
            The lattice parameter b of shape (batch_size,).
        c : Tensor
            The lattice parameter c of shape (batch_size,).
        alpha : Tensor
            The lattice angle alpha in radians of shape (batch_size,).
        beta : Tensor
            The lattice angle beta in radians of shape (batch_size,).
        gamma : Tensor
            The lattice angle gamma in radians of shape (batch_size,).
        
        Returns
        -------
        v1 : Tensor
            The first unit cell vector of shape (batch_size, 3).
        v2 : Tensor
            The second unit cell vector of shape (batch_size, 3).
        v3 : Tensor
            The third unit cell vector of shape (batch_size, 3).
        """
        v1 = torch.stack([a, torch.zeros_like(a), torch.zeros_like(a)]).t()
        v2 = torch.stack([b * torch.cos(gamma), b * torch.sin(gamma), torch.zeros_like(b)]).t()
        v3 = torch.stack([
            c * torch.cos(beta),
            c * (torch.cos(alpha) - torch.cos(beta) * torch.cos(gamma)) / torch.sin(gamma),
            torch.sqrt(c**2 - (c * torch.cos(beta))**2 - (c * (torch.cos(alpha) - torch.cos(beta) * torch.cos(gamma)) / torch.sin(gamma))**2)
        ]).t()

        return v1, v2, v3
    
    
    @staticmethod
    def calc_real_vectors_from_unit_cell_rad(unit_cell_rad):
        # Use torch.as_tensor to preserve the numpy array's dtype.
        unit_cell_rad_tensor = torch.as_tensor(unit_cell_rad)
        a = unit_cell_rad_tensor[0:1]
        b = unit_cell_rad_tensor[1:2]
        c = unit_cell_rad_tensor[2:3]
        alpha = unit_cell_rad_tensor[3:4]
        beta = unit_cell_rad_tensor[4:5]
        gamma = unit_cell_rad_tensor[5:6]
        return GixdDataGenerator.calc_real_unit_cell(a, b, c, alpha, beta, gamma)
    
    @staticmethod
    def calc_rotation_matrix(v1: Tensor, v2: Tensor, v3: Tensor,
                            top_layer_indices: Tensor) -> Tensor:
        """
        Fully-differentiable rotation that is numerically safe for the
        (0 0 1) and (0 0 2) orientations.
        """
        if top_layer_indices.dim() == 1:
            top_layer_indices = top_layer_indices.unsqueeze(0)

        bs      = v1.shape[0]
        device  = v1.device
        dtype   = v1.dtype
        eps     = 1e-16

        # ---------- identify the "identity" rows --------------------------------
        id_rows = torch.all(
            (top_layer_indices == v1.new_tensor([0, 0, 1])) |
            (top_layer_indices == v1.new_tensor([0, 0, 2])),
            dim=1
        )                                               # shape (bs,)

        # ---------- allocate output ---------------------------------------------
        R = torch.eye(3, dtype=dtype, device=device).repeat(bs, 1, 1)

        # ---------- nothing more to do if every row is (0 0 1)/(0 0 2) ----------
        if id_rows.all():
            return R

        # ---------- build the in-plane basis vectors (exactly as before) ---------
        hk_zero   = (top_layer_indices[:,0] == 0) & (top_layer_indices[:,1] == 0)
        hl_zero   = (top_layer_indices[:,0] == 0) & (top_layer_indices[:,2] == 0)
        kl_zero   = (top_layer_indices[:,1] == 0) & (top_layer_indices[:,2] == 0)
        h_zero    = (top_layer_indices[:,0] == 0) & (top_layer_indices[:,1] != 0) & (top_layer_indices[:,2] != 0)
        k_zero    = (top_layer_indices[:,1] == 0) & (top_layer_indices[:,0] != 0) & (top_layer_indices[:,2] != 0)
        l_zero    = (top_layer_indices[:,2] == 0) & (top_layer_indices[:,0] != 0) & (top_layer_indices[:,1] != 0)
        nothing_0 = (top_layer_indices != 0).all(dim=1)

        zeros = torch.zeros_like(v1)
        h = top_layer_indices[:,0].to(dtype=dtype).unsqueeze(1)
        k = top_layer_indices[:,1].to(dtype=dtype).unsqueeze(1)
        l = top_layer_indices[:,2].to(dtype=dtype).unsqueeze(1)

        x1 = (torch.where(hk_zero.unsqueeze(1), v1, zeros) +
            torch.where(hl_zero.unsqueeze(1), v1, zeros) +
            torch.where(kl_zero.unsqueeze(1), v2, zeros) +
            torch.where(h_zero.unsqueeze(1), v1, zeros) +
            torch.where(k_zero.unsqueeze(1), v2, zeros) +
            torch.where(l_zero.unsqueeze(1), v3, zeros) +
            torch.where(nothing_0.unsqueeze(1), v1/(h+eps) - v2/(k+eps), zeros))

        x2 = (torch.where(hk_zero.unsqueeze(1), v2, zeros) +
            torch.where(hl_zero.unsqueeze(1), v3, zeros) +
            torch.where(kl_zero.unsqueeze(1), v3, zeros) +
            torch.where(h_zero.unsqueeze(1), v2/(k+eps) - v3/(l+eps), zeros) +
            torch.where(k_zero.unsqueeze(1), v1/(h+eps) - v3/(l+eps), zeros) +
            torch.where(l_zero.unsqueeze(1), v1/(h+eps) - v2/(k+eps), zeros) +
            torch.where(nothing_0.unsqueeze(1), v1/(h+eps) - v3/(l+eps), zeros))

        # ---------- compute the rotation only for the non-identity rows ----------
        keep = ~id_rows
        if keep.any():
            n_new = torch.cross(x1[keep], x2[keep], dim=1)
            n_new = n_new / n_new.norm(dim=1, keepdim=True).clamp_min(eps)

            n_old = v1.new_tensor([0.0, 0.0, 1.0]).expand_as(n_new)
            u_raw = torch.cross(n_old, n_new, dim=1)
            sin_theta_mag = u_raw.norm(dim=1)
            u = u_raw / sin_theta_mag.unsqueeze(1).clamp_min(eps)  # safe, never 0/0

            # Rotate the top-layer normal back onto +z without the coarse acos clamp.
            theta = -torch.atan2(sin_theta_mag, n_new[:, 2])

            cos_t = torch.cos(theta)
            sin_t = torch.sin(theta)

            # Rodrigues formula vectorised
            ux, uy, uz = u[:,0], u[:,1], u[:,2]
            R_keep = (torch.stack([                       # (9, m)
                        cos_t + ux*ux*(1-cos_t),
                        ux*uy*(1-cos_t) - uz*sin_t,
                        ux*uz*(1-cos_t) + uy*sin_t,
                        uy*ux*(1-cos_t) + uz*sin_t,
                        cos_t + uy*uy*(1-cos_t),
                        uy*uz*(1-cos_t) - ux*sin_t,
                        uz*ux*(1-cos_t) - uy*sin_t,
                        uz*uy*(1-cos_t) + ux*sin_t,
                        cos_t + uz*uz*(1-cos_t)
                    ])                                     # default dim=0
                    .permute(1, 0)                         # (m, 9)
                    .reshape(-1, 3, 3)                     # (m, 3, 3)
            )

            R[keep] = R_keep                                     # write back

        return R

    @staticmethod
    def rotate_cell_by_matrix(v1: Tensor, v2: Tensor, v3: Tensor, R: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """
        Calculate the rotated unit cell vectors by a given rotation matrix.

        Parameters
        ----------
        v1 : Tensor
            The first unit cell vector of shape (batch_size, 3).
        v2 : Tensor
            The second unit cell vector of shape (batch_size, 3).
        v3 : Tensor
            The third unit cell vector of shape (batch_size, 3).
        R : Tensor
            The rotation matrix of shape (batch_size, 3, 3).
        
        Returns
        -------
        v1_rot : Tensor
            The first rotated unit cell vector of shape (batch_size, 3).
        v2_rot : Tensor
            The second rotated unit cell vector of shape (batch_size, 3).
        v3_rot : Tensor
            The third rotated unit cell vector of shape (batch_size, 3).
        """
        v1_rot = torch.matmul(R, v1.unsqueeze(2)).squeeze(2)
        v2_rot = torch.matmul(R, v2.unsqueeze(2)).squeeze(2)
        v3_rot = torch.matmul(R, v3.unsqueeze(2)).squeeze(2)
        return v1_rot, v2_rot, v3_rot

    @staticmethod
    def calc_reciprocal_cell(v1: Tensor, v2: Tensor, v3: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """
        Calculate the reciprocal unit cell vectors from the real-space unit cell vectors.

        Parameters
        ----------
        v1 : Tensor
            The first unit cell vector of shape (batch_size, 3).
        v2 : Tensor
            The second unit cell vector of shape (batch_size, 3).
        v3 : Tensor
            The third unit cell vector of shape (batch_size, 3).

        Returns
        -------
        k1 : Tensor
            The first reciprocal unit cell vector of shape (batch_size, 3).
        k2 : Tensor
            The second reciprocal unit cell vector of shape (batch_size, 3).
        k3 : Tensor
            The third reciprocal unit cell vector of shape (batch_size, 3).
        """
        volume = torch.einsum("ij,ij->i", torch.cross(v1, v2, dim=1), v3)
        k1 = torch.cross(v2, v3, dim=1) / volume.unsqueeze(1)
        k2 = torch.cross(v3, v1, dim=1) / volume.unsqueeze(1)
        k3 = torch.cross(v1, v2, dim=1) / volume.unsqueeze(1)
        return k1, k2, k3

    
    @staticmethod
    def get_scattering_vector(reciprocal_mtx, laue_indices):
        """
        Compute the scattering vectors using PyTorch.
        If reciprocal_mtx is 2D (shape [3,3]), use its transpose;
        if it is 3D (shape [batch,3,3]), use transpose(1,2).
        """
        two_pi = 2.0 * np.pi
        li = laue_indices.to(device=reciprocal_mtx.device, dtype=reciprocal_mtx.dtype)
        scattering_vector = two_pi * torch.matmul(li, reciprocal_mtx.transpose(1, 2))
        return scattering_vector

    
    def calc_q_points(self, reciprocal_mtx,
                      testing_number_missing_peaks=0, testing_number_extra_peaks=0):
        batch_size = reciprocal_mtx.shape[0]
        q_tensor = GixdDataGenerator.get_scattering_vector(reciprocal_mtx, self.laue_indices_tensor)
        q_z_no_noise = torch.abs(q_tensor[:, :, 2])
        q_xy_no_noise = torch.sqrt(torch.square(
            q_tensor[:, :, 0])+torch.square(q_tensor[:, :, 1]))
        
        # Specular-scan peaks have a measured q_z but no measured q_xy.
        # Identify the noiseless axis with a tolerance below the image
        # resolution that also accommodates lower-precision rotation arithmetic.
        is_specular = torch.isclose(
            q_xy_no_noise, torch.zeros_like(q_xy_no_noise), rtol=0.0, atol=1e-4
        )

        #add gaussian noise here and only here
        if self.data_mode == "testing":
            sig = self.feature_config.testing_gaussian_noise_sigma
        else:
            sig = torch.rand(
                batch_size,
                1,
                device=self.device,
                dtype=q_xy_no_noise.dtype
            ) * self.feature_config.training_gaussian_noise_sigma
        q_xy_mean = torch.where(is_specular, 0.0, q_xy_no_noise)
        q_xy_sigma = torch.where(is_specular, 0.0, sig)
        q_xy = torch.normal(mean=q_xy_mean, std=q_xy_sigma)
        q_z = torch.normal(mean=q_z_no_noise, std=sig)

        q_too_large = torch.logical_or(q_z >= self.q_z_max, q_xy >= self.q_xy_max)
        if self.feature_config.use_specular_scan:
            # Noise can make either coordinate negative independently.
            q_too_small = torch.logical_or(q_xy < 0.0, q_z < 0.0)
        else:
            q_too_small = torch.logical_or(q_xy <= self.feature_config.specular_cutoff, q_z < 0.0)
        q_out_of_bounds = torch.logical_or(q_too_large, q_too_small)
        q_zero_template = torch.zeros_like(q_xy)

        q_xy_clip = torch.where(q_out_of_bounds, q_zero_template, q_xy)
        q_z_clip = torch.where(q_out_of_bounds, q_zero_template, q_z)

        if self.data_mode == "testing":
            q_xy_final_list = []
            q_z_final_list = []
            max_length = q_xy_clip.size(1) + testing_number_extra_peaks
            for sidx in range(q_xy_clip.size(0)):
                if testing_number_missing_peaks > 0:
                    q_xy_unique, q_z_unique = self.filter_q_points(q_xy_clip[sidx], q_z_clip[sidx], tolerance=0.01)
                    pepper_mask = np.zeros_like(q_xy_unique, dtype=q_xy_unique.dtype)
                    pepper_mask[:testing_number_missing_peaks] = 1
                    pepper_mask[1:] = self.rng.permuted(pepper_mask[:-1])
                    zero_mask = np.zeros_like(pepper_mask, dtype=q_xy_unique.dtype)
                    q_xy_pepper = np.where(pepper_mask > 0.5, zero_mask, q_xy_unique)
                    q_z_pepper = np.where(pepper_mask > 0.5, zero_mask, q_z_unique)
                else:
                    q_xy_pepper = q_xy_clip[sidx].cpu().numpy()
                    q_z_pepper = q_z_clip[sidx].cpu().numpy()
                if testing_number_extra_peaks > 0:
                    q_xy_extra_peaks = self.rng.uniform(0.0, self.feature_config.max_q_xy, testing_number_extra_peaks)
                    q_z_extra_peaks = self.rng.uniform(0.0, self.feature_config.max_q_z, testing_number_extra_peaks)
                    q_xy_salt_pepper = np.concatenate([q_xy_pepper, q_xy_extra_peaks])
                    q_z_salt_pepper = np.concatenate([q_z_pepper, q_z_extra_peaks])
                else:
                    q_xy_salt_pepper = q_xy_pepper
                    q_z_salt_pepper = q_z_pepper
                tensor_xy = torch.tensor(q_xy_salt_pepper, dtype=q_xy_clip.dtype)
                tensor_z = torch.tensor(q_z_salt_pepper, dtype=q_z_clip.dtype)
                pad_xy = (0, max_length - tensor_xy.size(0))
                pad_z = (0, max_length - tensor_z.size(0))
                q_xy_final_list.append(torch.nn.functional.pad(tensor_xy, pad_xy, mode="constant", value=0))
                q_z_final_list.append(torch.nn.functional.pad(tensor_z, pad_z, mode="constant", value=0))
            q_xy_final = torch.stack(q_xy_final_list, dim=0)
            q_z_final = torch.stack(q_z_final_list, dim=0)
        else:
            q_xy_final = q_xy_clip
            q_z_final = q_z_clip

        return q_xy_final, q_z_final
    
    def calc_q_points_from_cell(self, a, b, c, alpha, beta, gamma, top_layer_index=None):
        if top_layer_index is None:
            top_layer_miller_indices = torch.as_tensor([self.top_layer_index], dtype=torch.int64)
        else:
            top_layer_miller_indices = torch.as_tensor([top_layer_index], dtype=torch.int64)
        
        v1, v2, v3 = self.calc_real_unit_cell(
            torch.as_tensor(a, dtype=torch.float64),
            torch.as_tensor(b, dtype=torch.float64),
            torch.as_tensor(c, dtype=torch.float64),
            torch.as_tensor(alpha, dtype=torch.float64),
            torch.as_tensor(beta, dtype=torch.float64),
            torch.as_tensor(gamma, dtype=torch.float64)
        )
        rotation_matrices = self.calc_rotation_matrix(v1, v2, v3, top_layer_miller_indices)
        v1_rot, v2_rot, v3_rot = self.rotate_cell_by_matrix(v1, v2, v3, rotation_matrices)
        reciprocal_mtx = torch.stack(self.calc_reciprocal_cell(v1_rot, v2_rot, v3_rot), dim=-1)
        #qs = _get_scattering_vector(reciprocal_mtx, laue_indices.to(reciprocal_mtx))
        q_xy, q_z = self.calc_q_points(reciprocal_mtx)
        return q_xy, q_z
    
    
    def calc_images_from_q_points(self, q_xy, q_z):
        assert q_xy.shape == q_z.shape, "Shapes of q_xy and q_z must match"
        # Handle different input shapes
        if len(q_xy.shape) == 2:
            batch_size, num_peaks = q_xy.shape
        else:
            batch_size, num_peaks = 1, q_xy.shape[0]
            q_xy = q_xy.unsqueeze(0)
            q_z = q_z.unsqueeze(0)

        xy_idx = (self.image_size * q_xy / self.q_xy_max).floor().to(torch.int32)
        z_idx = ((self.image_size - 1) * (self.q_z_max - q_z) / self.q_z_max).round().to(torch.int32)

        # Raster intensities do not need the geometry's float64 precision.
        # Keep q coordinates and pixel-index calculations in their input dtype;
        # allocate images directly in the float32 consumed by the network.
        intensity = self.max_intensity * torch.ones(batch_size * self.num_channels * num_peaks, dtype=torch.float32, device=q_xy.device)

        column0 = torch.repeat_interleave(torch.arange(batch_size, dtype=torch.int32, device=q_xy.device), repeats=self.num_channels * num_peaks)
        column1 = torch.repeat_interleave(z_idx.reshape(batch_size * num_peaks), repeats=self.num_channels)
        column2 = torch.repeat_interleave(xy_idx.reshape(batch_size * num_peaks), repeats=self.num_channels)
        total = batch_size * self.num_channels * num_peaks
        col_range = torch.arange(total, dtype=torch.int32, device=q_xy.device)
        column3 = torch.fmod(col_range, self.num_channels)

        # Initialize images tensor and scatter intensity
        images = torch.zeros((batch_size, self.image_size, self.image_size, self.num_channels), dtype=torch.float32, device=q_xy.device)
        images[column0, column1, column2, column3] = intensity
        images = images.clamp(0.0, self.max_intensity)

        if self.data_mode == "training":
            batch_size, image_height, image_width, num_channels = images.shape
            random_tensor_salt = torch.rand(batch_size, image_height, image_width, 1, device=images.device)

            max_salt_rate = self.feature_config.training_extra_peak_rate
            salt_rate = torch.rand(batch_size, 1, 1, 1, device=images.device) * max_salt_rate

            salt_mask = (random_tensor_salt < salt_rate).type_as(images)

            salt_mask = salt_mask.expand(-1, -1, -1, num_channels)

            # Mask after rasterisation, as in the historical training pipeline.
            # All degenerate q-point entries at one rendered image pixel share
            # one dropout decision and therefore form one observable peak.
            random_tensor_pepper = torch.rand(
                batch_size, image_height, image_width, 1, device=images.device
            )
            missing_peak_rate = self.feature_config.training_missing_peak_rate
            if not 0.0 <= missing_peak_rate <= 1.0:
                raise ValueError("features.training.missing_peak_rate must be between 0 and 1")
            pepper_rate = (
                torch.rand(batch_size, 1, 1, 1, device=images.device) * missing_peak_rate
            )
            pepper_mask = (random_tensor_pepper < pepper_rate).type_as(images)
            pepper_mask = pepper_mask.expand(-1, -1, -1, num_channels)
            images = images * (1.0 - pepper_mask)

            # These are binary peak-occupancy images: an extra peak landing
            # on an occupied pixel must not double its intensity.
            images = torch.maximum(images, salt_mask * self.feature_config.max_intensity)

        images[:,-1,0,:] = 0 #per definition direct beam is excluded
        return images
    
    def generate_data_for_inspection(self, data_size):
        #return cell_targets, images, lattice_system_targets, q_xy, q_z
        cell_targets, images, lattice_system_targets, q_xy, q_z, _, images_full = self.generate_data_batch(data_size, "uniform")
        if not os.path.exists(f"{self.path_config.root_path}/{self.data_mode}_generator_plots"):
            os.mkdir(f"{self.path_config.root_path}/{self.data_mode}_generator_plots")
        lattice_system_targets = np.int32(lattice_system_targets.numpy())
        return
        print(f"[Generator]: Storing {data_size} sample images to {self.path_config.root_path}/{self.data_mode}_generator_plots/")
        print(f"[Generator]: Range of images is from {images.cpu().numpy().min()} to {images.cpu().numpy().max()}")
        for idx in range(data_size):
            fig, axs = plt.subplots(1,2, figsize=(12,12))
            axs[0].imshow(np.mean(images.cpu().numpy()[idx],axis=-1)/self.feature_config.max_intensity)

            axs[1].imshow(np.mean(images_full.cpu().numpy()[idx],axis=-1)/self.feature_config.max_intensity)
            #plt.title(f"Lattice System = {LatticeSystem(lattice_system_targets[idx]).name}")
            plt.savefig(f"{self.path_config.root_path}/{self.data_mode}_generator_plots/test_plot_{idx}.png")
            plt.close()



if __name__=="__main__":
    #some code to test the data generator
    #testing_generator = GixdDataGenerator((1,0,2), "testing", target_mode="cell",
    #                                       num_phases=1)
    #testing_generator.generate_data_for_inspection(data_size=42)
    training_generator = GixdDataGenerator("uniform", "training", target_mode=["cell", "top_layer"],
                                           num_phases=1, use_gmm=False, calc_intensities=True
                                           )
    for i in trange(10000):
        training_generator.generate_data_for_inspection(data_size=100)
