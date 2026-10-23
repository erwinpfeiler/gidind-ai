import os
import pandas as pd
import numpy as np


import optuna
from niggli import cell_to_niggli
from gixd_utility import LatticeSystem
from config_loader import TargetConfig
from read_crystal_data_2 import filter_cod_dataframes, remove_duplicates_from_df, checked_niggli_cell, CELL_COLUMNS, REDUCED_COLUMNS
from tqdm import tqdm
import torch

import matplotlib.pyplot as plt
import pandas as pd

from pymatgen.core import Lattice, Structure
from pymatgen.core.periodic_table import Element
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.symmetry.groups import SpaceGroup

from gixd_utility import GixdUtility

class UnitCellGenerator:
    # --- mapping: lattice system -> allowed international (H-M) space group ranges (inclusive) ---
    _SG_RANGES = {
        "triclinic":      (1, 2),
        "monoclinic":     (3, 15),
        "orthorhombic":   (16, 74),
        "tetragonal":     (75, 142),
        "rhombohedral":   (143, 167),  # trigonal in hex setting also covered here
        "hexagonal":      (168, 194),
        "cubic":          (195, 230),
    }

    BEST_NUM_GMM_COMPONENTS =  {
        "triclinic":      150,
        "monoclinic":     140,
        "orthorhombic":   100,
        "tetragonal":     140,
        "rhombohedral":   100,  # trigonal in hex setting also covered here
        "hexagonal":      130,
        "cubic":          10,
    }

    def __init__(self, lattice_types=LatticeSystem._member_names_, return_niggli=True,
                 gmm_for_sampling=True, calc_intensities=False, seed=None,
                 optimize_gmm=True):
        self.target_config = TargetConfig()
        self.lattice_types = lattice_types
        self.return_niggli = return_niggli
        self.gmm_for_sampling = gmm_for_sampling
        self.calc_intensities = calc_intensities
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)
        self.gaussian_mixture_models = {}
        self.optimize_gmm = optimize_gmm

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[UnitCellGenerator]: Using device {self.device}", flush=True)

        distance_bounds = (self.target_config.min_distance,
                           self.target_config.max_distance)
        angle_bounds = (self.target_config.min_angle,
                        self.target_config.max_angle)
        self.lower_bounds_arr = np.array([distance_bounds[0], distance_bounds[0],
                                            distance_bounds[0], angle_bounds[0],
                                            angle_bounds[0], angle_bounds[0]])
        self.upper_bounds_arr = np.array([distance_bounds[1], distance_bounds[1],
                                            distance_bounds[1], angle_bounds[1],
                                            angle_bounds[1], angle_bounds[1]])
        
        # TODO: move to config file
        self.intensity_rel_threshold = 0.01 # discard reflections with I/Imax < 1%
        self.xray_wavelength = 1.0 # Å

        # Precompute H (2π * hkl) once, both int and float forms for intensity calculation
        self.two_theta_bounds = GixdUtility._two_theta_range_from_q_limits()
        self.laue_indices = GixdUtility.calc_laue_index()
        self._H_int  = torch.tensor(self.laue_indices, device=self.device, dtype=torch.int16)  # [N,3]
        self._H_2pi  = (2.0*np.pi) * self._H_int.to(torch.float32)                              # [N,3]

        self._centering_cache: dict[str, torch.Tensor] = {}

        self.load_needed_models()
        if self.gmm_for_sampling: # only load COD and learn GMM if GMM will be used for sampling later
            paths = [f"./{folder}/{system}.npy" for folder in ("training_unit_cells", "test_unit_cells")
                     for system in self.lattice_types]
            if any(os.path.exists(path) for path in paths):
                # Cached paper splits must never be silently regenerated from a live COD export.
                merged_lattice_data = {}
            else:
                merged_lattice_data = filter_cod_dataframes(
                    ["./cod_cif_files/" + str(path_idx) + ".pkl" for path_idx in range(1,10)])
            self.load_data(merged_lattice_data)
            self.learn_crystal_distribution(150)

    def load_needed_models(self):
        if self.gmm_for_sampling: # only load COD and learn GMM if GMM will be used for sampling later
            from sklearn.mixture import GaussianMixture
            from sklearn.model_selection import train_test_split

    def change_device(self, use_gpu:bool = True):
        """"
        Method to change self.device to gpu or cpu after loading the object with pickle
        GPU is strongly recommended if intensities are used
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
        print(f"[UnitCellGenerator]: Switched device to {self.device} (use_gpu={use_gpu})", flush=True)


    def _random_spacegroup_number(self, lattice_system: str) -> int:
        lo, hi = self._SG_RANGES[lattice_system]
        return int(self.rng.integers(lo, hi + 1))
    
    def _sample_species_for_asu(self, n_au: int) -> list[str]:
        """
        Decide element types for the n_au asymmetric-unit sites.
        Strategy:
        - with prob p_all_same: all sites share one element (e.g., all C or all Al)
        - else: choose a motif (oxide, halide, mixed) and fill ASU sites accordingly
        """
        # --- knobs (consider moving to config) ---
        p_all_same      = getattr(self, "p_all_same_species", 0.5)    # 50% all-same
        p_oxide_motif   = getattr(self, "p_oxide_motif", 0.3)          # 30% of mixed become oxide-like
        p_halide_motif  = getattr(self, "p_halide_motif", 0.2)         # 20% halide-like; rest generic mix
        # element pools
        cations   = np.array(["Al","Si","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn",
                            "Ga","Ge","Zr","Nb","Mo","Sn","Pb","Mg","Ca","Sr","Ba",
                            "Na","K","Li","Cs","Rb"])
        halides   = np.array(["F","Cl","Br","I"])
        organics  = np.array(["C","N","O","S","F","Cl","Br","I"])      # omit H to avoid vanishing scattering
        all_pool  = np.unique(np.concatenate([cations, organics]))

        # TODO: for organics use molecules instead of just organic atoms randomly distributed

        rng = self.rng

        # Case 1: all-same
        if rng.random() < p_all_same:
            elem = rng.choice(all_pool)
            return [elem] * n_au

        # Case 2: mixed motifs
        r = rng.random()
        species = []
        if r < p_oxide_motif:
            # Ensure O appears at least once; fill remaining with cations
            species.append("O")
            while len(species) < n_au:
                species.append(rng.choice(cations))
        elif r < p_oxide_motif + p_halide_motif:
            # Ensure at least one halogen; pair with common cations
            species.append(rng.choice(halides))
            while len(species) < n_au:
                species.append(rng.choice(np.array(["Na","K","Ag","Cu","Pb","Zn","Cs"])))
        else:
            # Generic mix from all_pool (could include O, C, etc.)
            # Allow repeats so stoichiometries like [C, C, N] are possible
            species = list(rng.choice(all_pool, size=n_au, replace=True))

        rng.shuffle(species)
        return species

    @staticmethod
    def rebase_structure_to_lattice(structure: Structure, target_lat: Lattice, tol: float = 1e-8) -> Structure:
        """
        Change basis of `structure` from its current lattice A_src to `target_lat` = A_dst.
        Keeps *Cartesian* positions fixed; converts fractional coords: f_dst = A_dst^{-1} A_src f_src.
        Optionally deduplicates coincident sites introduced by centering→primitive changes.
        """
        A_src = np.asarray(structure.lattice.matrix, dtype=float)   # 3x3
        A_dst = np.asarray(target_lat.matrix, dtype=float)          # 3x3
        M = np.linalg.solve(A_dst, A_src)                           # A_dst f_dst = A_src f_src

        f_src = np.asarray([s.frac_coords for s in structure.sites], dtype=float)   # N×3
        f_dst = (f_src @ M.T)                                              # N×3
        f_dst = f_dst - np.floor(f_dst)                                    # wrap to [0,1)

        species = [s.specie for s in structure.sites]

        # optional: deduplicate numerically coincident sites after basis change
        if tol is not None and tol > 0:
            keys = np.round(f_dst / tol).astype(np.int64)
            _, idx = np.unique(keys, axis=0, return_index=True)
            f_dst = f_dst[idx]
            species = [species[i] for i in idx]

        return Structure(target_lat, species, f_dst.tolist())
    
    def _conventional_lattice_for_spacegroup(self, sg_num_or_symbol,
                                         a, b, c, alpha, beta, gamma,
                                         eps_angle: float = 1e-3) -> Lattice:
        sg = (SpaceGroup.from_int_number(sg_num_or_symbol)
            if isinstance(sg_num_or_symbol, int) else SpaceGroup(sg_num_or_symbol))
        system = sg.crystal_system      # 'cubic','tetragonal','hexagonal','trigonal','orthorhombic','monoclinic','triclinic'
        cent   = sg.symbol[0]           # P, I, F, C, R, A, B

        a = float(np.clip(a, self.lower_bounds_arr[0], self.upper_bounds_arr[0]))
        b = float(np.clip(b, self.lower_bounds_arr[1], self.upper_bounds_arr[1]))
        c = float(np.clip(c, self.lower_bounds_arr[2], self.upper_bounds_arr[2]))

        if system == "cubic":
            a0 = float((a + b + c) / 3.0)
            return Lattice.cubic(a0)  # α=β=γ=90°

        if system == "tetragonal":
            a0 = float((a + b) / 2.0)
            return Lattice.tetragonal(a0, float(c))  # α=β=γ=90°

        if system == "hexagonal":
            a0 = float((a + b) / 2.0)
            return Lattice.hexagonal(a0, float(c))   # α=β=90°, γ=120°

        if system == "trigonal":
            # IMPORTANT: use hexagonal axes for BOTH P… and R… trigonal SGs when calling from_spacegroup.
            # (pymatgen's builder expects the conventional hex setting for rhombohedral groups, too.)
            a0 = float((a + b) / 2.0)
            return Lattice.hexagonal(a0, float(c))   # α=β=90°, γ=120°

        if system == "orthorhombic":
            return Lattice.orthorhombic(float(a), float(b), float(c))  # all 90°

        if system == "monoclinic":
            beta0 = float(beta)
            if abs(beta0 - 90.0) < 0.5:
                beta0 = 90.0 + (2.0 if beta0 >= 90.0 else -2.0)
            return Lattice.monoclinic(float(a), float(b), float(c), beta0)  # α=γ=90°, β≠90°

        # triclinic
        return Lattice.from_parameters(float(a), float(b), float(c),
                                    float(alpha), float(beta), float(gamma))

    def _build_random_structure_from_cell(self, unit_cell_deg, lattice_system: str):
        # Target (Niggli or your chosen unique basis)
        aN, bN, cN, aN_deg, bN_deg, cN_deg = map(float, unit_cell_deg)
        lat_target = Lattice.from_parameters(aN, bN, cN, aN_deg, bN_deg, cN_deg)

        # 1) choose SG first
        sg_num = self._random_spacegroup_number(lattice_system)
        sg = SpaceGroup.from_int_number(sg_num)
        centering_letter = sg.symbol[0]  # 'P','I','F','C','R','A','B'

        # 2) build conventional lattice for this SG (robust to P/R trigonal split)
        lat_conv = self._conventional_lattice_for_spacegroup(
            sg.symbol, aN, bN, cN, aN_deg, bN_deg, cN_deg
        )

        # 3) (optional) guard against pathological zero volume
        if abs(np.linalg.det(lat_conv.matrix)) < 1e-6:
            return None

        # 4) expand asymmetric unit under this SG in conventional cell
        n_au = int(self.rng.integers(1, 4))
        coords_au = self.rng.random((n_au, 3)).tolist()
        species   = self._sample_species_for_asu(n_au)
        try:
            struct_conv = Structure.from_spacegroup(sg_num, lat_conv, species, coords_au)
        except ValueError as e:
            # DEBUG aid: print once to confirm we're really passing the right metric
            print(f"[SG={sg.symbol} sys={sg.crystal_system}] conventional lattice bad: "
                f"abc={lat_conv.abc}, angles={lat_conv.angles}"
                f"niggli = {unit_cell_deg}"
                f"lattice sys = {lattice_system}")
            raise

        # 5) rebase to your target (Niggli) so Miller indices line up with your basis
        struct = self.rebase_structure_to_lattice(struct_conv, lat_target)
        return struct, centering_letter


    def _centering_mask_from_symbol(self, symbol: str) -> torch.Tensor:
        """
        Return [N_hkl] bool mask for lattice centering: P, A, B, C, I, F, R (hex setting).
        Uses torch ops on precomputed self._H_int and caches results.
        """
        if not symbol:
            symbol = "P"
        c = symbol[0]  # first letter of H-M symbol
        if c in self._centering_cache:
            return self._centering_cache[c]

        H = self._H_int
        h, k, l = H[:, 0].to(torch.int16), H[:, 1].to(torch.int16), H[:, 2].to(torch.int16)

        if c == "P":
            mask = torch.ones(h.shape, dtype=torch.bool, device=self.device)
        elif c == "I":
            mask = ((h + k + l) & 1) == 0
        elif c == "F":
            # all even or all odd
            eh, ek, el = (h & 1), (k & 1), (l & 1)
            mask = (eh == ek) & (ek == el)
        elif c == "A":
            mask = ((k + l) & 1) == 0
        elif c == "B":
            mask = ((h + l) & 1) == 0
        elif c == "C":
            mask = ((h + k) & 1) == 0
        elif c == "R":
            # Rhombohedral in hex axes: -h + k + l = 3n  (equivalently ( -h + k + l ) % 3 == 0)
            mask = torch.remainder(-h + k + l, 3) == 0
        else:
            mask = torch.ones(h.shape, dtype=torch.bool, device=self.device)

        self._centering_cache[c] = mask
        return mask

    def _fast_mask_from_structure(self, structure, intensity_threshold: float,
                                  centering_letter: str | None = None,
                                  use_parity_prefilter: bool = False) -> torch.Tensor:
        """
        Vectorized |F_hkl|^2 mask. No symmetry analysis.
        If use_parity_prefilter=True, we apply centering rules; only do this when
        structure is in the conventional basis.
        """
        if structure is None or len(structure.sites) == 0:
            print(f"_fast_mask_from_structure has structure is None or no sites: = {structure}")
            return torch.zeros(self._H_int.shape[0], dtype=torch.bool, device=self.device)

        # ---- centering prefilter (optional; off by default for rebased structures) ----
        if use_parity_prefilter and centering_letter:
            center_mask = self._centering_mask_from_symbol(centering_letter)  # uses self._H_int
        else:
            center_mask = torch.ones(self._H_int.shape[0], dtype=torch.bool, device=self.device)

        R_np = np.asarray([s.frac_coords for s in structure.sites], dtype=np.float32)  # [S,3]
        Z_np = np.asarray([s.specie.Z   for s in structure.sites], dtype=np.float32)  # [S]
        R = torch.from_numpy(R_np).to(self.device).T        # [3,S]
        Z = torch.from_numpy(Z_np).to(self.device)          # [S]

        if not center_mask.any():
            return center_mask

        H = self._H_2pi[center_mask]                                       # [N_allowed,3]
        phases = H @ R                                                      # [N_allowed,S]
        # Real-valued sum: Fr = Σ Z cos φ, Fi = Σ Z sin φ
        Fr = (torch.cos(phases) * Z).sum(dim=1)
        Fi = (torch.sin(phases) * Z).sum(dim=1)
        I  = Fr*Fr + Fi*Fi                                                 # [N_allowed]

        # Relative threshold within this structure
        Imax = torch.max(I) if I.numel() else torch.tensor(0.0, device=self.device)
        keep_allowed = I >= (float(intensity_threshold) * Imax.clamp_min(1e-12))

        # Scatter back into full [N_hkl] mask
        mask = torch.zeros(self._H_int.shape[0], dtype=torch.bool, device=self.device)
        mask[center_mask] = keep_allowed
        return mask
    def _build_per_sample_hkl_mask(self, lattice_system_str, unit_cells_deg, intensity_threshold=0.01):
        N = len(self.laue_indices)
        out = torch.zeros((unit_cells_deg.shape[0], N), dtype=torch.bool, device=self.device)

        for i, unit_cell_deg in enumerate(unit_cells_deg):

            struct, centering_letter = self._build_random_structure_from_cell(
                unit_cell_deg, lattice_system_str
            )
            # --- FAST path ---
            out[i] = self._fast_mask_from_structure(struct, intensity_threshold, centering_letter)

            # fallback if fast method fails
            # if not out[i].any() and getattr(self, "use_xrd_fallback", False):
            #     allowed = self._allowed_hkls_from_structure(struct, intensity_threshold,
            #               wavelength=getattr(self.target_config, "xray_wavelength", 1.5406),
            #               two_theta_range=self.two_theta_bounds)
            #     if allowed:
            #         laue_list = [tuple(map(int, hkls)) for hkls in self.laue_indices]
            #         out[i] = torch.tensor([hk in allowed for hk in laue_list],
            #                               dtype=torch.bool, device=self.device)
        return out

    def _allowed_hkls_from_structure(self, structure, intensity_threshold: float = 0.01,
                                 wavelength: float | None = None):
        """
        Return a dict {(h,k,l): I_rel} for reflections with relative intensity >= intensity_threshold.
        - intensity_threshold is relative (0..1) w.r.t. the max intensity in the pattern.
        - wavelength: if None, XRDCalculator default is used (Cu Kα ~1.5406 Å).
        """
        calc = XRDCalculator(wavelength=wavelength) if wavelength is not None else XRDCalculator()
        try:
            patt = calc.get_pattern(structure, two_theta_range=self.two_theta_bounds)
        except:
            print(f"Failed for struct: {structure}")
            raise ValueError

        intens = np.asarray(patt.y, dtype=float) 

        if intens.size == 0:
            return {}

        # Relative intensities
        imax = float(intens.max())
        relI = intens / (imax if imax > 0 else 1.0)

        allowed = {}
        # patt.hkls is a list of families; each family is a list of dicts like {'hkl': (h,k,l), 'multiplicity': m}
        for family, irel in zip(patt.hkls, relI):
            if irel >= intensity_threshold:
                for d in family:
                    h, k, l = map(int, d["hkl"])
                    allowed[(h, k, l)] = float(irel)
                    # include Friedel pair (kinematic approximation, centrosymmetric intensity)
                    allowed[(-h, -k, -l)] = float(irel)
        return allowed


    def load_data(self, merged_data):
        """Keep cached splits intact; fresh partitions use globally deduplicated Niggli keys."""
        import hashlib
        import json
        from pathlib import Path
        from sklearn.model_selection import train_test_split

        folders = [Path("training_unit_cells"), Path("test_unit_cells")]
        paths = [folder / f"{system}.npy" for folder in folders for system in self.lattice_types]
        present = [path.is_file() for path in paths]
        if any(present) and not all(present):
            raise ValueError("Incomplete cached train/test split. Refusing to mix old and freshly split COD data; "
                             "use a separate empty working directory to build a new dataset.")
        for folder in folders:
            folder.mkdir(exist_ok=True)
        self.train_crystals, self.test_crystals = {}, {}
        self.train_niggli_crystals, self.test_niggli_crystals = {}, {}
        seen = set()
        manifest = {"schema_version": 1, "seed": self.seed, "test_fraction": 0.2,
                    "reducer": "niggli.cell_to_niggli", "niggli_key_decimal_places": 3,
                    "deduplication_scope": "global_before_any_split", "systems": {}}
        for system in self.lattice_types:
            train_path, test_path = [folder / f"{system}.npy" for folder in folders]
            if all(present):
                train = np.atleast_2d(np.loadtxt(train_path, dtype=np.float64))
                test = np.atleast_2d(np.loadtxt(test_path, dtype=np.float64))
            else:
                data = remove_duplicates_from_df(merged_data[system], seen_keys=seen)
                conventional = data[CELL_COLUMNS].to_numpy(dtype=np.float64)
                reduced = data[REDUCED_COLUMNS].to_numpy(dtype=np.float64)
                train_indices, test_indices = train_test_split(
                    np.arange(len(data)), test_size=0.2, shuffle=True, random_state=self.seed)
                train, test = conventional[train_indices], conventional[test_indices]
                self.train_niggli_crystals[system] = reduced[train_indices]
                self.test_niggli_crystals[system] = reduced[test_indices]
                # Conventional values are needed for symmetry-specific GMM coordinates.
                # Canonical, unrounded cells are saved alongside the very same partition.
                np.savetxt(train_path, train)
                np.savetxt(test_path, test)
                for folder, indices in zip(folders, (train_indices, test_indices)):
                    np.save(folder / f"{system}_niggli.npy", reduced[indices])
                manifest["systems"][system] = {
                    "training_rows": len(train), "test_rows": len(test),
                    "training_sha256": hashlib.sha256(train_path.read_bytes()).hexdigest(),
                    "test_sha256": hashlib.sha256(test_path.read_bytes()).hexdigest()}
            self.train_crystals[system], self.test_crystals[system] = train, test
        if all(present):
            # Validate cached cross-system identities too. Never repair a historical
            # split implicitly or learn a new sampler from a contaminated cache.
            training_keys = {tuple(np.round(checked_niggli_cell(cell), 3))
                             for cells in self.train_crystals.values() for cell in cells}
            for system, cells in self.test_crystals.items():
                if any(tuple(np.round(checked_niggli_cell(cell), 3)) in training_keys for cell in cells):
                    raise ValueError(f"Cached split has Niggli-equivalent train/test overlap ({system}). "
                                     "Create a fresh split in a separate directory; files are unchanged.")
        if not any(present):
            Path("training_unit_cells/split_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    @staticmethod
    def _gmm_coordinates(cells, lattice_type):
        """Legacy GMM coordinates, retained for compatibility with the original prior."""
        cells = np.asarray(cells, dtype=np.float64)
        if lattice_type == "triclinic":
            return cells.copy()
        if lattice_type == "orthorhombic":
            return cells[:, :3].copy()
        if lattice_type == "cubic":
            return cells[:, :1].copy()
        if lattice_type == "hexagonal":
            return cells[:, [0, 2]].copy()
        if lattice_type == "rhombohedral":
            return cells[:, [0, 3]].copy()
        if lattice_type == "tetragonal":
            # Preserve the original min/max representation, including c < a.
            values = np.zeros((len(cells), 2), dtype=np.float32)
            values[:, 0] = np.amin(cells[:, :3], axis=1)
            values[:, 1] = np.amax(cells[:, :3], axis=1)
            return values
        if lattice_type == "monoclinic":
            # Preserve original length order and oblique-angle selection.
            values = np.zeros((len(cells), 4), dtype=np.float32)
            values[:, :3] = cells[:, :3]
            alpha, beta, gamma = cells[:, 3], cells[:, 4], cells[:, 5]
            best_angle = np.where(np.abs(alpha - 90) > np.abs(beta - 90), alpha, beta)
            values[:, 3] = np.where(np.abs(best_angle - 90) > np.abs(gamma - 90), best_angle, gamma)
            return values
        raise ValueError(f"Unknown lattice system: {lattice_type}")

    def learn_crystal_distribution(self, max_num_components):
        from sklearn.model_selection import train_test_split
        from sklearn.mixture import GaussianMixture
        self.validation_crystals = {}
        self.gmm_training_crystals = {}
        seen_training_keys = set()
        for lattice_type in self.lattice_types:
            # Cached historical training arrays can contain equivalent rows too.
            # Deduplicate before the inner development split as well.
            train_crystals = remove_duplicates_from_df(
                pd.DataFrame(self.train_crystals[lattice_type], columns=CELL_COLUMNS),
                seen_keys=seen_training_keys)[CELL_COLUMNS].to_numpy(dtype=np.float64)

            train_crystals, valid_crystals = train_test_split(train_crystals, test_size=0.2,
                                                            shuffle=True, random_state=self.seed)
            self.validation_crystals[lattice_type] = valid_crystals.copy()
            self.gmm_training_crystals[lattice_type] = train_crystals.copy()
            print(f"Lattice type to optimize now = {lattice_type}")
            def objective(trial: optuna.trial.Trial, return_gmm=False):
                if self.optimize_gmm:
                    num_components = trial.suggest_int("num_components", 1, max_num_components)
                else:
                    num_components = self.BEST_NUM_GMM_COMPONENTS[lattice_type]
                    trial.suggest_int("num_components", num_components, num_components)
                nonlocal train_crystals
                nonlocal valid_crystals
                train_values = self._gmm_coordinates(train_crystals, lattice_type)
                valid_values = self._gmm_coordinates(valid_crystals, lattice_type)
                gmm = GaussianMixture(num_components, random_state=np.random.RandomState(self.seed)).fit(train_values)
                valid_score = gmm.score(valid_values)
                if return_gmm:
                    return valid_score, gmm
                return valid_score
            study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=self.seed))
            study.optimize(objective, n_trials= 7 if self.optimize_gmm else 1)
            best_num_components = study.best_params["num_components"]
            print(f"Best num_components = {best_num_components}")
            self.gaussian_mixture_models[lattice_type] = objective(study.best_trial, True)[1]

    def _project_metric_for_spacegroup(self, sg_symbol: str,
                                   a: float, b: float, c: float,
                                   alpha: float, beta: float, gamma: float,
                                   eps_angle: float = 1e-3):
        """
        Coerce (a,b,c,alpha,beta,gamma) to the conventional metric required by the
        given space group symbol (e.g. 'P321', 'R-3m', 'C2/m', 'Fm-3m', ...).
        Angles in degrees. Returns a,b,c,alpha,beta,gamma.
        """
        # Start within your global box, keep a hair off boundaries to avoid singular cases
        a = float(np.clip(a, self.lower_bounds_arr[0], self.upper_bounds_arr[0]))
        b = float(np.clip(b, self.lower_bounds_arr[1], self.upper_bounds_arr[1]))
        c = float(np.clip(c, self.lower_bounds_arr[2], self.upper_bounds_arr[2]))
        alpha = float(np.clip(alpha, self.lower_bounds_arr[3], self.upper_bounds_arr[3]))
        beta  = float(np.clip(beta,  self.lower_bounds_arr[4], self.upper_bounds_arr[4]))
        gamma = float(np.clip(gamma, self.lower_bounds_arr[5], self.upper_bounds_arr[5]))

        # First-letter centering + crystal system encoded in symbol
        sym = sg_symbol.strip()
        cent = sym[0]  # P, C, I, F, R, A, B

        # Parse “crystal system” at a high level
        # (pymatgen gives it too if you need: SpaceGroup(...).crystal_system)
        is_cubic      = any(x in sym for x in ["m-3m", "m-3", "3m"]) and cent in ("P","I","F") and ("6" not in sym and "4" not in sym)
        is_tetragonal = "4" in sym or "4/m" in sym
        is_hexagonal  = "6" in sym or "6/m" in sym
        is_trigonal   = ("3" in sym) and not is_hexagonal and not is_tetragonal  # P3, R3, P-3, R-3, P312, P321, ...
        is_orthorhombic = any(sym.startswith(s) for s in ["P2_12_12_1","P222","C222","F222","I222","Pnma","Cmcm","Pbca","Pmma"]) or ("mm2" in sym)  # heuristic OK
        is_monoclinic = ("2" in sym or "m" in sym) and not (is_trigonal or is_tetragonal or is_hexagonal or is_orthorhombic or is_cubic)
        is_triclinic  = sym.lower() in ("p1", "p-1")

        if is_cubic:
            x = float((a + b + c) / 3.0)
            a = b = c = x
            alpha = beta = gamma = 90.0

        elif is_tetragonal:
            s = float((a + b) / 2.0)
            a = b = s
            alpha = beta = gamma = 90.0

        elif is_hexagonal:
            s = float((a + b) / 2.0)
            a = b = s
            alpha = beta = 90.0
            gamma = 120.0

        elif is_trigonal:
            if cent == "R":
                # rhombohedral setting
                x = float((a + b + c) / 3.0)
                a = b = c = x
                ang = float((alpha + beta + gamma) / 3.0)
                if abs(ang - 90.0) < 0.5:
                    ang = 90.0 + (2.0 if np.random.random() < 0.5 else -2.0)
                alpha = beta = gamma = np.clip(ang, 60.0 + eps_angle, 120.0 - eps_angle)
            else:
                # trigonal in hexagonal axes (e.g. P321, P3, P-3, P312, ...)
                s = float((a + b) / 2.0)
                a = b = s
                alpha = beta = 90.0
                gamma = 120.0

        elif is_orthorhombic:
            alpha = beta = gamma = 90.0

        elif is_monoclinic:
            # b-unique conventional setting: α=γ=90°, β free
            alpha = 90.0
            gamma = 90.0
            # keep β away from exactly 90°
            if abs(beta - 90.0) < 0.1:
                beta = 90.0 + (1.0 if beta >= 90.0 else -1.0)

        else:
            # triclinic: no extra metric constraints
            pass

        # tiny eps to avoid exact 60°/120° edges
        alpha = float(np.clip(alpha, 60.0 + eps_angle, 120.0 - eps_angle))
        beta  = float(np.clip(beta,  60.0 + eps_angle, 120.0 - eps_angle))
        gamma = float(np.clip(gamma, 60.0 + eps_angle, 120.0 - eps_angle))
        return a, b, c, alpha, beta, gamma

    def _project_to_bravais_metric(self, lattice_system: str,
                               a: float, b: float, c: float,
                               alpha: float, beta: float, gamma: float,
                               eps_angle: float = 1e-3):
        """
        Coerce (a,b,c,alpha,beta,gamma) to the conventional metric for the given lattice system.
        Angles in degrees. Returns a,b,c,alpha,beta,gamma after projection.
        """
        # Clamp inside your global box but keep a hair away from boundaries
        a = float(np.clip(a, self.lower_bounds_arr[0], self.upper_bounds_arr[0]))
        b = float(np.clip(b, self.lower_bounds_arr[1], self.upper_bounds_arr[1]))
        c = float(np.clip(c, self.lower_bounds_arr[2], self.upper_bounds_arr[2]))
        alpha = float(np.clip(alpha, self.lower_bounds_arr[3], self.upper_bounds_arr[3]))
        beta  = float(np.clip(beta,  self.lower_bounds_arr[4], self.upper_bounds_arr[4]))
        gamma = float(np.clip(gamma, self.lower_bounds_arr[5], self.upper_bounds_arr[5]))

        ls = (lattice_system or "").lower()

        if ls == "cubic":
            # a=b=c; α=β=γ=90°
            x = float((a + b + c) / 3.0)
            a = b = c = x
            alpha = beta = gamma = 90.0

        elif ls == "tetragonal":
            # a=b≠c; α=β=γ=90°
            s = float((a + b) / 2.0)
            a = b = s
            alpha = beta = gamma = 90.0

        elif ls == "orthorhombic":
            # α=β=γ=90°
            alpha = beta = gamma = 90.0

        elif ls == "hexagonal":
            # a=b≠c; α=β=90°, γ=120°
            s = float((a + b) / 2.0)
            a = b = s
            alpha = beta = 90.0
            gamma = 120.0

        elif ls in ("trigonal", "rhombohedral"):
            # rhombohedral setting: a=b=c; α=β=γ≠90°
            x = float((a + b + c) / 3.0)
            a = b = c = x
            # choose a common angle away from 90°
            ang = float((alpha + beta + gamma) / 3.0)
            if abs(ang - 90.0) < 0.5:
                ang = 90.0 + (2.0 if np.random.random() < 0.5 else -2.0)
            alpha = beta = gamma = np.clip(ang, 60.0 + eps_angle, 120.0 - eps_angle)

        elif ls == "monoclinic":
            # b-unique conventional setting: α=γ=90°, β free
            angles = np.array([alpha, beta, gamma], dtype=float)
            dev = np.abs(angles - 90.0)
            # pick the angle most deviant from 90° and assign it to β
            beta_new = float(angles[int(np.argmax(dev))])
            # keep β inside bounds and off exactly 90°
            if abs(beta_new - 90.0) < 0.1:
                beta_new = 90.0 + (1.0 if beta_new >= 90.0 else -1.0)
            beta  = np.clip(beta_new, 60.0 + eps_angle, 120.0 - eps_angle)
            alpha = 90.0
            gamma = 90.0

        else:
            # triclinic or unknown: leave as-is (your general bounds still apply)
            pass

        # tiny eps to avoid exact 60°/120° edges
        alpha = float(np.clip(alpha, 60.0 + eps_angle, 120.0 - eps_angle))
        beta  = float(np.clip(beta,  60.0 + eps_angle, 120.0 - eps_angle))
        gamma = float(np.clip(gamma, 60.0 + eps_angle, 120.0 - eps_angle))
        return a, b, c, alpha, beta, gamma

    def sample_unit_cells_from_gmm(self, num_crystals, lattice_type):
        assert len(self.gaussian_mixture_models) == len(self.lattice_types), "First GMMs need to be trained before sampling is possible"
        gmm = self.gaussian_mixture_models[lattice_type]
        # Drive sampling through the Generator stream saved by the training
        # checkpoint hooks; sklearn's private RandomState is not checkpointed.
        previous_random_state = gmm.random_state
        try:
            gmm.random_state = int(self.rng.integers(0, np.iinfo(np.int32).max))
            gmm_sample = self.rng.permutation(gmm.sample(num_crystals)[0], axis=0)
        finally:
            gmm.random_state = previous_random_state
        if lattice_type == "triclinic":
            sample = gmm_sample.copy()
        elif lattice_type == "orthorhombic":
            sample = 90.00 * np.ones((num_crystals, 6), np.float32)
            sample[:, 0:3] = gmm_sample
        elif lattice_type == "cubic":
            sample = 90.00 * np.ones((num_crystals, 6), np.float32)
            
            sample[:, 0:3] = gmm_sample
        elif lattice_type == "tetragonal":
            sample = 90.00 * np.ones((num_crystals, 6), np.float32)
            
            sample[:,0] = gmm_sample[:,0]
            sample[:,1] = gmm_sample[:,0]
            sample[:,2] = gmm_sample[:,1]
        elif lattice_type == "rhombohedral":
            sample = 90.00 * np.ones((num_crystals, 6), np.float32)
            
            sample[:,0] = gmm_sample[:,0]
            sample[:,1] = gmm_sample[:,0]
            sample[:,2] = gmm_sample[:,0]
            sample[:,3] = gmm_sample[:,1]
            sample[:,4] = gmm_sample[:,1]
            sample[:,5] = gmm_sample[:,1]
        elif lattice_type == "hexagonal":
            sample = 90.00 * np.ones((num_crystals, 6), np.float32)
            
            sample[:,0] = gmm_sample[:,0]
            sample[:,1] = gmm_sample[:,0]
            sample[:,2] = gmm_sample[:,1]
            sample[:,5] = 120.0 * np.ones((num_crystals), np.float32)
        elif lattice_type == "monoclinic":
            sample = 90.00 * np.ones((num_crystals, 6), np.float32)
            
            sample[:,0] = gmm_sample[:,0]
            sample[:,1] = gmm_sample[:,1]
            sample[:,2] = gmm_sample[:,2]
            sample[:,5] = gmm_sample[:,3]
        else:
            raise ValueError("Invalid lattice_type = "+str(lattice_type))

        if not self.return_niggli:
            return sample
        return self._convert_to_niggli(sample, num_crystals)

    def sample_unit_cells_uniform(self, num_crystals, lattice_type):
        triclinic_sample = self.rng.uniform(low=0.0, high=1.0, size=(num_crystals, 6)).reshape((num_crystals,6))
        triclinic_sample = triclinic_sample * (self.upper_bounds_arr-self.lower_bounds_arr) + self.lower_bounds_arr
        if lattice_type == "triclinic":
            sample = triclinic_sample
        elif lattice_type == "orthorhombic":
            sample = 90.00 * np.ones((num_crystals, 6), np.float32)
            sample[:, 0:3] = triclinic_sample[:, 0:3]
        elif lattice_type == "cubic":
            sample = 90.00 * np.ones((num_crystals, 6), np.float32)
            sample[:, 0:3] = triclinic_sample[:, 0:1]
        elif lattice_type == "tetragonal":
            sample = 90.00 * np.ones((num_crystals, 6), np.float32)
            sample[:,0] = triclinic_sample[:,0]
            sample[:,1] = triclinic_sample[:,0]
            sample[:,2] = triclinic_sample[:,1]
        elif lattice_type == "rhombohedral":
            sample = 90.00 * np.ones((num_crystals, 6), np.float32)
            sample[:,0] = triclinic_sample[:,0]
            sample[:,1] = triclinic_sample[:,0]
            sample[:,2] = triclinic_sample[:,0]
            sample[:,3] = triclinic_sample[:,1]
            sample[:,4] = triclinic_sample[:,1]
            sample[:,5] = triclinic_sample[:,1]
        elif lattice_type == "hexagonal":
            sample = 90.00 * np.ones((num_crystals, 6), np.float32)
            sample[:,0] = triclinic_sample[:,0]
            sample[:,1] = triclinic_sample[:,0]
            sample[:,2] = triclinic_sample[:,1]
            sample[:,5] = 120.0 * np.ones((num_crystals), np.float32)
        elif lattice_type == "monoclinic":
            sample = 90.00 * np.ones((num_crystals, 6), np.float32)
            sample[:,0] = triclinic_sample[:,0]
            sample[:,1] = triclinic_sample[:,1]
            sample[:,2] = triclinic_sample[:,2]
            sample[:,5] = triclinic_sample[:,3]
        else:
            raise ValueError("Invalid lattice_type = "+str(lattice_type))

        if not self.return_niggli:
            return sample
        return self._convert_to_niggli(sample, num_crystals)

    def _convert_to_niggli(self, sample, num_crystals):
        niggli_sample = np.zeros_like(sample)
        for i in range(num_crystals):
            if np.any(sample[i] < self.lower_bounds_arr) or \
                    np.any(sample[i] > self.upper_bounds_arr):
                niggli_sample[i] = 2*self.upper_bounds_arr
            else:
                try:
                    niggli_sample[i] = cell_to_niggli(sample[i])
                except RuntimeError:
                    print(f"crippled sample: {sample[i]}")
                    niggli_sample[i] = np.array(6*[np.nan])
        return niggli_sample
    
    def sample_unit_cells(self, num_crystals, lattice_type):
        # a wrapper for sample_unit_cells_from_gmm and sample_unit_cells_uniform
        if self.gmm_for_sampling:
            return self.sample_unit_cells_from_gmm(num_crystals, lattice_type)
        else:
            return self.sample_unit_cells_uniform(num_crystals, lattice_type)

    def get_training_unit_cells(self, num_crystals, lattice_type): 
        # for training we take the unit cells from a GMM trained on COD
        # because then we can generate an infinite amount of them
        adaptive_sample_size = np.int32(np.round(num_crystals * 1.3))
        sample = self.sample_unit_cells(adaptive_sample_size,
                                                 lattice_type) #gmm2 is not gmm but mostly uniformly distributed
        while True:
            sample = self._filter_sample(sample)
            if sample.shape[0] > num_crystals:
                break
            adaptive_sample_size *= 2
            sample = self.sample_unit_cells(adaptive_sample_size,
                                                     lattice_type)
        if self.calc_intensities:
            mask = self._build_per_sample_hkl_mask(lattice_type, sample[:num_crystals, :], 0.01)
        else:
            mask = torch.ones((num_crystals, len(self.laue_indices)),dtype=torch.bool)
        return sample[:num_crystals, :], mask

    def _filter_sample(self, sample):
        """
        Filter by (a,b,c,α,β,γ) bounds and by minimum cell height (altitude).
        Angles in 'sample' are in degrees.
        Uses TargetConfig.min_cell_height if available; otherwise defaults to 1.0 Å.
        """
        # 1) Basic box constraints (lengths & angles)
        in_bounds = np.all(sample <= self.upper_bounds_arr, axis=1) & \
                    np.all(sample >= self.lower_bounds_arr, axis=1)
        if not np.any(in_bounds):
            return sample[:0]  # empty with correct shape

        s = sample[in_bounds]

        # 2) Compute cell volume and altitudes (heights) along a, b, c
        a, b, c = s[:, 0], s[:, 1], s[:, 2]
        alpha = np.deg2rad(s[:, 3]); beta  = np.deg2rad(s[:, 4]); gamma = np.deg2rad(s[:, 5])

        ca, cb, cg = np.cos(alpha), np.cos(beta), np.cos(gamma)
        sa, sb, sg = np.sin(alpha), np.sin(beta), np.sin(gamma)

        # Niggli/standard volume formula: V = abc * sqrt(1 + 2cosαcosβcosγ - cos²α - cos²β - cos²γ)
        discr = 1.0 + 2.0*ca*cb*cg - ca**2 - cb**2 - cg**2
        discr = np.maximum(discr, 0.0)              # numerical safety
        V = a * b * c * np.sqrt(discr)

        # Base areas and corresponding altitudes (avoid divide-by-zero)
        eps = 1e-12
        base_a = np.maximum(b * c * sa, eps)
        base_b = np.maximum(a * c * sb, eps)
        base_c = np.maximum(a * b * sg, eps)

        h_a = V / base_a
        h_b = V / base_b
        h_c = V / base_c
        min_height = np.nanmin(np.stack([h_a, h_b, h_c], axis=1), axis=1)

        # 3) Height threshold
        h_min = getattr(self.target_config, "min_cell_height", 1.5)  # default 1.0 Å if not configured
        height_mask = (min_height >= h_min) & np.isfinite(min_height)

        return s[height_mask]
    
    def get_testing_unit_cells(self, num_crystals, lattice_type):
        sample = np.copy(self.test_crystals[lattice_type])
        #sample = self._filter_sample(self._convert_to_niggli(sample, sample.shape[0]))
        sample = self._filter_sample(sample)
        sample = self.rng.choice(sample, num_crystals, replace=True)
        #print(f"Sampling {num_crystals} {lattice_type} crystals: ")
        #for i in range(num_crystals):
        #    print(f"[{i}]: {sample[i]}")
        sample = self._convert_to_niggli(sample, num_crystals)
        if self.calc_intensities:
            mask = self._build_per_sample_hkl_mask(lattice_type, sample, 0.01)
        else:
            mask = torch.ones((num_crystals, len(self.laue_indices)),dtype=torch.bool)
        return sample, mask

    def get_validation_unit_cells(self, num_crystals, lattice_type):
        pools = getattr(self, "validation_crystals", None)
        if pools is None:
            import warnings
            if not getattr(self, "_warned_legacy_validation", False):
                warnings.warn("Legacy sampler lacks a saved independent validation pool. "
                              "Validation uses the non-test COD training partition; independence "
                              "from the fitted GMM cannot be certified. Rebuild for new training.", RuntimeWarning)
                self._warned_legacy_validation = True
            pools = self.train_crystals
        sample = self._filter_sample(np.copy(pools[lattice_type]))
        if len(sample) == 0:
            raise ValueError(f"No eligible validation cells for {lattice_type}")
        # Always return the requested size, including when a crystal system has a small pool.
        sample = self.rng.choice(sample, num_crystals, replace=num_crystals > len(sample))
        sample = self._convert_to_niggli(sample, num_crystals)
        if self.calc_intensities:
            mask = self._build_per_sample_hkl_mask(lattice_type, sample, 0.01)
        else:
            mask = torch.ones((num_crystals, len(self.laue_indices)), dtype=torch.bool)
        return sample, mask


    def plot_corner_unit_cells(self, num, lattice_type, data_mode="testing", save_path=None, **kwargs):
        """
        Generates a corner plot for unit cell data based on the specified data_mode.
        Parameters:
            num (int): number of samples to plot.
            lattice_type (str): the lattice type to select.
            data_mode (str): one of "training", "testing", or "validation".
            save_path (str, optional): file path to save the figure.
            **kwargs: additional keyword arguments to pass to the plotting function.
        Returns:
            The matplotlib figure object.
        """
        import corner

        labels = ["a", "b", "c", r"$\alpha$", r"$\beta$", r"$\gamma$"]

        if data_mode.lower() in ["testing", "test"]:
            data = self.get_testing_unit_cells(num, lattice_type)
        elif data_mode.lower() == "validation":
            data = self.get_validation_unit_cells(num, lattice_type)
        elif data_mode.lower() == "training":
            data = self.get_training_unit_cells(num, lattice_type)
        else:
            raise ValueError("data_mode must be one of 'training', 'testing', or 'validation'")

        fig = corner.corner(
            data,
            labels=labels,
            show_titles=False,
            label_kwargs={"fontsize": 32},
            bins=30,
            plot_contours=False,
            range=list(zip(self.lower_bounds_arr, self.upper_bounds_arr)),
            **kwargs
        )

        # Fix y-axis labels on the left column manually
        ndim = len(labels)
        axes = np.array(fig.axes).reshape((ndim, ndim))
        for i in range(1, ndim):
            ax = axes[i, 0]
            ax.set_ylabel(labels[i], fontsize=32)

        if save_path:
            fig.savefig(save_path)
            plt.close(fig)
        else:
            return fig
        
    def export_unit_cell_samples(self, num, lattice_type, file_path, data_mode="testing"):
        labels = ["a", "b", "c", "alpha", "beta", "gamma"]
        
        
        if data_mode.lower() == "training":
            data = self.get_training_unit_cells(num, lattice_type)
        elif data_mode.lower() == "validation":
            data = self.get_validation_unit_cells(num, lattice_type)
        elif data_mode.lower() == "testing":
            data = self.get_testing_unit_cells(num, lattice_type)
        else:
            raise ValueError("data_mode must be one of 'training', 'validation', or 'testing'")
        
        # Create a DataFrame and add a column for lattice_type
        df = pd.DataFrame(data, columns=labels)
        df["lattice_type"] = lattice_type
        df.to_csv(file_path, index=False)


if __name__ == '__main__':
    lattice_system = "triclinic"
    generator = UnitCellGenerator(lattice_types=[lattice_system], return_niggli=True,
                                  gmm_for_sampling=True, seed=42)
    

"""time analysis on one core CPU for 1024 batch:
uniform sampling, niggli and filtering: 0.51 s
gmm sampling, niggli and filtering    : 0.44 s
gmm + intensities                     : 69.36 s
gmm + intensities fast on laptop GPU  : 2.15 s
"""
