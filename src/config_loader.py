import os
from importlib.resources import files
from pathlib import Path

import json5

DEFAULT_CONFIG_PATH = files("gidind_ai").joinpath("config.json5")
"""The package's reviewer-facing inference configuration."""

CONFIG_OVERRIDE_ENV = "GIDIND_AI_CONFIG"
"""Optional absolute path to a standalone training configuration."""


def get_config_path() -> Path:
    """Return the active configuration file.

    Training jobs use ``GIDIND_AI_CONFIG`` to select one immutable experiment
    configuration without editing the package default used for inference.
    """
    configured_path = os.environ.get(CONFIG_OVERRIDE_ENV)
    if configured_path:
        path = Path(configured_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(
                f"{CONFIG_OVERRIDE_ENV} points to a missing configuration file: {path}"
            )
        return path
    return Path(DEFAULT_CONFIG_PATH)


# Kept for older callers that import CONFIG_PATH directly. New callers that
# support a runtime override should call get_config_path() when they read/copy it.
CONFIG_PATH = get_config_path()



class Config:
    def __init__(self):
        self.config_path = get_config_path()
        with self.config_path.open("r") as fh:
            self._config = json5.load(fh)

    def reload_config_file(self):
        del self._config
        self.__init__()

    
        
    def get_property(self, property_names:list):
        config = self._config
        for property_name in property_names:
            if property_name not in config:
                raise ValueError("Invalid property name: "+str(property_names))
            config = config[property_name]
        return config

    def has_property(self, property_names:list):
        config = self._config
        for property_name in property_names:
            if property_name not in config:
                return False
            config = config[property_name]
        return True
    
    def set_property(self, property_names:list, value):
        config = self._config
        last_key = property_names[-1]
        for property_name in property_names:
            if property_name not in config:
                raise ValueError("Invalid property name: "+str(property_names))
            if property_name != last_key:
                config = config[property_name]
            else:
                config[property_name] = value

class TestConfig(Config):
    
    @property
    def value1(self):
        return self.get_property(["test", "test_value1"])
    

class FeatureConfig(Config):
    
    @property
    def image_size(self):
        return self.get_property(["features", "image_size"])
    

    @property
    def miller_indices(self):
        using_reduced_miller_indices = self.get_property(["features",
                                                          "use_reduced_miller_indices"])
        if using_reduced_miller_indices:
            miller_indices_list = self.get_property(["features", "reduced_miller_indices"])
            return [tuple(hkl) for hkl in miller_indices_list]
        else:
            miller_indices_list = self.get_property(["features", "miller_indices"])
            return [tuple(hkl) for hkl in miller_indices_list]
        
    @property
    def number_channels(self):
        return self.get_property(["features", "number_channels"])
    
    @property
    def min_intensity(self):
        return self.get_property(["features", "min_intensity"])
    
    @property
    def max_intensity(self):
        return self.get_property(["features", "max_intensity"])
    
    @property
    def max_laue_index(self):
        return self.get_property(["features", "max_laue_index"])
    
    @property
    def max_q_xy(self):
        return self.get_property(["features", "max_q_xy"])
    
    @property
    def max_q_z(self):
        return self.get_property(["features", "max_q_z"])
    
    @property
    def use_specular_scan(self):
        return self.get_property(["features", "use_specular_scan"])
    
    @property
    def specular_cutoff(self):
        return self.get_property(["features", "specular_cutoff"])
    
    @property
    def training_extra_peak_rate(self):
        return self.get_property(["features", "training", "extra_peak_rate"])
    
    @property
    def training_missing_peak_rate(self):
        """Maximum per-image probability for masking each rendered peak pixel."""
        return float(self.get_property(["features", "training", "missing_peak_rate"]))
    
    @property
    def testing_number_extra_peaks(self):
        return self.get_property(["features", "testing", "number_extra_peaks"])

    @testing_number_extra_peaks.setter
    def testing_number_extra_peaks(self, new_value):
        self.set_property(["features", "testing", "number_extra_peaks"], new_value)
    
    @property
    def testing_number_missing_peaks(self):
        return self.get_property(["features", "testing", "number_missing_peaks"])

    @testing_number_missing_peaks.setter
    def testing_number_missing_peaks(self, new_value):
        self.set_property(["features", "testing", "number_missing_peaks"], new_value)
    
    @property
    def training_gaussian_noise_sigma(self):
        return self.get_property(["features", "training", "gaussian_noise_sigma"])
    
    @property
    def testing_gaussian_noise_sigma(self):
        return self.get_property(["features", "testing", "gaussian_noise_sigma"])
    
    @testing_gaussian_noise_sigma.setter
    def testing_gaussian_noise_sigma(self, new_value):
        self.set_property(["features", "testing", "gaussian_noise_sigma"], new_value)
    

class TargetConfig(Config):
    
    @property
    def min_angle(self):
        return self.get_property(["targets", "min_angle"])
    
    @property
    def max_angle(self):
        return self.get_property(["targets", "max_angle"])
    
    @property
    def min_distance(self):
        return self.get_property(["targets", "min_distance"])
    
    @property
    def max_distance(self):
        return self.get_property(["targets", "max_distance"])
    
    @property
    def min_cell_height(self):
        return self.get_property(["targets", "min_cell_height"])


class MachineLearningHyperparameter(Config):

    @property
    def training_seed(self):
        """The single seed for Python, NumPy, PyTorch, and data streams."""
        if self.has_property(["ml_hyperparameters", "seed"]):
            return int(self.get_property(["ml_hyperparameters", "seed"]))
        return 42

    @property
    def deterministic_algorithms(self):
        if self.has_property(["ml_hyperparameters", "deterministic_algorithms"]):
            return bool(
                self.get_property(["ml_hyperparameters", "deterministic_algorithms"])
            )
        return False

    @property
    def checkpoint_every_n_epochs(self):
        if self.has_property(["ml_hyperparameters", "checkpoint_every_n_epochs"]):
            value = int(
                self.get_property(["ml_hyperparameters", "checkpoint_every_n_epochs"])
            )
            if value < 1:
                raise ValueError("ml_hyperparameters.checkpoint_every_n_epochs must be positive")
            return value
        return 1

    @property
    def num_epochs(self):
        return self.get_property(["ml_hyperparameters", "num_epochs"])
    
    @property
    def initial_lr(self):
        return self.get_property(["ml_hyperparameters", "initial_lr"])

    @property
    def simulation_batch_size(self):
        if self.has_property(["ml_hyperparameters", "simulation", "batch_size"]):
            return self.get_property(["ml_hyperparameters", "simulation", "batch_size"])
        return max(self.training_batch_size, self.validation_batch_size)
    
    @property
    def training_batch_size(self):
        return self.get_property(["ml_hyperparameters", "training", "batch_size"])
    
    @property
    def training_epoch_size(self):
        if self.has_property(["ml_hyperparameters", "training", "samples_per_epoch"]):
            return self.get_property(["ml_hyperparameters", "training", "samples_per_epoch"])
        return self.get_property(["ml_hyperparameters", "training", "epoch_size"])
    
    @property
    def validation_batch_size(self):
        return self.get_property(["ml_hyperparameters", "validation", "batch_size"])
    
    @property
    def validation_epoch_size(self):
        if self.has_property(["ml_hyperparameters", "validation", "samples_per_epoch"]):
            return self.get_property(["ml_hyperparameters", "validation", "samples_per_epoch"])
        return self.get_property(["ml_hyperparameters", "validation", "epoch_size"])
    
    @property
    def testing_batch_size(self):
        return self.get_property(["ml_hyperparameters", "testing", "batch_size"])

    @property
    def testing_epoch_size(self):
        if self.has_property(["ml_hyperparameters", "testing", "samples_per_epoch"]):
            return self.get_property(["ml_hyperparameters", "testing", "samples_per_epoch"])
        return self.testing_batch_size
    

class RefinementHyperparameters(Config):

    @property
    def matching_q_cutoff_1(self):
        return self.get_property(["refinement_hyperparameters",
                                  "matching_q_cutoff_1"])
    
    @property
    def matching_q_cutoff_2(self):
        return self.get_property(["refinement_hyperparameters",
                                  "matching_q_cutoff_2"])
    
    @property
    def mismatch_tolerance_rate(self):
        return self.get_property(["refinement_hyperparameters",
                                  "mismatch_tolerance_rate"])
    
    @property
    def maximal_refinement_loss(self):
        return self.get_property(["refinement_hyperparameters",
                                  "maximal_refinement_loss"])


class PathConfig(Config):

    def __init__(self):
        super().__init__()
        self._root_path = self.get_property(["paths",
                                            "root_path"])
        
    @property
    def root_path(self):
        return self._root_path
    
    @root_path.setter
    def root_path(self, root_path):
        self._root_path = root_path
    
    @property
    def model_path(self):
        return os.path.join(self._root_path,
                            self.get_property(["paths", "model_path"]))
    
    @property
    def training_history_path(self):
        return os.path.join(self._root_path,
                            self.get_property(["paths", "training_history_path"]))
    
    @property
    def test_data_path(self):
        return os.path.join(self._root_path,
                            self.get_property(["paths", "test_data_path"]))
    
    @property
    def hkl_symmetry_path(self):
        return os.path.join(self._root_path,
                            self.get_property(["paths", "hkl_symmetry_path"]))
    
    @property
    def unit_cell_generator_path(self):
        return os.path.join(self._root_path,
                            self.get_property(["paths", "unit_cell_generator_path"]))
    
    @property
    def prediction_sanity_check_path(self):
        return os.path.join(self._root_path,
                            self.get_property(["paths", "prediction_sanity_check_path"]))
    
    @property
    def ann_toxic_cases_path(self):
        return os.path.join(self._root_path,
                            self.get_property(["paths", "ann_toxic_cases_path"]))
    
    @property
    def wrong_classifications_path(self):
        return os.path.join(self._root_path,
                            self.get_property(["paths", "wrong_classifications_path"]))
    
    @property
    def right_classifications_path(self):
        return os.path.join(self._root_path,
                            self.get_property(["paths", "right_classifications_path"]))
    
    @property
    def hkl_symmetry_map_path(self):
        return os.path.join(self._root_path,
                            self.get_property(["paths", "hkl_symmetry_map_path"]))
    
    @property
    def hkl_symmetry_dict_path(self):
        return os.path.join(self._root_path,
                            self.get_property(["paths", "hkl_symmetry_dict_path"]))
    
    @property
    def bad_predictions_path(self):
        return os.path.join(self._root_path,
                            self.get_property(["paths", "bad_predictions_path"]))


"""
class Config:
    def __init__(self, config_file):
        with open(config_file, 'r') as f:
            self.config = json.load(f)
    
    def __getattr__(self, item):
        return self.config.get(item, None)
    
    def set_hyperparameters(self, hyperparams):
        self.config.update(hyperparams)"""
    
