import numpy as np
import os
import copy
from tqdm import tqdm
import time
import pandas as pd
from multiprocessing import Process, Lock, Queue, Pool, Manager,\
    cpu_count
import json
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import multiprocessing as mp
import ast
from polishing_fit import (
    match_laue_indices,
    direct_inversion_loss,
    direct_inversion_pytorch
)
try:
    from seaborn import heatmap
    import seaborn as sns
except:
    sns = None
try:
    import optuna
except:
    optuna = None

# custom imports
from gixd_data_generator import GixdDataGenerator
from gixd_utility import GixdUtility, LatticeSystem
from config_loader import MachineLearningHyperparameter, PathConfig, FeatureConfig
from neural_network import *
from gixd_prediction import GixdPrediction
from timeit import default_timer as timer
from tqdm import trange

import torch
torch.multiprocessing.set_sharing_strategy('file_system') #to prevent RuntimeError: received 0 items of ancdata
torch.set_float32_matmul_precision('high')
from torchvision import transforms
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import lightning as L
from lightning.pytorch.tuner import Tuner
from lightning.pytorch.callbacks import Callback, LearningRateMonitor, DeviceStatsMonitor,\
    ModelCheckpoint, StochasticWeightAveraging
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
import torchvision
import torch.distributions as dist
from torch.optim.lr_scheduler import LambdaLR
import random
from captum.attr import IntegratedGradients

from config_loader import TargetConfig, RefinementHyperparameters, FeatureConfig,\
    MachineLearningHyperparameter


def gmm_nll_loss(y_true, means, vars, weights, eps: float = 1e-6):
    """
    Numerically stable negative log-likelihood for a diagonal GMM.

    y_true:  (B, D)
    means:   (B, K, D)
    vars:    (B, K, D)  -- interpreted as variances
    weights: (B, K)     -- logits or probabilities
    """
    # Do the distribution math in float32 (safer than bf16/float16)
    means   = means.to(torch.float32)
    vars    = vars.to(torch.float32)
    weights = weights.to(torch.float32)
    y_true  = y_true.to(torch.float32)

    # Replace NaN/Inf early, then clamp to a safe range
    # (zeros → eps, negative → eps, huge → capped)
    vars = torch.nan_to_num(vars, nan=1.0, posinf=1e6, neginf=1.0)
    vars = vars.clamp(min=eps, max=1e6)

    # Convert to std and keep it strictly positive
    scale = torch.sqrt(vars).clamp(min=eps, max=1e3)

    # Broadcast targets to (B, K, D)
    y_expanded = y_true.unsqueeze(1).expand_as(means)

    # Stabilize mixture weights:
    # If weights are logits, softmax normalizes them;
    # if already probabilities, softmax is still numerically safe.
    mix_probs = F.softmax(weights, dim=1).clamp(min=eps)
    log_mix   = torch.log(mix_probs)

    # Log-likelihood under each component, sum over D
    gaussians = dist.Normal(loc=means, scale=scale)
    log_probs = gaussians.log_prob(y_expanded).sum(dim=-1)  # (B, K)

    # Log-sum-exp across components, then mean over batch
    nll = -torch.logsumexp(log_mix + log_probs, dim=1).mean()
    return nll

def sample_from_gmm(means, vars, weights):
    """
    Samples an output from a GMM given its parameters.
    """
    num_gaussians = means.shape[1]
    
    # Sample a Gaussian component
    gaussian_idx = torch.multinomial(weights, 1).squeeze()  # Choose a Gaussian based on weights
    
    # Gather the corresponding mean and variance
    chosen_means = means[torch.arange(means.shape[0]), gaussian_idx]
    chosen_vars = vars[torch.arange(vars.shape[0]), gaussian_idx]
    
    # Sample from the selected Gaussian
    sampled_output = chosen_means + torch.sqrt(chosen_vars) * torch.randn_like(chosen_means)
    
    return sampled_output

class GixdModule(L.LightningModule):
    def __init__(self, num_phases, use_gmm: bool, num_gmm, learning_rate,
                 training_batch_size, validation_batch_size, test_batch_size,
                 simulation_batch_size,
                 max_epochs,
                 training_samples_per_epoch, validation_samples_per_epoch, test_samples_per_epoch,
                 training_data_generator:GixdDataGenerator,
                 validation_data_generator:GixdDataGenerator,
                 test_data_generator:GixdDataGenerator, pretrained: bool = True):
        super().__init__()
        self.save_hyperparameters(ignore=[
            "training_data_generator", "validation_data_generator", "test_data_generator"
        ])
        self.num_phases = num_phases
        self.use_gmm = use_gmm
        self.gixd_model = GixdModel(num_phases, True, num_gmm) if use_gmm \
            else GixdPointModel(num_phases, use_transformer=False, multi_head=True, pretrained=pretrained)
        self.learning_rate = learning_rate
        self.training_batch_size = training_batch_size
        self.validation_batch_size = validation_batch_size
        self.test_batch_size = test_batch_size
        self.simulation_batch_size = simulation_batch_size
        self.max_epochs = int(max_epochs)
        self.training_samples_per_epoch = training_samples_per_epoch
        self.validation_samples_per_epoch = validation_samples_per_epoch
        self.test_samples_per_epoch = test_samples_per_epoch
        self.training_data_generator = training_data_generator
        self.validation_data_generator = validation_data_generator
        self.test_data_generator = test_data_generator
        self.transforms = transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    
    def _apply_runtime_config_to_generators(self):
        """Make all underlying generators reflect the configured batch/sample sizes."""
        def _set(gen_dict, batch_size, samples_per_epoch):
            if isinstance(gen_dict, dict):
                uniform = gen_dict.get("uniform")
                if uniform is not None:
                    uniform.batch_size = batch_size
                    uniform.epoch_size = samples_per_epoch
                stream = gen_dict.get("stream")
                if stream is not None:
                    stream.batch_size = batch_size
                    stream.samples_per_epoch = samples_per_epoch
                    stream.chunk = max(1, self.simulation_batch_size)
                    stream.buffer_cap = max(3 * batch_size, stream.chunk)
            else:
                gen_dict.batch_size = batch_size
                gen_dict.epoch_size = samples_per_epoch

        _set(self.training_data_generator, self.training_batch_size, self.training_samples_per_epoch)
        _set(self.validation_data_generator, self.validation_batch_size, self.validation_samples_per_epoch)
        _set(self.test_data_generator, self.test_batch_size, self.test_samples_per_epoch)

    # Lightning calls this before/around the batch-size search and regular fit
    def on_fit_start(self):
        self._apply_runtime_config_to_generators()

    def on_train_epoch_start(self):
        self._apply_runtime_config_to_generators()

    def on_validation_epoch_start(self):
        self._apply_runtime_config_to_generators()

    @staticmethod
    def _uniform_generator(generator_dict):
        if isinstance(generator_dict, dict):
            return generator_dict.get("uniform")
        return generator_dict

    def _capture_generator_rng_states(self):
        states = {}
        for name, generator_dict in (
            ("training", self.training_data_generator),
            ("validation", self.validation_data_generator),
            ("testing", self.test_data_generator),
        ):
            generator = self._uniform_generator(generator_dict)
            if generator is None:
                continue
            state = {}
            rng = getattr(generator, "rng", None)
            if rng is not None:
                state["generator_rng"] = copy.deepcopy(rng.bit_generator.state)
            unit_cell_generator = getattr(generator, "unit_cell_generator", None)
            unit_cell_rng = getattr(unit_cell_generator, "rng", None)
            if unit_cell_rng is not None:
                state["unit_cell_rng"] = copy.deepcopy(unit_cell_rng.bit_generator.state)
            if state:
                states[name] = state
        return states

    def _restore_generator_rng_states(self, states):
        for name, generator_dict in (
            ("training", self.training_data_generator),
            ("validation", self.validation_data_generator),
            ("testing", self.test_data_generator),
        ):
            state = states.get(name)
            generator = self._uniform_generator(generator_dict)
            if not state or generator is None:
                continue
            if "generator_rng" in state:
                generator.rng.bit_generator.state = state["generator_rng"]
            if "unit_cell_rng" in state and getattr(generator, "unit_cell_generator", None):
                generator.unit_cell_generator.rng.bit_generator.state = state["unit_cell_rng"]

    def on_save_checkpoint(self, checkpoint):
        """Persist all random streams at an epoch-boundary rolling checkpoint."""
        rng_state = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "data_generators": self._capture_generator_rng_states(),
        }
        if torch.cuda.is_available():
            rng_state["torch_cuda"] = torch.cuda.get_rng_state_all()
        checkpoint["gidind_ai_rng_state"] = rng_state

    def on_load_checkpoint(self, checkpoint):
        """Restore random streams before Lightning resumes the next epoch."""
        rng_state = checkpoint.get("gidind_ai_rng_state")
        if not rng_state:
            return
        random.setstate(rng_state["python"])
        np.random.set_state(rng_state["numpy"])
        torch.set_rng_state(rng_state["torch_cpu"])
        if torch.cuda.is_available() and "torch_cuda" in rng_state:
            torch.cuda.set_rng_state_all(rng_state["torch_cuda"])
        self._restore_generator_rng_states(rng_state.get("data_generators", {}))

    def compute_min_mae(self, targets: torch.Tensor, means: torch.Tensor) -> torch.Tensor:
        """
        For each sample, compute the MAE to every GMM component mean and take the minimum.
        Finally, average across the batch.
        targets: [B, D]
        means:   [B, K, D]
        Returns a scalar tensor.
        """
        targets = targets.to(torch.float32)
        means   = means.to(torch.float32)

        # [B, K, D]: absolute error per component
        abs_err = (means - targets.unsqueeze(1)).abs()
        # [B, K]: MAE over D for each component
        comp_mae = abs_err.mean(dim=-1)
        # [B]: best component per sample
        best_mae = comp_mae.min(dim=1).values
        # scalar
        return best_mae.mean()


    def compute_max_weight_mae(self, targets: torch.Tensor, means: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """
        For each sample, select the component with the largest weight and compute the MAE to its mean.
        Finally, average across the batch.
        targets:  [B, D]
        means:    [B, K, D]
        weights:  [B, K]
        Returns a scalar tensor.
        """
        targets = targets.to(torch.float32)
        means   = means.to(torch.float32)
        weights = weights.to(torch.float32)

        # [B]: index of the heaviest component per sample
        top_idx = weights.argmax(dim=1)

        # Gather the corresponding means: [B, D]
        b = torch.arange(means.size(0), device=means.device)
        top_means = means[b, top_idx, :]

        # MAE over D, then average over batch
        return torch.nn.functional.l1_loss(top_means, targets, reduction="mean")

    def forward(self, inputs):
        imgs, top_layers = inputs
        if imgs.dim() == 5:
            imgs = imgs.squeeze(0).permute(0, 3, 1, 2)
        elif imgs.dim() == 4:
            # Rasterizer output is [batch, height, width, channels].
            imgs = imgs.permute(0, 3, 1, 2)
        imgs = imgs.float().to(self.device)
        # The rasterizer already excludes the direct-beam pixel in every
        # channel. Do not modify these NCHW views of buffered image storage.
        top_layers = top_layers.to(self.device).view(-1)
        return self.gixd_model(self.transforms(imgs), top_layers)


    def _shared_step(self, batch, batch_idx, stage, should_log=True, return_metrics=False):
        (inputs, top_layers), targets, lattice_system, q_xy, q_z = batch
        targets = targets.squeeze(0).float().to(self.device)
        outputs = self((inputs, top_layers))

        if self.use_gmm:
            means, vars, weights = outputs
            loss = gmm_nll_loss(targets, means, vars, weights)
            pred = (means * weights.unsqueeze(-1)).sum(dim=1)
            mae = F.l1_loss(pred, targets)
            min_mae = self.compute_min_mae(targets, means)
            max_mae = self.compute_max_weight_mae(targets, means, weights)
            std = torch.sqrt(vars).mean()
        else:
            loss = F.mse_loss(outputs, targets)
            mae = F.l1_loss(outputs, targets)
            min_mae = mae
            max_mae = mae
            std = torch.zeros((), dtype=loss.dtype, device=loss.device)

        metrics = {
            f"{stage}_loss": loss,
            f"{stage}_mae": mae,
            f"{stage}_min_mae": min_mae,
            f"{stage}_max_weight_mae": max_mae,
            f"{stage}_std": std,
        }
        if should_log:
            self.log_dict(metrics, prog_bar=True)
        if return_metrics:
            return metrics
        return loss

    def training_step(self, batch, batch_idx): return self._shared_step(batch, batch_idx, "train")
    def validation_step(self, batch, batch_idx): return self._shared_step(batch, batch_idx, "val")
    def test_step(self, batch, batch_idx): return self._shared_step(batch, batch_idx, "test")

    @torch.no_grad()
    def summarize_single_batch(self, batch, stage):
        previous_mode = self.training
        self.eval()
        metrics = self._shared_step(batch, 0, stage, should_log=False, return_metrics=True)
        if previous_mode:
            self.train()
        summary = {}
        for key, value in metrics.items():
            if torch.is_tensor(value):
                summary[key] = float(value.detach().cpu().item())
            else:
                summary[key] = float(value)
        return summary

    def _choose_generator(self, generator_dict):
        """Pick filtered stream if present, else the uniform generator."""
        if isinstance(generator_dict, dict) and (generator_dict.get("stream") is not None):
            # filtered stream available
            # print(f"Using filtered stream generator for {stage}")
            return generator_dict["stream"]
        # print(f"Using default (unfiltered) generator for {stage}")
        return generator_dict["uniform"] if isinstance(generator_dict, dict) else generator_dict

    def _dataloader(self, generator_dict, batch_size, samples_per_epoch):
        if isinstance(generator_dict, dict):
            uniform = generator_dict["uniform"]
            uniform.batch_size = batch_size
            uniform.epoch_size = samples_per_epoch

            stream = generator_dict.get("stream")
            if stream is not None:
                stream.batch_size = batch_size
                stream.samples_per_epoch = samples_per_epoch

        gen = self._choose_generator(generator_dict)
        return DataLoader(
            gen,
            batch_size=None,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False
        )

    def train_dataloader(self): return self._dataloader(
        self.training_data_generator, self.training_batch_size, self.training_samples_per_epoch)
    def val_dataloader(self):   return self._dataloader(
        self.validation_data_generator, self.validation_batch_size, self.validation_samples_per_epoch)
    def test_dataloader(self):  return self._dataloader(
        self.test_data_generator, self.test_batch_size, self.test_samples_per_epoch)

    def get_optimizer_schedule_config(self):
        total_epochs = max(3, int(self.max_epochs))
        warmup_epochs = max(1, int(round(total_epochs * 0.05)))
        hold_epochs = 0
        swa_epochs = max(1, int(round(total_epochs * 0.10)))
        decay_epochs = total_epochs - warmup_epochs - hold_epochs - swa_epochs

        if decay_epochs < 1:
            decay_epochs = 1
            if warmup_epochs + hold_epochs + decay_epochs + swa_epochs > total_epochs:
                hold_epochs = max(0, total_epochs - warmup_epochs - decay_epochs - swa_epochs)
        if warmup_epochs + hold_epochs + decay_epochs + swa_epochs > total_epochs:
            swa_epochs = max(1, total_epochs - warmup_epochs - hold_epochs - decay_epochs)

        min_lr_ratio = 0.1
        min_lr = self.learning_rate * min_lr_ratio
        swa_epoch_start = total_epochs - swa_epochs + 1
        return {
            "total_epochs": total_epochs,
            "warmup_epochs": warmup_epochs,
            "hold_epochs": hold_epochs,
            "decay_epochs": decay_epochs,
            "swa_epochs": swa_epochs,
            "min_lr_ratio": min_lr_ratio,
            "min_lr": min_lr,
            "swa_epoch_start": swa_epoch_start,
        }

    def configure_optimizers(self):
        schedule_cfg = self.get_optimizer_schedule_config()

        def lr_lambda(epoch):
            if epoch < schedule_cfg["warmup_epochs"]:
                return (epoch + 1) / schedule_cfg["warmup_epochs"]
            if epoch < schedule_cfg["warmup_epochs"] + schedule_cfg["hold_epochs"]:
                return 1.0
            if epoch < schedule_cfg["warmup_epochs"] + schedule_cfg["hold_epochs"] + schedule_cfg["decay_epochs"]:
                decay_progress = (
                    epoch - schedule_cfg["warmup_epochs"] - schedule_cfg["hold_epochs"] + 1
                ) / schedule_cfg["decay_epochs"]
                cosine_value = 0.5 * (1.0 + np.cos(np.pi * decay_progress))
                return schedule_cfg["min_lr_ratio"] + (1.0 - schedule_cfg["min_lr_ratio"]) * cosine_value
            return schedule_cfg["min_lr_ratio"]

        optimizer = torch.optim.Adam(self.gixd_model.parameters(), lr=self.learning_rate)
        scheduler = {
            "scheduler": LambdaLR(optimizer, lr_lambda=lr_lambda),
            "interval": "epoch"
        }
        return [optimizer], [scheduler]
    

class MetricsHistoryPlotCallback(Callback):
    def _metric_uses_log_scale(self, metric_name):
        return metric_name in ["loss", "mae"] or metric_name.startswith("lr")

    def _metric_label(self, metric_name):
        return metric_name.replace("_", " ").replace("-", " ").replace("/", " ").title()

    def _metric_filename(self, metric_name):
        return metric_name.replace("/", "_")

    def _get_metrics_path(self, trainer):
        for logger in trainer.loggers:
            if isinstance(logger, CSVLogger):
                return os.path.join(logger.log_dir, "metrics.csv")
        return None

    def _collect_series(self, metrics_df, metric_column):
        valid_rows = metrics_df.loc[metrics_df[metric_column].notna()].copy()
        if valid_rows.empty:
            return None, None, None

        if "step" in valid_rows.columns and valid_rows["step"].notna().any():
            x_values = valid_rows["step"].to_numpy()
            x_label = "Step"
        elif "epoch" in valid_rows.columns and valid_rows["epoch"].notna().any():
            x_values = valid_rows["epoch"].to_numpy()
            x_label = "Epoch"
        else:
            x_values = np.arange(valid_rows.shape[0])
            x_label = "Index"

        y_values = valid_rows[metric_column].to_numpy()
        return x_values, y_values, x_label

    def _plot_metric_pair(self, metrics_df, output_dir, metric_name):
        train_column = f"train_{metric_name}"
        val_column = f"val_{metric_name}"
        has_train = train_column in metrics_df.columns
        has_val = val_column in metrics_df.columns

        if not has_train and not has_val:
            return

        fig, ax = plt.subplots(figsize=(10, 6))
        plotted_anything = False
        x_label = "Step"

        if has_train:
            train_x, train_y, train_label = self._collect_series(metrics_df, train_column)
            if train_x is not None:
                x_label = train_label
                ax.plot(train_x, train_y, color="#c0392b", linewidth=2.0, label="Training")
                plotted_anything = True

        if has_val:
            val_x, val_y, val_label = self._collect_series(metrics_df, val_column)
            if val_x is not None:
                if not plotted_anything:
                    x_label = val_label
                ax.plot(val_x, val_y, color="#2980b9", linewidth=2.0, marker="o", label="Validation")
                plotted_anything = True

        if not plotted_anything:
            plt.close(fig)
            return

        metric_label = self._metric_label(metric_name)
        ax.set_xlabel(x_label)
        ax.set_ylabel(metric_label)
        ax.set_title(metric_label)
        if self._metric_uses_log_scale(metric_name):
            ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"{self._metric_filename(metric_name)}.svg"), format="svg")
        plt.close(fig)

    def _plot_single_metric(self, metrics_df, output_dir, metric_name):
        metric_x, metric_y, x_label = self._collect_series(metrics_df, metric_name)
        if metric_x is None:
            return

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(metric_x, metric_y, color="#1f7a8c", linewidth=2.0, marker="o", label=metric_name)
        metric_label = self._metric_label(metric_name)
        ax.set_xlabel(x_label)
        ax.set_ylabel(metric_label)
        ax.set_title(metric_label)
        if self._metric_uses_log_scale(metric_name):
            ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"{self._metric_filename(metric_name)}.svg"), format="svg")
        plt.close(fig)

    def _write_metric_plots(self, trainer):
        metrics_path = self._get_metrics_path(trainer)
        if metrics_path is None:
            return
        if not os.path.exists(metrics_path):
            return

        metrics_df = pd.read_csv(metrics_path)
        if metrics_df.empty:
            return

        metric_names = set()
        standalone_metric_names = set()
        for column_name in metrics_df.columns:
            if column_name.startswith("train_"):
                metric_names.add(column_name[len("train_"):])
            elif column_name.startswith("val_"):
                metric_names.add(column_name[len("val_"):])
            elif column_name not in ["epoch", "step"] and metrics_df[column_name].notna().any():
                standalone_metric_names.add(column_name)

        output_dir = os.path.dirname(metrics_path)
        for metric_name in sorted(metric_names):
            self._plot_metric_pair(metrics_df, output_dir, metric_name)
        for metric_name in sorted(standalone_metric_names):
            self._plot_single_metric(metrics_df, output_dir, metric_name)

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        for logger in trainer.loggers:
            if isinstance(logger, CSVLogger):
                logger.save()
        self._write_metric_plots(trainer)


class GixdRegressor():

    def __init__(self, training_data_generators: dict, validation_data_generators: dict,
                 test_data_generators: dict, num_phases=1, use_gmm=False,
                 refinement_backend="auto"):
        self.refinement_backend = refinement_backend
        self.ml_hyperparameters = MachineLearningHyperparameter()
        self.path_config = PathConfig()
        self.feature_config = FeatureConfig()
        self.target_config = TargetConfig()
        self.refinement_hyperparameters = RefinementHyperparameters()
        self.used_top_layers = self.feature_config.miller_indices
        self.loaded_models = dict()
        self.gixd_model = None
        self.top_layer_specific_heads = dict()
        self.feature_extractor = None
        self.training_data_generators = training_data_generators
        self.validation_data_generators = validation_data_generators
        self.test_data_generators = test_data_generators
        # init or load models and log files for all used top_layers
        self.prediction_known_hkl_df = {}
        self.use_gmm = use_gmm
        self.num_phases=num_phases
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("[Regressor]: Using device ",self.device, "for neural networks")

    def _checkpoint_init_kwargs(self):
        return {
            "num_phases": self.num_phases,
            "use_gmm": self.use_gmm,
            "num_gmm": 2 * self.num_phases,
            "learning_rate": self.ml_hyperparameters.initial_lr,
            "training_batch_size": self.ml_hyperparameters.training_batch_size,
            "validation_batch_size": self.ml_hyperparameters.validation_batch_size,
            "test_batch_size": self.ml_hyperparameters.testing_batch_size,
            "simulation_batch_size": self.ml_hyperparameters.simulation_batch_size,
            "max_epochs": self.ml_hyperparameters.num_epochs,
            "training_samples_per_epoch": self.ml_hyperparameters.training_epoch_size,
            "validation_samples_per_epoch": self.ml_hyperparameters.validation_epoch_size,
            "test_samples_per_epoch": self.ml_hyperparameters.testing_epoch_size,
            "training_data_generator": self.training_data_generators["uniform"],
            "validation_data_generator": self.validation_data_generators["uniform"],
            "test_data_generator": self.test_data_generators["uniform"],
        }

    def _load_default_checkpoint(self, checkpoint_path=None):
        checkpoint_candidates = (
            [os.fspath(checkpoint_path)] if checkpoint_path is not None else [
                os.path.join(self.path_config.model_path, "default_model_swa.ckpt"),
                os.path.join(self.path_config.model_path, "default_model.ckpt"),
            ]
        )
        checkpoint_path = None
        for candidate_path in checkpoint_candidates:
            if os.path.exists(candidate_path):
                checkpoint_path = candidate_path
                break
        if checkpoint_path is None:
            raise FileNotFoundError(
                f"Could not find checkpoint in {self.path_config.model_path}"
            )
        self.gixd_model = GixdModule.load_from_checkpoint(
            checkpoint_path,
            strict=True,
            pretrained=False,
            **self._checkpoint_init_kwargs()
        )
        self.gixd_model.to(self.device)
        return checkpoint_path

    def load_checkpoint(self, checkpoint_path):
        """Load an explicit Lightning checkpoint for a portable inference run."""
        return self._load_default_checkpoint(checkpoint_path)

    def _ensure_model_loaded(self):
        if self.gixd_model is None:
            self._load_default_checkpoint()

    def _set_testing_noise_parameters(self, num_missing_peaks, num_extra_peaks, gaussian_shift_sigma):
        test_feature_config = self.test_data_generators["uniform"].feature_config
        test_feature_config.testing_number_missing_peaks = int(num_missing_peaks)
        test_feature_config.testing_number_extra_peaks = int(num_extra_peaks)
        test_feature_config.testing_gaussian_noise_sigma = float(gaussian_shift_sigma)

    @staticmethod
    def _calculate_convergence_stats_from_dataframe(
        df,
        result_prefix="refined",
        distance_tolerance=1e-1,
        angle_tolerance=1e-1
    ):
        true_columns = [
            "true_a", "true_b", "true_c",
            "true_alpha", "true_beta", "true_gamma"
        ]
        result_columns = [
            f"{result_prefix}_a", f"{result_prefix}_b", f"{result_prefix}_c",
            f"{result_prefix}_alpha", f"{result_prefix}_beta", f"{result_prefix}_gamma"
        ]
        required_columns = true_columns + result_columns
        missing_columns = [column_name for column_name in required_columns if column_name not in df.columns]
        if missing_columns:
            raise ValueError(
                "Missing required columns for convergence analysis: "
                + ", ".join(missing_columns)
            )

        valid_df = df.dropna(subset=result_columns).copy()
        if valid_df.empty:
            return {
                "rate": 0.0,
                "num_converged": 0,
                "num_total": 0,
                "distance_tolerance": float(distance_tolerance),
                "angle_tolerance": float(angle_tolerance),
            }

        distance_errors = np.vstack([
            np.abs(valid_df["true_a"].to_numpy() - valid_df[f"{result_prefix}_a"].to_numpy()),
            np.abs(valid_df["true_b"].to_numpy() - valid_df[f"{result_prefix}_b"].to_numpy()),
            np.abs(valid_df["true_c"].to_numpy() - valid_df[f"{result_prefix}_c"].to_numpy()),
        ]).T
        angle_errors = np.vstack([
            np.abs(valid_df["true_alpha"].to_numpy() - valid_df[f"{result_prefix}_alpha"].to_numpy()),
            np.abs(valid_df["true_beta"].to_numpy() - valid_df[f"{result_prefix}_beta"].to_numpy()),
            np.abs(valid_df["true_gamma"].to_numpy() - valid_df[f"{result_prefix}_gamma"].to_numpy()),
        ]).T

        converged_mask = np.logical_and(
            np.all(distance_errors < distance_tolerance, axis=1),
            np.all(angle_errors < angle_tolerance, axis=1),
        )
        num_total = int(valid_df.shape[0])
        num_converged = int(np.sum(converged_mask))
        rate = float(num_converged / num_total) if num_total > 0 else 0.0
        return {
            "rate": rate,
            "num_converged": num_converged,
            "num_total": num_total,
            "distance_tolerance": float(distance_tolerance),
            "angle_tolerance": float(angle_tolerance),
        }

    @staticmethod
    def _calculate_convergence_rate_from_dataframe(
        df,
        result_prefix="refined",
        distance_tolerance=1e-1,
        angle_tolerance=1e-1
    ):
        return GixdRegressor._calculate_convergence_stats_from_dataframe(
            df,
            result_prefix=result_prefix,
            distance_tolerance=distance_tolerance,
            angle_tolerance=angle_tolerance,
        )["rate"]

    def _predict_normalized_cells(self, images, top_layer_indices):
        self._ensure_model_loaded()
        self.gixd_model.eval()
        if images.dim() == 3:
            images = images.unsqueeze(0)

        with torch.no_grad():
            images = images.to(self.device)
            top_layers = torch.as_tensor(
                top_layer_indices,
                dtype=torch.long,
                device=self.device
            ).view(-1)
            outputs = self.gixd_model([images, top_layers])
            if isinstance(outputs, tuple):
                means, _, weights = outputs
                pred_norm = (means * weights.unsqueeze(-1)).sum(dim=1)
            else:
                pred_norm = outputs
        return pred_norm.detach().cpu()

    def _predict_cells_from_images(self, images, top_layer_indices):
        pred_norm = self._predict_normalized_cells(images, top_layer_indices)
        pred_cells = GixdUtility.denormalize_cell_deg(pred_norm)
        if torch.is_tensor(pred_cells):
            pred_cells = pred_cells.detach().cpu().numpy()
        else:
            pred_cells = np.asarray(pred_cells)
        return pred_cells

    def _extract_clean_q_point_lists_from_dataframe(self, df, tolerance=0.01):
        q_xy_list = []
        q_z_list = []
        for q_xy_json, q_z_json in zip(df["q_xy"], df["q_z"]):
            q_xy = torch.tensor(json.loads(q_xy_json), dtype=torch.float64)
            q_z = torch.tensor(json.loads(q_z_json), dtype=torch.float64)
            clean_q_xy, clean_q_z = GixdDataGenerator.filter_q_points(q_xy, q_z, tolerance)
            q_xy_list.append(clean_q_xy.detach().cpu().numpy().astype(np.float64))
            q_z_list.append(clean_q_z.detach().cpu().numpy().astype(np.float64))
        return q_xy_list, q_z_list

    @staticmethod
    def _pad_q_point_lists(q_xy_list, q_z_list):
        if len(q_xy_list) != len(q_z_list):
            raise ValueError("q_xy_list and q_z_list must have the same length")

        max_length = max((len(q_values) for q_values in q_xy_list), default=0)
        q_xy_batch = torch.zeros((len(q_xy_list), max_length), dtype=torch.float64)
        q_z_batch = torch.zeros((len(q_z_list), max_length), dtype=torch.float64)

        for idx, (q_xy_values, q_z_values) in enumerate(zip(q_xy_list, q_z_list)):
            if len(q_xy_values) == 0:
                continue
            q_xy_batch[idx, :len(q_xy_values)] = torch.as_tensor(q_xy_values, dtype=torch.float64)
            q_z_batch[idx, :len(q_z_values)] = torch.as_tensor(q_z_values, dtype=torch.float64)
        return q_xy_batch, q_z_batch

    @staticmethod
    def _make_benchmark_rng(rng_seed, sample_index, stream_id):
        return np.random.default_rng(int(rng_seed) + 100_003 * int(stream_id) + int(sample_index))

    def _clip_q_points_for_testing(self, q_xy, q_z):
        q_xy = np.asarray(q_xy, dtype=np.float64).copy()
        q_z = np.asarray(q_z, dtype=np.float64).copy()

        q_too_large = np.logical_or(q_z >= self.feature_config.max_q_z, q_xy >= self.feature_config.max_q_xy)
        if self.feature_config.use_specular_scan:
            q_too_small = np.logical_and(q_xy < 0.0, q_z < 0.0)
        else:
            q_too_small = np.logical_or(q_xy <= self.feature_config.specular_cutoff, q_z < 0.0)

        q_out_of_bounds = np.logical_or(q_too_large, q_too_small)
        q_xy[q_out_of_bounds] = 0.0
        q_z[q_out_of_bounds] = 0.0
        return q_xy, q_z

    def _apply_missing_peaks_to_clean_q(self, q_xy, q_z, num_missing, sample_index, rng_seed):
        q_xy_case = np.asarray(q_xy, dtype=np.float64).copy()
        q_z_case = np.asarray(q_z, dtype=np.float64).copy()
        if num_missing <= 0 or q_xy_case.size == 0:
            return q_xy_case, q_z_case

        rng = self._make_benchmark_rng(rng_seed, sample_index, stream_id=11)
        candidate_indices = np.arange(q_xy_case.size, dtype=np.int64)
        if candidate_indices.size > 1:
            ordered_indices = np.concatenate([
                candidate_indices[:1],
                rng.permutation(candidate_indices[1:])
            ])
        else:
            ordered_indices = candidate_indices

        remove_indices = ordered_indices[:min(int(num_missing), ordered_indices.size)]
        q_xy_case[remove_indices] = 0.0
        q_z_case[remove_indices] = 0.0
        return q_xy_case, q_z_case

    def _apply_spurious_peaks_to_clean_q(self, q_xy, q_z, num_spurious, sample_index, rng_seed):
        q_xy_case = np.asarray(q_xy, dtype=np.float64).copy()
        q_z_case = np.asarray(q_z, dtype=np.float64).copy()
        if num_spurious <= 0:
            return q_xy_case, q_z_case

        rng = self._make_benchmark_rng(rng_seed, sample_index, stream_id=17)
        q_xy_extra = rng.uniform(
            0.0,
            self.feature_config.max_q_xy,
            size=int(num_spurious)
        ).astype(np.float64)
        q_z_extra = rng.uniform(
            0.0,
            self.feature_config.max_q_z,
            size=int(num_spurious)
        ).astype(np.float64)
        return (
            np.concatenate([q_xy_case, q_xy_extra]),
            np.concatenate([q_z_case, q_z_extra]),
        )

    def _apply_shift_noise_to_clean_q(self, q_xy, q_z, shift_sigma, sample_index, rng_seed):
        q_xy_case = np.asarray(q_xy, dtype=np.float64).copy()
        q_z_case = np.asarray(q_z, dtype=np.float64).copy()
        if shift_sigma <= 0.0 or q_xy_case.size == 0:
            return q_xy_case, q_z_case

        rng = self._make_benchmark_rng(rng_seed, sample_index, stream_id=23)
        q_xy_shifted = q_xy_case + float(shift_sigma) * rng.normal(0.0, 1.0, size=q_xy_case.size).astype(np.float64)
        q_z_shifted = q_z_case + float(shift_sigma) * rng.normal(0.0, 1.0, size=q_z_case.size).astype(np.float64)
        return self._clip_q_points_for_testing(q_xy_shifted, q_z_shifted)

    def _build_prediction_dataframe_from_q_point_lists(self, base_df, q_xy_list, q_z_list):
        q_xy_batch, q_z_batch = self._pad_q_point_lists(q_xy_list, q_z_list)
        images = self.test_data_generators["uniform"].calc_images_from_q_points(q_xy_batch, q_z_batch)
        top_layer_indices = base_df["true_top_layer"].astype(int).to_numpy()
        pred_cells = self._predict_cells_from_images(images, top_layer_indices)
        nonzero_counts = [
            int((image > 0).sum().item() // self.feature_config.number_channels)
            for image in images
        ]

        rows = []
        base_df = base_df.reset_index(drop=True)
        for idx, row in base_df.iterrows():
            rows.append({
                "sample_index": int(row["sample_index"]),
                "true_a": float(row["true_a"]),
                "true_b": float(row["true_b"]),
                "true_c": float(row["true_c"]),
                "true_alpha": float(row["true_alpha"]),
                "true_beta": float(row["true_beta"]),
                "true_gamma": float(row["true_gamma"]),
                "pred_a": float(pred_cells[idx, 0]),
                "pred_b": float(pred_cells[idx, 1]),
                "pred_c": float(pred_cells[idx, 2]),
                "pred_alpha": float(pred_cells[idx, 3]),
                "pred_beta": float(pred_cells[idx, 4]),
                "pred_gamma": float(pred_cells[idx, 5]),
                "lattice_system": int(row["lattice_system"]),
                "true_top_layer": int(row["true_top_layer"]),
                "assumed_top_layer": int(row["true_top_layer"]),
                "nonzero_pixel_count": nonzero_counts[idx],
                "q_xy": json.dumps(np.asarray(q_xy_list[idx], dtype=float).tolist()),
                "q_z": json.dumps(np.asarray(q_z_list[idx], dtype=float).tolist()),
            })
        return pd.DataFrame(rows)

    def _write_fixed_ground_truth_case(
        self,
        base_df,
        q_xy_list,
        q_z_list,
        prediction_csv,
        refined_csv,
        num_processes
    ):
        prediction_df = self._build_prediction_dataframe_from_q_point_lists(base_df, q_xy_list, q_z_list)
        prediction_df.to_csv(prediction_csv, index=False)
        self.refine_predictions(
            input_csv=prediction_csv,
            output_csv=refined_csv,
            num_processes=num_processes
        )
        refined_df = pd.read_csv(refined_csv)
        return prediction_df, refined_df, self._calculate_convergence_stats_from_dataframe(
            refined_df,
            result_prefix="refined"
        )

    def _plot_missing_extra_peak_benchmark(self, benchmark_df, output_svg):
        fig, ax = plt.subplots(figsize=(10, 6))
        negative_df = benchmark_df[benchmark_df["x_value"] < 0]
        positive_df = benchmark_df[benchmark_df["x_value"] > 0]
        zero_df = benchmark_df[benchmark_df["x_value"] == 0]

        if not negative_df.empty:
            ax.axvspan(negative_df["x_value"].min() - 0.5, -0.5, color="#f4cccc", alpha=0.35)
            ax.plot(
                negative_df["x_value"],
                100.0 * negative_df["convergence_rate"],
                color="#c0392b",
                marker="o",
                linewidth=2.0,
                label="Missing peaks"
            )
        if not positive_df.empty:
            ax.axvspan(0.5, positive_df["x_value"].max() + 0.5, color="#cfe2f3", alpha=0.35)
            ax.plot(
                positive_df["x_value"],
                100.0 * positive_df["convergence_rate"],
                color="#2980b9",
                marker="o",
                linewidth=2.0,
                label="Extra peaks"
            )
        if not zero_df.empty:
            ax.plot(
                zero_df["x_value"],
                100.0 * zero_df["convergence_rate"],
                color="#555555",
                marker="o",
                linewidth=2.0,
                label="No perturbation"
            )

        ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
        ax.set_xlabel("Missing / Extra Peaks")
        ax.set_ylabel("Convergence Rate (%)")
        ax.set_title("Refinement Convergence vs Missing or Extra Peaks")
        ax.set_xticks(benchmark_df["x_value"].tolist())
        ax.set_ylim(0.0, 100.0)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_svg, format="svg")
        plt.close(fig)

    def _plot_shift_benchmark(self, benchmark_df, output_svg):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(
            benchmark_df["shift_sigma"],
            100.0 * benchmark_df["convergence_rate"],
            color="#1f7a8c",
            marker="o",
            linewidth=2.0
        )
        ax.set_xlabel("Random Peak Shift Amplitude")
        ax.set_ylabel("Convergence Rate (%)")
        ax.set_title("Refinement Convergence vs Peak Position Shift")
        ax.set_ylim(0.0, 100.0)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_svg, format="svg")
        plt.close(fig)

    def run_noise_benchmarks(
        self,
        output_dir,
        num_samples=1024,
        max_missing_or_extra_peaks=7,
        num_shift_values=20,
        max_shift_amplitude=0.02,
        num_processes=1,
    ):
        os.makedirs(output_dir, exist_ok=True)
        self._ensure_model_loaded()

        test_feature_config = self.test_data_generators["uniform"].feature_config
        original_missing = test_feature_config.testing_number_missing_peaks
        original_extra = test_feature_config.testing_number_extra_peaks
        original_shift = test_feature_config.testing_gaussian_noise_sigma

        missing_extra_rows = []
        for x_value in range(-max_missing_or_extra_peaks, max_missing_or_extra_peaks + 1):
            num_missing = max(0, -x_value)
            num_extra = max(0, x_value)
            self._set_testing_noise_parameters(num_missing, num_extra, 0.0)

            case_tag = f"missing_{num_missing}" if x_value < 0 else \
                f"extra_{num_extra}" if x_value > 0 else "baseline"
            prediction_csv = os.path.join(output_dir, f"{case_tag}_predictions.csv")
            refined_csv = os.path.join(output_dir, f"{case_tag}_refined.csv")

            self.predict_and_save_csv(
                num_samples=num_samples,
                output_csv=prediction_csv,
                unknown_top_layer=False
            )
            self.refine_predictions(
                input_csv=prediction_csv,
                output_csv=refined_csv,
                num_processes=num_processes
            )
            refined_df = pd.read_csv(refined_csv)
            convergence_rate = self._calculate_convergence_rate_from_dataframe(refined_df)
            missing_extra_rows.append({
                "x_value": x_value,
                "num_missing_peaks": num_missing,
                "num_extra_peaks": num_extra,
                "convergence_rate": convergence_rate
            })

        missing_extra_df = pd.DataFrame(missing_extra_rows).sort_values("x_value")
        missing_extra_df.to_csv(
            os.path.join(output_dir, "missing_extra_peak_benchmark.csv"),
            index=False
        )
        self._plot_missing_extra_peak_benchmark(
            missing_extra_df,
            os.path.join(output_dir, "missing_extra_peak_benchmark.svg")
        )

        shift_rows = []
        shift_values = np.linspace(0.0, max_shift_amplitude, num_shift_values)
        for shift_sigma in shift_values:
            self._set_testing_noise_parameters(0, 0, shift_sigma)
            shift_tag = f"shift_{shift_sigma:.6f}".replace(".", "p")
            prediction_csv = os.path.join(output_dir, f"{shift_tag}_predictions.csv")
            refined_csv = os.path.join(output_dir, f"{shift_tag}_refined.csv")

            self.predict_and_save_csv(
                num_samples=num_samples,
                output_csv=prediction_csv,
                unknown_top_layer=False
            )
            self.refine_predictions(
                input_csv=prediction_csv,
                output_csv=refined_csv,
                num_processes=num_processes
            )
            refined_df = pd.read_csv(refined_csv)
            convergence_rate = self._calculate_convergence_rate_from_dataframe(refined_df)
            shift_rows.append({
                "shift_sigma": float(shift_sigma),
                "convergence_rate": convergence_rate
            })

        shift_df = pd.DataFrame(shift_rows).sort_values("shift_sigma")
        shift_df.to_csv(
            os.path.join(output_dir, "shift_noise_benchmark.csv"),
            index=False
        )
        self._plot_shift_benchmark(
            shift_df,
            os.path.join(output_dir, "shift_noise_benchmark.svg")
        )

        self._set_testing_noise_parameters(original_missing, original_extra, original_shift)
        return {
            "missing_extra_csv": os.path.join(output_dir, "missing_extra_peak_benchmark.csv"),
            "missing_extra_svg": os.path.join(output_dir, "missing_extra_peak_benchmark.svg"),
            "shift_csv": os.path.join(output_dir, "shift_noise_benchmark.csv"),
            "shift_svg": os.path.join(output_dir, "shift_noise_benchmark.svg"),
        }

    def _plot_missing_spurious_convergence(self, benchmark_df, output_svg):
        fig, ax = plt.subplots(figsize=(10, 6))
        negative_df = benchmark_df[benchmark_df["x_value"] < 0]
        positive_df = benchmark_df[benchmark_df["x_value"] > 0]
        zero_df = benchmark_df[benchmark_df["x_value"] == 0]

        if not negative_df.empty:
            ax.axvspan(negative_df["x_value"].min() - 0.5, -0.5, color="#f4cccc", alpha=0.35)
            ax.plot(
                negative_df["x_value"],
                100.0 * negative_df["convergence_rate"],
                color="#c0392b",
                marker="o",
                linewidth=2.0,
                label="Missing peaks"
            )
        if not positive_df.empty:
            ax.axvspan(0.5, positive_df["x_value"].max() + 0.5, color="#cfe2f3", alpha=0.35)
            ax.plot(
                positive_df["x_value"],
                100.0 * positive_df["convergence_rate"],
                color="#2980b9",
                marker="o",
                linewidth=2.0,
                label="Spurious peaks"
            )
        if not zero_df.empty:
            ax.plot(
                zero_df["x_value"],
                100.0 * zero_df["convergence_rate"],
                color="#555555",
                marker="o",
                linewidth=2.0,
                label="No perturbation"
            )

        ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
        ax.set_xlabel("Missing / Spurious Points")
        ax.set_ylabel("Convergence Rate (%)")
        ax.set_title("Refinement Convergence vs Missing or Spurious Points")
        ax.set_xticks(benchmark_df["x_value"].tolist())
        ax.set_ylim(0.0, 100.0)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_svg, format="svg")
        plt.close(fig)

    def _plot_fixed_shift_convergence(self, benchmark_df, output_svg):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(
            benchmark_df["shift_sigma"],
            100.0 * benchmark_df["convergence_rate"],
            color="#1f7a8c",
            marker="o",
            linewidth=2.0
        )
        ax.set_xlabel("Peak Position Noise Sigma")
        ax.set_ylabel("Convergence Rate (%)")
        ax.set_title("Refinement Convergence vs Peak Position Noise")
        ax.set_ylim(0.0, 100.0)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_svg, format="svg")
        plt.close(fig)

    def run_fixed_ground_truth_convergence_benchmarks(
        self,
        output_dir,
        num_samples=1024,
        max_missing_points=8,
        max_spurious_points=5,
        num_shift_values=20,
        max_shift_amplitude=0.01,
        num_processes=1,
        rng_seed=42,
    ):
        os.makedirs(output_dir, exist_ok=True)
        missing_spurious_dir = os.path.join(output_dir, "missing_spurious_cases")
        shift_dir = os.path.join(output_dir, "shift_cases")
        os.makedirs(missing_spurious_dir, exist_ok=True)
        os.makedirs(shift_dir, exist_ok=True)

        self._ensure_model_loaded()
        test_feature_config = self.test_data_generators["uniform"].feature_config
        original_missing = test_feature_config.testing_number_missing_peaks
        original_extra = test_feature_config.testing_number_extra_peaks
        original_shift = test_feature_config.testing_gaussian_noise_sigma

        try:
            self._set_testing_noise_parameters(0, 0, 0.0)

            baseline_prediction_csv = os.path.join(output_dir, "baseline_predictions.csv")
            baseline_refined_csv = os.path.join(output_dir, "baseline_refined.csv")
            self.predict_and_save_csv(
                num_samples=num_samples,
                output_csv=baseline_prediction_csv,
                unknown_top_layer=False
            )
            self.refine_predictions(
                input_csv=baseline_prediction_csv,
                output_csv=baseline_refined_csv,
                num_processes=num_processes
            )

            baseline_prediction_df = pd.read_csv(baseline_prediction_csv).reset_index(drop=True)
            baseline_refined_df = pd.read_csv(baseline_refined_csv).reset_index(drop=True)
            baseline_convergence = self._calculate_convergence_stats_from_dataframe(
                baseline_refined_df,
                result_prefix="refined",
                distance_tolerance=1e-1,
                angle_tolerance=1e-1
            )
            clean_q_xy_list, clean_q_z_list = self._extract_clean_q_point_lists_from_dataframe(
                baseline_prediction_df
            )

            missing_spurious_rows = []
            x_values = list(range(-int(max_missing_points), 0)) + list(range(0, int(max_spurious_points) + 1))
            for x_value in x_values:
                num_missing_peaks = max(0, -int(x_value))
                num_spurious_peaks = max(0, int(x_value))
                if x_value < 0:
                    case_tag = f"missing_{num_missing_peaks:02d}"
                elif x_value > 0:
                    case_tag = f"spurious_{num_spurious_peaks:02d}"
                else:
                    case_tag = "baseline"

                prediction_csv = os.path.join(missing_spurious_dir, f"{case_tag}_predictions.csv")
                refined_csv = os.path.join(missing_spurious_dir, f"{case_tag}_refined.csv")

                if x_value == 0:
                    baseline_prediction_df.to_csv(prediction_csv, index=False)
                    baseline_refined_df.to_csv(refined_csv, index=False)
                    convergence_stats = baseline_convergence
                else:
                    q_xy_case_list = []
                    q_z_case_list = []
                    for sample_index, (q_xy_values, q_z_values) in enumerate(zip(clean_q_xy_list, clean_q_z_list)):
                        if x_value < 0:
                            q_xy_case, q_z_case = self._apply_missing_peaks_to_clean_q(
                                q_xy_values,
                                q_z_values,
                                num_missing_peaks,
                                sample_index,
                                rng_seed
                            )
                        else:
                            q_xy_case, q_z_case = self._apply_spurious_peaks_to_clean_q(
                                q_xy_values,
                                q_z_values,
                                num_spurious_peaks,
                                sample_index,
                                rng_seed
                            )
                        q_xy_case_list.append(q_xy_case)
                        q_z_case_list.append(q_z_case)

                    _, _, convergence_stats = self._write_fixed_ground_truth_case(
                        baseline_prediction_df,
                        q_xy_case_list,
                        q_z_case_list,
                        prediction_csv,
                        refined_csv,
                        num_processes
                    )

                missing_spurious_rows.append({
                    "x_value": int(x_value),
                    "num_missing_peaks": int(num_missing_peaks),
                    "num_spurious_peaks": int(num_spurious_peaks),
                    "convergence_rate": float(convergence_stats["rate"]),
                })

            missing_spurious_df = pd.DataFrame(missing_spurious_rows).sort_values("x_value")
            missing_spurious_summary_csv = os.path.join(output_dir, "missing_spurious_convergence_summary.csv")
            missing_spurious_svg = os.path.join(output_dir, "missing_spurious_convergence.svg")
            missing_spurious_df.to_csv(missing_spurious_summary_csv, index=False)
            self._plot_missing_spurious_convergence(missing_spurious_df, missing_spurious_svg)

            shift_rows = []
            shift_values = np.linspace(0.0, float(max_shift_amplitude), int(num_shift_values))
            for shift_sigma in shift_values:
                shift_tag = f"shift_{shift_sigma:.6f}".replace(".", "p")
                prediction_csv = os.path.join(shift_dir, f"{shift_tag}_predictions.csv")
                refined_csv = os.path.join(shift_dir, f"{shift_tag}_refined.csv")

                if np.isclose(float(shift_sigma), 0.0):
                    baseline_prediction_df.to_csv(prediction_csv, index=False)
                    baseline_refined_df.to_csv(refined_csv, index=False)
                    convergence_stats = baseline_convergence
                else:
                    q_xy_case_list = []
                    q_z_case_list = []
                    for sample_index, (q_xy_values, q_z_values) in enumerate(zip(clean_q_xy_list, clean_q_z_list)):
                        q_xy_case, q_z_case = self._apply_shift_noise_to_clean_q(
                            q_xy_values,
                            q_z_values,
                            float(shift_sigma),
                            sample_index,
                            rng_seed
                        )
                        q_xy_case_list.append(q_xy_case)
                        q_z_case_list.append(q_z_case)

                    _, _, convergence_stats = self._write_fixed_ground_truth_case(
                        baseline_prediction_df,
                        q_xy_case_list,
                        q_z_case_list,
                        prediction_csv,
                        refined_csv,
                        num_processes
                    )

                shift_rows.append({
                    "shift_sigma": float(shift_sigma),
                    "convergence_rate": float(convergence_stats["rate"]),
                })

            shift_df = pd.DataFrame(shift_rows).sort_values("shift_sigma")
            shift_summary_csv = os.path.join(output_dir, "shift_convergence_summary.csv")
            shift_svg = os.path.join(output_dir, "shift_convergence.svg")
            shift_df.to_csv(shift_summary_csv, index=False)
            self._plot_fixed_shift_convergence(shift_df, shift_svg)
        finally:
            self._set_testing_noise_parameters(original_missing, original_extra, original_shift)

        return {
            "baseline_predictions_csv": baseline_prediction_csv,
            "baseline_refined_csv": baseline_refined_csv,
            "missing_spurious_summary_csv": missing_spurious_summary_csv,
            "missing_spurious_svg": missing_spurious_svg,
            "shift_summary_csv": shift_summary_csv,
            "shift_svg": shift_svg,
        }

    def _print_final_swa_batch_metrics(self):
        stage_specs = [
            ("train", self.gixd_model.train_dataloader),
            ("val", self.gixd_model.val_dataloader),
        ]

        for stage_name, dataloader_fn in stage_specs:
            batch = next(iter(dataloader_fn()))
            metrics = self.gixd_model.summarize_single_batch(batch, stage_name)
            metric_order = [
                f"{stage_name}_loss",
                f"{stage_name}_mae",
                f"{stage_name}_min_mae",
                f"{stage_name}_max_weight_mae",
                f"{stage_name}_std",
            ]
            metric_text = ", ".join(
                [f"{metric_name}={metrics[metric_name]:.6e}" for metric_name in metric_order]
            )
            print(f"[Final SWA] {stage_name}: {metric_text}", flush=True)
    
    def train_gixd_model(self, model_name="default_model", resume_checkpoint=None):
        from gidind_ai.training import ACCUMULATE_GRAD_BATCHES
        from gidind_ai.training_runtime import configure_training_seed

        torch.set_num_threads(min(4, torch.get_num_threads()))
        configure_training_seed(
            self.ml_hyperparameters.training_seed,
            deterministic=self.ml_hyperparameters.deterministic_algorithms,
        )
        self.gixd_model = GixdModule(
            num_phases=self.num_phases,
            use_gmm=self.use_gmm, num_gmm=2*self.num_phases,
            learning_rate=self.ml_hyperparameters.initial_lr,
            training_batch_size=self.ml_hyperparameters.training_batch_size,
            validation_batch_size=self.ml_hyperparameters.validation_batch_size,
            test_batch_size=self.ml_hyperparameters.testing_batch_size,
            simulation_batch_size=self.ml_hyperparameters.simulation_batch_size,
            max_epochs=self.ml_hyperparameters.num_epochs,
            training_samples_per_epoch=self.ml_hyperparameters.training_epoch_size,
            validation_samples_per_epoch=self.ml_hyperparameters.validation_epoch_size,
            test_samples_per_epoch=self.ml_hyperparameters.testing_epoch_size,
            training_data_generator=self.training_data_generators,
            validation_data_generator=self.validation_data_generators,
            test_data_generator=self.test_data_generators)
        
        # Ensure that all 48 heads are instantiated with batch size = 1:
        num_heads = len(self.feature_config.miller_indices)
        for i in range(num_heads):
            dummy_input = torch.randn(1, self.feature_config.image_size,
                                  self.feature_config.image_size,
                                  self.feature_config.number_channels).to(self.device)
            dummy_top = torch.tensor([[i]]).to(self.device).long()
            _ = self.gixd_model([dummy_input, dummy_top])



        logger = [TensorBoardLogger(self.path_config.training_history_path),
                  CSVLogger(self.path_config.training_history_path, "history.csv")]
        schedule_cfg = self.gixd_model.get_optimizer_schedule_config()
        print(
            f"[TrainingSchedule] total_epochs={schedule_cfg['total_epochs']} "
            f"warmup_epochs={schedule_cfg['warmup_epochs']} "
            f"hold_epochs={schedule_cfg['hold_epochs']} "
            f"decay_epochs={schedule_cfg['decay_epochs']} "
            f"swa_epochs={schedule_cfg['swa_epochs']} "
            f"base_lr={self.ml_hyperparameters.initial_lr:.6e} "
            f"min_lr={schedule_cfg['min_lr']:.6e} "
            f"swa_epoch_start={schedule_cfg['swa_epoch_start']}",
            flush=True
        )
        rolling_checkpoint = ModelCheckpoint(
            dirpath=self.path_config.model_path,
            save_last=True,
            save_top_k=0,
            every_n_epochs=self.ml_hyperparameters.checkpoint_every_n_epochs,
            save_on_train_epoch_end=False,
            save_on_exception=True,
        )
        trainer = L.Trainer(max_epochs=self.ml_hyperparameters.num_epochs,
                            accelerator="gpu" if torch.cuda.is_available() else "cpu",
                            #profiler="simple",
                            precision=("bf16-mixed" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "32-true"), #16-mixed 32-true
                            accumulate_grad_batches=ACCUMULATE_GRAD_BATCHES,
                            deterministic=self.ml_hyperparameters.deterministic_algorithms,
                            log_every_n_steps=1,
                            num_sanity_val_steps=0,
                            #profiler="simple",
                            logger=logger,
                            callbacks=[
                                MetricsHistoryPlotCallback(),
                                LearningRateMonitor(logging_interval='epoch'),
                                ModelCheckpoint(dirpath=self.path_config.model_path,
                                                monitor="val_min_mae",
                                                mode="min",
                                                save_top_k=1,
                                                filename=model_name), #filename="model-{epoch:03d}-{val_loss:.4f}")
                                rolling_checkpoint,
                                StochasticWeightAveraging(swa_lrs=schedule_cfg["min_lr"],
                                                          swa_epoch_start=schedule_cfg["swa_epoch_start"],
                                                          annealing_epochs=1,
                                                          annealing_strategy="cos")
                            ])
        tuner = Tuner(trainer)

        #tuner.scale_batch_size(self.gixd_model, mode="binsearch")

        if False:
            lr_finder = tuner.lr_find(self.gixd_model, min_lr=1e-6, max_lr=1e-3,
                                      early_stop_threshold=None, num_training=70)
            print(lr_finder.results)
            fig = lr_finder.plot(suggest=True)
            #plt.ylim(0.0, 0.5)
            fig.savefig(f"{self.path_config.root_path}/lr_test.png")
            plt.close()

            new_lr = lr_finder.suggestion()

        #trainer.fit(model=self.gixd_model, ckpt_path=f"{self.path_config.model_path}/{model_name}.ckpt")
        # These are our own full training checkpoints, including NumPy RNG
        # state; restricted weights-only loading cannot restore that state.
        trainer.fit(model=self.gixd_model, ckpt_path=resume_checkpoint, weights_only=False)
        self._print_final_swa_batch_metrics()
        #trainer.save_checkpoint(f"{self.path_config.model_path}/{model_name}_swa.ckpt")
        #trainer.test(model=self.gixd_model, ckpt_path=f"{self.path_config.model_path}/{model_name}_swa.ckpt")

        if False:
            with torch.no_grad():
                (images, top_layer), cell_targets_norm = self.test_data_generators["uniform"].__getitem__(42)
                means, vars, weights = self.gixd_model(images)
                #pred = sample_from_gmm(means, vars, weights)
                for sidx in range(images.shape[0]):
                    print(f"True: {GixdUtility.denormalize_cell_deg(cell_targets_norm[sidx])}")
                    mean_sample = means.cpu().numpy()[sidx]
                    var_sample = vars.cpu().numpy()[sidx]
                    weight_sample = weights.cpu().numpy()[sidx]
                    for i in range(weight_sample.shape[0]):
                        print(f"   Weights = {weight_sample[i]}")
                        print(f"   Means: {GixdUtility.denormalize_cell_deg(mean_sample[i])}")
                        print(f"   Std = {GixdUtility.denormalize_cell_deg(np.sqrt(var_sample[i])+mean_sample[i])-GixdUtility.denormalize_cell_deg(mean_sample[i])}")
                #predictions = trainer.predict(gixself.gixd_model_model, return_predictions=True,
                #                              ckpt_path=f"{self.path_config.model_path}/swa_model_huber.ckpt")

    def predict_and_save_csv(self, num_samples=100, output_csv="predictions.csv", unknown_top_layer=False,
                             data_split="testing"):
        """
        Load the locally stored gixd_model checkpoint (using GixdModule) and evaluate it on the test data.
        
        The test data generator __getitem__ is assumed to return a tuple:
        ((images, true_top_layers), true_cell_targets, lattice_system)
        where lattice_system is the specific value for the sample.
        
        For each sample, the function:
        - Loads a batch from test_data_generators["uniform"]
        - Uses images and true_top_layers as provided (no additional permute is applied,
            since the forward method of GixdModule already handles the required format).
        - Computes the number of pixels with a value > 0 in each image.
        - Depending on unknown_top_layer:
            • If False, candidate_top_layers is set to the true top_layer (wrapped as a tensor of shape [1]).
            • If True, candidate_top_layers is set to all candidate indices from 0 to len(self.feature_config.miller_indices)-1.
        - The image is replicated for each candidate and the model (GixdModule) is run with the candidate
            top_layer indices.
        - For each candidate prediction, a row is saved containing:
                sample_index,
                true cell parameters,
                predicted cell parameters,
                lattice_system (the extra returned value),
                true top_layer,
                assumed candidate top_layer,
                and the nonzero pixel count.
                
        All results are accumulated and then stored in a CSV file.
        """
        if data_split not in {"validation", "testing"}:
            raise ValueError("data_split must be validation or testing")
        generators = (self.validation_data_generators if data_split == "validation"
                      else self.test_data_generators)
        if self.gixd_model is None:
            self._load_default_checkpoint()

        self.gixd_model.eval()
        generators["uniform"].batch_size = np.int32(np.clip(num_samples, 1, 32))
        generators["uniform"].epoch_size = num_samples
        dataloader = DataLoader(generators["uniform"], batch_size=1)
        results = []
        sample_index = 0

        # For unknown top_layer mode, prepare the full candidate list.
        full_candidate_top_layers = (
            torch.arange(len(self.feature_config.miller_indices)).to(self.device).long()
            if unknown_top_layer else None
        )

        

        with torch.no_grad():
            for batch in tqdm(dataloader, total=len(dataloader), desc="Processing Batches"):
                ((images, true_top_layers), true_targets, lattice_system, q_xy, q_z) = batch

                images = images.squeeze(0)
                lattice_system = lattice_system.squeeze(0)
                q_xy = q_xy.squeeze(0)
                q_z = q_z.squeeze(0)
                nonzero_counts = [(img > 0).sum().item() for img in images]
                true_top_layers = true_top_layers.squeeze(0).to(self.device).long()
                true_cells = GixdUtility.denormalize_cell_deg(true_targets.squeeze(0).cpu().numpy())
                for i in range(images.shape[0]):
                    image = images[i].unsqueeze(0)  # shape (1, H, W, C)
                    nonzero_count = nonzero_counts[i]
                    # Extract sample-specific q_xy and q_z arrays and convert them to JSON strings.
                    sample_qxy = q_xy[i].cpu().numpy()
                    sample_qz = q_z[i].cpu().numpy()
                    sample_qxy_json = json.dumps(sample_qxy.tolist())
                    sample_qz_json = json.dumps(sample_qz.tolist())
                    if unknown_top_layer:
                        candidate_top_layers = full_candidate_top_layers
                        num_candidates = candidate_top_layers.shape[0]
                    else:
                        candidate_top_layers = true_top_layers[i].unsqueeze(0)
                        num_candidates = 1
                    image_repeated = image.repeat(num_candidates, 1, 1, 1)
                    pred_cells = self._predict_cells_from_images(image_repeated, candidate_top_layers)
                    for j in range(num_candidates):
                        if True:#10 <= nonzero_count//self.feature_config.number_channels < 30:
                            results.append({
                                "sample_index": sample_index,
                                "data_split": data_split,
                                "true_a": true_cells[i, 0],
                                "true_b": true_cells[i, 1],
                                "true_c": true_cells[i, 2],
                                "true_alpha": true_cells[i, 3],
                                "true_beta": true_cells[i, 4],
                                "true_gamma": true_cells[i, 5],
                                "pred_a": pred_cells[j, 0],
                                "pred_b": pred_cells[j, 1],
                                "pred_c": pred_cells[j, 2],
                                "pred_alpha": pred_cells[j, 3],
                                "pred_beta": pred_cells[j, 4],
                                "pred_gamma": pred_cells[j, 5],
                                "lattice_system": int(lattice_system[i].item()),
                                "true_top_layer": int(true_top_layers[i].item()),
                                "assumed_top_layer": int(candidate_top_layers[j].item()),
                                "nonzero_pixel_count": nonzero_count // self.feature_config.number_channels,
                                "q_xy": sample_qxy_json,
                                "q_z": sample_qz_json
                            })
                    sample_index += 1
        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False)

    def analyze_predictions(self, csv_file, output_dir, font_size=18, 
                        horiz_line_dist=1.0, horiz_line_angle=1, 
                        percentiles=[50, 90, 95]):
        """
        Read a CSV file of predictions (with columns including true_a, true_b, true_c,
        and either pred_a, pred_b, ... or if available refined_a, refined_b, etc., and lattice_system),
        then for each lattice system:
        • Compute the MAE for distances and angles.
        • Create scatter plots for each coordinate (a, b, c and α, β, γ) using axis limits 
            from self.target_config.
        • Create a combined figure with two violin plots (one for distance errors and one for angle errors)
            on a log scale with y-axis limits [1e-7, 1e2]. A horizontal red line is drawn on each plot,
            with the value annotated.
        • Compute the requested percentiles for each lattice system and overall.
        All figures are saved to output_dir.
        Returns a dictionary mapping each lattice system (by name) to its MAE values and percentiles.
        """
        os.makedirs(output_dir, exist_ok=True)
        df = pd.read_csv(csv_file)

        # Use refined columns if available; otherwise use original predicted columns.
        refined_columns = [
            "refined_a", "refined_b", "refined_c",
            "refined_alpha", "refined_beta", "refined_gamma"
        ]
        use_refined = all(column_name in df.columns for column_name in refined_columns) and df["refined_a"].notna().any()
        pred_prefix = "refined" if use_refined else "pred"
        result_columns = [
            f"{pred_prefix}_a", f"{pred_prefix}_b", f"{pred_prefix}_c",
            f"{pred_prefix}_alpha", f"{pred_prefix}_beta", f"{pred_prefix}_gamma"
        ]

        print(f"Starting analysis for {pred_prefix} results")

        # Create a mapping from numeric value to lattice system name.
        lattice_map = {ls.value: ls.name for ls in LatticeSystem}
        groups = df.groupby("lattice_system")

        dist_errors_dict = {}
        angle_errors_dict = {}
        mean_errors = {}
        percentiles_dict = {}
        dist_max_errors_dict = {}
        angle_max_errors_dict = {}
        convergence_dict = {}

        tc = self.target_config
        dist_lim = (tc.min_distance, tc.max_distance)
        angle_lim = (tc.min_angle, tc.max_angle)  # assume angles in degrees

        for lattice, group in groups:
            # Get lattice name from numeric value if necessary.
            if isinstance(lattice, str):
                lattice_name = lattice.capitalize()
            else:
                lattice_name = lattice_map.get(int(lattice), str(lattice)).capitalize()

            valid_group = group.dropna(subset=result_columns).copy()
            if valid_group.empty:
                continue

            # Compute the per‐coordinate *mean* errors (unchanged).
            dist_error = (np.abs(valid_group["true_a"] - valid_group[f"{pred_prefix}_a"]) +
                          np.abs(valid_group["true_b"] - valid_group[f"{pred_prefix}_b"]) +
                          np.abs(valid_group["true_c"] - valid_group[f"{pred_prefix}_c"])) / 3.0
            angle_error = (np.abs(valid_group["true_alpha"] - valid_group[f"{pred_prefix}_alpha"]) +
                           np.abs(valid_group["true_beta"]  - valid_group[f"{pred_prefix}_beta"]) +
                           np.abs(valid_group["true_gamma"] - valid_group[f"{pred_prefix}_gamma"])) / 3.0

            dist_errs = dist_error.values
            angle_errs = angle_error.values

            # --- new: build arrays of the *max* absolute error across components ---
            # stack the three distance‐component errors, shape (N,3)
            dist_comp = np.vstack([
                np.abs(valid_group["true_a"].values      - valid_group[f"{pred_prefix}_a"].values),
                np.abs(valid_group["true_b"].values      - valid_group[f"{pred_prefix}_b"].values),
                np.abs(valid_group["true_c"].values      - valid_group[f"{pred_prefix}_c"].values),
            ]).T
            dist_max_errs = dist_comp.max(axis=1)

            # same for the three angles
            angle_comp = np.vstack([
                np.abs(valid_group["true_alpha"].values  - valid_group[f"{pred_prefix}_alpha"].values),
                np.abs(valid_group["true_beta"].values   - valid_group[f"{pred_prefix}_beta"].values),
                np.abs(valid_group["true_gamma"].values  - valid_group[f"{pred_prefix}_gamma"].values),
            ]).T
            angle_max_errs = angle_comp.max(axis=1)
            # ------------------------------------------------------------------------

            # store for violin plots (unchanged)
            dist_errors_dict[lattice_name]  = dist_errs
            angle_errors_dict[lattice_name] = angle_errs
            # store the *max*‐error arrays for later overall percentiles
            dist_max_errors_dict[lattice_name]  = dist_max_errs
            angle_max_errors_dict[lattice_name] = angle_max_errs

            mean_errors[lattice_name] = {
                "mae_distance": np.mean(dist_errs),
                "mae_angle":    np.mean(angle_errs)
            }
            convergence_dict[lattice_name] = self._calculate_convergence_stats_from_dataframe(
                valid_group,
                result_prefix=pred_prefix
            )

            # --- new: percentiles on the *max* error arrays ---
            percentiles_dict[lattice_name] = {
                "distance": np.percentile(dist_max_errs, percentiles),
                "angle":    np.percentile(angle_max_errs, percentiles)
            }

            # Scatter plots: one point per coordinate.
            plt.figure()
            plt.scatter(valid_group["true_a"], valid_group[f"{pred_prefix}_a"], alpha=0.5, label="a")
            plt.scatter(valid_group["true_b"], valid_group[f"{pred_prefix}_b"], alpha=0.5, label="b")
            plt.scatter(valid_group["true_c"], valid_group[f"{pred_prefix}_c"], alpha=0.5, label="c")
            plt.plot(dist_lim, dist_lim, 'r--')
            plt.xlabel("True Distance (Å)", fontsize=font_size)
            plt.ylabel("Predicted Distance (Å)", fontsize=font_size)
            plt.title(f"{lattice_name} - True vs. Predicted Distances", fontsize=font_size)
            plt.xlim(dist_lim)
            plt.ylim(dist_lim)
            plt.legend(fontsize=font_size)
            plt.savefig(os.path.join(output_dir, f"scatter_distance_{lattice_name}.png"))
            plt.close()

            plt.figure()
            plt.scatter(valid_group["true_alpha"], valid_group[f"{pred_prefix}_alpha"], alpha=0.5, label="α")
            plt.scatter(valid_group["true_beta"], valid_group[f"{pred_prefix}_beta"], alpha=0.5, label="β")
            plt.scatter(valid_group["true_gamma"], valid_group[f"{pred_prefix}_gamma"], alpha=0.5, label="γ")
            plt.plot(angle_lim, angle_lim, 'r--')
            plt.xlabel("True Angle (°)", fontsize=font_size)
            plt.ylabel("Predicted Angle (°)", fontsize=font_size)
            plt.title(f"{lattice_name} - True vs. Predicted Angles", fontsize=font_size)
            plt.xlim(angle_lim)
            plt.ylim(angle_lim)
            plt.legend(fontsize=font_size)
            plt.savefig(os.path.join(output_dir, f"scatter_angle_{lattice_name}.png"))
            plt.close()

        if not dist_max_errors_dict:
            return {
                "mean_errors": mean_errors,
                "percentiles": {"overall": {"distance": np.array([]), "angle": np.array([])}},
                "convergence": {
                    "overall": self._calculate_convergence_stats_from_dataframe(
                        df,
                        result_prefix=pred_prefix
                    )
                }
            }

        all_dist_max  = np.concatenate(list(dist_max_errors_dict.values()))
        all_angle_max = np.concatenate(list(angle_max_errors_dict.values()))
        overall_percentiles = {
            "distance": np.percentile(all_dist_max, percentiles),
            "angle":    np.percentile(all_angle_max, percentiles)
        }
        convergence_dict["overall"] = self._calculate_convergence_stats_from_dataframe(
            df,
            result_prefix=pred_prefix
        )

        # Prepare data for violin plots.
        dist_data = pd.DataFrame({
            "lattice_system": np.concatenate([np.repeat(l, len(errors)) for l, errors in dist_errors_dict.items()]),
            "error": np.concatenate(list(dist_errors_dict.values()))
        })
        angle_data = pd.DataFrame({
            "lattice_system": np.concatenate([np.repeat(l, len(errors)) for l, errors in angle_errors_dict.items()]),
            "error": np.concatenate(list(angle_errors_dict.values()))
        })

        # Combined violin figure.
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8))
        violin_dist = sns.violinplot(x="lattice_system", y="error", data=dist_data, ax=ax1,
                                    log_scale=True, color="aquamarine")
        violin_dist.set_yticks(np.logspace(-7, 1, 9))
        violin_dist.set_yticklabels(violin_dist.get_yticklabels(), size=15)
        xtick_labels = [str(txt.get_text()).capitalize() for txt in violin_dist.get_xticklabels()]
        violin_dist.set_xticks(violin_dist.get_xticks())
        violin_dist.set_xticklabels(xtick_labels, size=15)
        ax1.xaxis.tick_top()
        ax1.minorticks_off()
        ax1.tick_params(axis="x", top=True, bottom=False)
        ax1.set_ylim(1e-8, 1e1)
        ax1.set_title("Distance Errors", fontsize=font_size)
        ax1.set_xlabel("")
        ax1.set_ylabel("Error (Å)", fontsize=font_size)
        ax1.axhline(y=horiz_line_dist, color='red', linestyle='--')
        ax1.text(1.03, horiz_line_dist, f"{horiz_line_dist:.1f} Å", color='red', fontsize=font_size,
                transform=ax1.get_yaxis_transform(), va="bottom", ha="center")

        violin_angle = sns.violinplot(x="lattice_system", y="error", data=angle_data, ax=ax2,
                                    log_scale=True, color="cornflowerblue")
        violin_angle.set_yticks(np.logspace(-7, 2, 10))
        violin_angle.set_yticklabels(violin_angle.get_yticklabels(), size=15)
        ax2.set_xticks([])
        ax2.set_ylim(1e-8, 1e2)
        ax2.set_title("Angle Errors", fontsize=font_size)
        ax2.set_xlabel("Lattice System", fontsize=font_size)
        ax2.set_ylabel("Error (°)", fontsize=font_size)
        ax2.axhline(y=horiz_line_angle, color='red', linestyle='--')
        ax2.text(1.03, horiz_line_angle, f"{horiz_line_angle:.1f} °", color='red', fontsize=font_size,
                transform=ax2.get_yaxis_transform(), va="bottom", ha="center")

        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "violin_combined_errors.png"))
        plt.close(fig)

        percentiles_dict = {**percentiles_dict, "overall": overall_percentiles}
        return {
            "mean_errors": mean_errors,
            "percentiles": percentiles_dict,
            "convergence": convergence_dict
        }


    @staticmethod
    def write_analysis_report(report_dict, output_txt, percentile_labels=["50th", "90th", "95th"]):
        """
        Write a text report based on the analysis dictionary returned by analyze_predictions.
        The report includes mean absolute errors and the selected percentiles for each lattice system,
        as well as overall results.
        """
        convergence_dict = report_dict.get("convergence", {})
        overall_convergence = convergence_dict.get("overall")

        with open(output_txt, "w") as f:
            f.write("GIXD Predictions Analysis Report\n")
            f.write("=" * 40 + "\n\n")

            if overall_convergence is not None:
                f.write("Strict Convergence Thresholds:\n")
                f.write(
                    "  All distance errors < "
                    f"{overall_convergence['distance_tolerance']:.4e} Å and "
                    f"all angle errors < {overall_convergence['angle_tolerance']:.4e} °\n\n"
                )
            
            f.write("Mean Absolute Errors per Lattice System:\n")
            for lattice, errors in report_dict["mean_errors"].items():
                lattice_name = str(lattice).capitalize()
                f.write(f"Lattice System: {lattice_name}\n")
                f.write(f"  Mean Distance Error: {errors['mae_distance']:.4e} Å\n")
                f.write(f"  Mean Angle Error: {errors['mae_angle']:.4e} °\n")
                lattice_convergence = convergence_dict.get(lattice)
                if lattice_convergence is not None:
                    f.write(
                        "  Strict Convergence Rate: "
                        f"{100.0 * lattice_convergence['rate']:.2f}% "
                        f"({lattice_convergence['num_converged']}/{lattice_convergence['num_total']})\n"
                    )
                f.write("\n")
            
            f.write("Percentiles per Lattice System:\n")
            for lattice, perc in report_dict["percentiles"].items():
                if lattice != "overall":
                    lattice_name = str(lattice).capitalize()
                    f.write(f"Lattice System: {lattice_name}\n")
                    f.write("  Distance Errors Percentiles:\n")
                    for label, val in zip(percentile_labels, perc["distance"]):
                        f.write(f"    {label}: {val:.4e} Å\n")
                    f.write("  Angle Errors Percentiles:\n")
                    for label, val in zip(percentile_labels, perc["angle"]):
                        f.write(f"    {label}: {val:.4e} °\n")
                    f.write("\n")
                        
            overall = report_dict["percentiles"].get("overall")
            if overall is not None:
                f.write("Overall Percentiles:\n")
                f.write("  Distance Errors Percentiles:\n")
                for label, val in zip(percentile_labels, overall["distance"]):
                    f.write(f"    {label}: {val:.4e} Å\n")
                f.write("  Angle Errors Percentiles:\n")
                for label, val in zip(percentile_labels, overall["angle"]):
                    f.write(f"    {label}: {val:.4e} °\n")
            if overall_convergence is not None:
                f.write("\nOverall Strict Convergence:\n")
                f.write(
                    "  Strict Convergence Rate: "
                    f"{100.0 * overall_convergence['rate']:.2f}% "
                    f"({overall_convergence['num_converged']}/{overall_convergence['num_total']})\n"
                )
            
        print(f"Analysis report saved to: {output_txt}")

    def plot_saliency_maps(
        self,
        num_samples: int = 32,
        target_index: int = None,
        save_path: str = None
    ):
        """
        Sample a batch of `num_samples` test images, compute Integrated Gradients
        attributions for either a single output dimension (if `target_index` is set)
        or the L2‐norm of all outputs, and plot each original image alongside its
        saliency map.  Also prints and returns the per‐sample percentage of total
        attribution located in the first column of pixels.

        Args:
            num_samples: how many random test samples to visualize.
            target_index: if provided, compute IG w.r.t. a single output dim (0–5);
                          otherwise use the L2‐norm of the 6 outputs.
            save_path: if set, save the figure here instead of showing.
        Returns:
            percents: a 1D torch.Tensor of length `num_samples`, where each entry
                      is the percent of that sample’s total attribution in column 0.
        """
        if self.gixd_model is None:
            self._load_default_checkpoint()
        # 1) Fetch a random batch
        gen = self.test_data_generators["uniform"]
        gen.batch_size = num_samples
        (imgs_np, layers_np), _, _, _, _ = gen.__getitem__(0)
        imgs = torch.tensor(imgs_np, dtype=torch.float32).to(self.device)
        layers = torch.tensor(layers_np, dtype=torch.long).to(self.device)
        imgs.requires_grad_(True)

        # 2) Define model‐forward for captum
        def model_forward(x: torch.Tensor):
            out = self.gixd_model([x, layers])
            if isinstance(out, tuple):
                means, _, weights = out
                return (means * weights.unsqueeze(-1)).sum(dim=1)
            return out  # shape [B,6]

        # 3) Prepare baselines (all‐zeros)
        baseline = torch.zeros_like(imgs)

        # 4) Compute Integrated Gradients
        ig = IntegratedGradients(model_forward)
        if target_index is None:
            # attribute L2-norm of outputs
            def l2_norm(inp):
                return model_forward(inp).norm(dim=1)
            attributions = IntegratedGradients(l2_norm) \
                                .attribute(imgs, baselines=baseline)
        else:
            attributions = ig.attribute(
                imgs,
                baselines=baseline,
                target=target_index
            )

        # 5) Collapse channels → single saliency map per sample
        saliency = attributions.abs().max(dim=-1)[0].detach()  # [B,H,W]

        #print(f"sal shape = {saliency.shape}")

        # 6) Compute percentage in first column
        #    sum over rows of column 0, divided by sum over all pixels
        first_col = saliency[:, :, 0].view(num_samples, -1).sum(dim=1)
        total    = saliency.view(num_samples, -1).sum(dim=1)
        percents = (first_col / total * 100)

        avg_pct = percents.mean().item()
        print(f"Average % of IG‐attribution in first column: {avg_pct:.2f}%")

        # 7) Plot originals vs saliency
        fig, axes = plt.subplots(
            nrows=num_samples, ncols=2,
            figsize=(6, 3 * num_samples),
            squeeze=False
        )
        for i in range(num_samples):
            # original as grayscale mean
            img = imgs[i].detach().cpu().numpy()
            img_gray = img.mean(axis=-1)

            ax_img, ax_sal = axes[i]
            ax_img.imshow(img_gray, cmap='gray', interpolation="none")
            ax_img.axis('off')
            ax_img.set_title('Original')
            
            im_sal = ax_sal.imshow(saliency[i].cpu()/saliency[i].cpu().max(), cmap='jet', interpolation='none')
            ax_sal.axis('off')
            title = 'IG Saliency'
            if target_index is not None:
                title += f' (dim {target_index})'
            ax_sal.set_title(title)
            fig.colorbar(im_sal, ax=ax_sal, fraction=0.046,
                         pad=0.04)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
            plt.close(fig)
        else:
            plt.show()

        return percents, saliency[0].cpu()/saliency[i].cpu().max()
    
    def refine_predictions(self, input_csv, output_csv, num_processes=1, refinement_backend=None):
        """
        Process the input CSV file in batches. For each sample missing refined values,
        spawn up to num_processes processes that call direct_inversion (via global_refine_sample)
        and update the CSV with new refined columns, showing a progress bar.
        """
        def _to_numpy(arr):
            if torch.is_tensor(arr):
                return arr.detach().cpu().numpy()
            return np.asarray(arr)

        from gidind_ai.batched_refinement import safe_process_count
        num_processes = safe_process_count(self.device, num_processes)
        refinement_backend = refinement_backend or getattr(self, 'refinement_backend', 'auto')
        df = pd.read_csv(input_csv)

        # Create refined columns if they don't exist.
        for col in ["refined_a", "refined_b", "refined_c", "refined_alpha", "refined_beta", "refined_gamma"]:
            if col not in df.columns:
                df[col] = pd.NA

        # Identify rows missing refined values.
        df_missing = df[df["refined_a"].isna()].copy()
        sample_dicts = df_missing.to_dict(orient="records")

        # Partial function with extra args
        from functools import partial
        func = partial(
            global_refine_sample,
            data_generator=self.test_data_generators["uniform"],
            target_config=self.target_config,
            feature_config=self.feature_config,
            return_uncertainties=True,
            refinement_backend=refinement_backend
        )
    
        if num_processes == 1:
            refined_results = list(tqdm(map(func, sample_dicts), total=len(sample_dicts),
                                        desc="Refining samples", unit="sample"))
        else:
            with Pool(processes=num_processes) as pool:
                refined_results = list(tqdm(pool.imap(func, sample_dicts), total=len(sample_dicts),
                                            desc="Refining samples", unit="sample"))

        for i, sample in enumerate(sample_dicts):
            if refined_results[i] is not None:
                for key, value in refined_results[i].items():
                    if key.startswith("refined"):
                        df.loc[df["sample_index"] == sample["sample_index"], key] = value

        df.to_csv(output_csv, index=False)

        os.makedirs(self.path_config.prediction_sanity_check_path, exist_ok=True)
        for filename in os.listdir(self.path_config.prediction_sanity_check_path):
            if filename.startswith("plot_dict_") and filename.endswith(".png"):
                os.remove(os.path.join(self.path_config.prediction_sanity_check_path, filename))

        plotted_samples = 0
        for i, sample in enumerate(sample_dicts):
            if plotted_samples >= 16:
                break
            if refined_results[i] is None:
                continue

            init_plot_dict = refined_results[i]["debug_init_plot_dict"]
            refined_plot_dict = refined_results[i]["debug_refined_plot_dict"]

            fig, axs = plt.subplots(1, 2, figsize=(25, 14))
            fig.suptitle(f"Matching and Refinement Visualization (sample {sample['sample_index']})", fontsize=32)

            for ax, plot_dict, title in zip(
                axs,
                [init_plot_dict, refined_plot_dict],
                ["Neural Network Results", "Refinement Fit Results"]
            ):
                sim_q = _to_numpy(plot_dict["sim_q"])
                test_qxy = _to_numpy(plot_dict["test_qxy"])
                test_qz = _to_numpy(plot_dict["test_qz"])

                ax.scatter(
                    sim_q[:, 0],
                    sim_q[:, 1],
                    s=180,
                    facecolors='none',
                    edgecolors='g',
                    marker='o',
                    label="simulated"
                )
                ax.scatter(
                    test_qxy,
                    test_qz,
                    s=180,
                    c="r",
                    marker="x",
                    label="observed"
                )
                ax.legend(fontsize=14)
                ax.set_title(title, fontsize=32)
                ax.set_xlabel(r"$q_{xy}\,/\,\mathrm{Å}$", fontsize=32)
                ax.set_ylabel(r"$q_{z}\,/\,\mathrm{Å}$", fontsize=32)
                ax.set_xlim(0, self.feature_config.max_q_xy)
                ax.set_ylim(0, self.feature_config.max_q_z)
                ax.tick_params(axis='both', which='major', labelsize=18)

            plt.savefig(
                f"{self.path_config.prediction_sanity_check_path}/plot_dict_{plotted_samples:03d}.png"
            )
            plt.close(fig)
            plotted_samples += 1


    @staticmethod
    def _evaluate_sample(
        sample,
        hyperparams,
        device,
        feature_config,
        test_data_generator,
        target_config,
        refinement_hyperparameters,
        distance_tolerance=1e-3,
        angle_tolerance=1e-3
    ):
        """
        Helper function to evaluate a single sample with given hyperparameters.
        Returns 1 if the refined cell satisfies the strict convergence thresholds.
        """
        try:
            q_xy = torch.tensor(json.loads(sample["q_xy"]), dtype=torch.float64, device=device)
            q_z = torch.tensor(json.loads(sample["q_z"]), dtype=torch.float64, device=device)
        except Exception:
            print(f"Exception in setting q")
            return 0

        try:
            first_predicted_cell_deg = torch.tensor([
                float(sample["pred_a"]),
                float(sample["pred_b"]),
                float(sample["pred_c"]),
                float(sample["pred_alpha"]),
                float(sample["pred_beta"]),
                float(sample["pred_gamma"])
            ], dtype=torch.float64, device=device)
        except Exception:
            print(f"Exception in reading predictions")
            return 0

        true_cell_deg = [
            float(sample["true_a"]),
            float(sample["true_b"]),
            float(sample["true_c"]),
            float(sample["true_alpha"]),
            float(sample["true_beta"]),
            float(sample["true_gamma"])
        ]

        try:
            true_top_layer = int(sample["true_top_layer"])
            trial_top_layer = feature_config.miller_indices[true_top_layer]
        except Exception:
            trial_top_layer = feature_config.miller_indices[0]

        try:
            result = direct_inversion_pytorch(
                first_predicted_cell_deg=first_predicted_cell_deg,
                observed_qxy=q_xy,
                observed_qz=q_z,
                trial_top_layer=trial_top_layer,
                data_generator=test_data_generator,
                method=hyperparams["method"],
                loss_type=hyperparams["loss_type"],
                q_cutoff=hyperparams["q_cutoff"],
                maxiter=hyperparams["maxiter"],
                outer_loops=hyperparams["outer_loops"],
                matching_method=hyperparams['matching_method'],
                torch_lbfgs_lr=hyperparams.get("torch_lbfgs_lr", 1.0)
            )
        except Exception:
            return 0

        refined = result.get("refined_predicted_cell_deg")
        if refined is None:
            #print(f"No result returned")
            return 0

        if isinstance(refined, torch.Tensor):
            refined_np = refined.detach().cpu().numpy()
        else:
            refined_np = refined

        true_cell_np = np.array(true_cell_deg, dtype=np.float64)
        if (
            np.all(np.abs(true_cell_np[:3] - refined_np[:3]) < distance_tolerance)
            and np.all(np.abs(true_cell_np[3:] - refined_np[3:]) < angle_tolerance)
        ):
            return 1
        return 0

    def optimize_refinement_hyperparameters(
        self,
        input_csv: str = None,
        n_trials: int = 20,
        num_samples: int = 256,
        prediction_csv: str = None,
        num_processes: int = None,
        convergence_tolerance: float = None,
        distance_tolerance: float = 1e-3,
        angle_tolerance: float = 1e-3,
        sampler_seed: int = 42,
        method_choices = None,
        matching_method_choices = None,
        q_cutoff_min: float = 0.03,
        q_cutoff_max: float = 0.12,
        q_cutoff_log: bool = False,
        torch_lbfgs_lr_min: float = 0.2,
        torch_lbfgs_lr_max: float = 2.0,
        torch_lbfgs_lr_log: bool = True,
        maxiter: int = 500,
        outer_loops: int = 20,
        include_expensive_methods: bool = False,
        enable_pruning: bool = True,
        pruning_fraction: float = 0.2,
        pruning_interval_samples: int = 8,
        pruner_startup_trials: int = 1,
        max_in_flight: int = None
    ):
        """
        Optimize refinement hyperparameters with Optuna.

        If `input_csv` is omitted, a fresh predictions CSV is generated using
        `predict_and_save_csv(..., data_split="validation")`. The optimization
        target is the strict convergence rate: every refined distance error must
        be below `distance_tolerance` and every refined angle error must be below
        `angle_tolerance`. If `convergence_tolerance` is provided, it is used
        for both distance and angle thresholds.

        By default, the search excludes the slow exact second-order solvers
        (`trust-exact`, `newton-exact`) because they are much more likely to
        trigger very long sample evaluations and the repeated `torchmin`
        linear-algebra deprecation warnings seen in large tuning runs. Set
        `include_expensive_methods=True` or pass `method_choices` explicitly
        to include them again. The default tuning space is intentionally narrow:
        `method in {"bfgs", "cg"}`, `matching_method in {"hybrid", "one2one",
        "simulated"}`, fixed `loss_type="l2"`, and a modest `q_cutoff` range
        centered around the historically useful values. If `method_choices`
        includes `"torch-lbfgs"`, Optuna also tunes `torch_lbfgs_lr`.
        """
        if optuna is None:
            raise ImportError("optuna is not installed")

        import concurrent

        if convergence_tolerance is not None:
            distance_tolerance = float(convergence_tolerance)
            angle_tolerance = float(convergence_tolerance)

        if not 0.0 < pruning_fraction <= 1.0:
            raise ValueError("pruning_fraction must be in the interval (0, 1].")
        if pruning_interval_samples < 1:
            raise ValueError("pruning_interval_samples must be at least 1.")
        if pruner_startup_trials < 0:
            raise ValueError("pruner_startup_trials cannot be negative.")
        if q_cutoff_min <= 0.0 or q_cutoff_max <= 0.0:
            raise ValueError("q_cutoff_min and q_cutoff_max must be positive.")
        if q_cutoff_min >= q_cutoff_max:
            raise ValueError("q_cutoff_min must be smaller than q_cutoff_max.")
        if torch_lbfgs_lr_min <= 0.0 or torch_lbfgs_lr_max <= 0.0:
            raise ValueError("torch_lbfgs_lr_min and torch_lbfgs_lr_max must be positive.")
        if torch_lbfgs_lr_min >= torch_lbfgs_lr_max:
            raise ValueError("torch_lbfgs_lr_min must be smaller than torch_lbfgs_lr_max.")
        if maxiter < 1:
            raise ValueError("maxiter must be at least 1.")
        if outer_loops < 1:
            raise ValueError("outer_loops must be at least 1.")

        if input_csv is None:
            if prediction_csv is None:
                prediction_csv = os.path.join(
                    self.path_config.root_path,
                    f"optuna_refinement_predictions_{num_samples}.csv"
                )
            self.predict_and_save_csv(
                num_samples=num_samples,
                output_csv=prediction_csv,
                unknown_top_layer=False,
                data_split="validation"
            )
            input_csv = prediction_csv
        else:
            prediction_csv = input_csv

        df = pd.read_csv(input_csv)
        if "data_split" not in df or not df["data_split"].eq("validation").all():
            raise ValueError("Refinement tuning requires validation-only CSVs with data_split=validation; "
                             "legacy/unlabelled or test CSVs are not permitted.")
        samples = df.to_dict(orient="records")
        total_samples = len(samples)
        if total_samples == 0:
            raise ValueError(f"No samples found in {input_csv}")

        if num_processes is None:
            num_processes = max(1, cpu_count() - 1)

        if method_choices is None:
            method_choices = ["bfgs", "cg"]
            if include_expensive_methods:
                method_choices.extend(["newton-cg", "newton-exact", "trust-exact"])
        method_choices = list(dict.fromkeys(method_choices))
        if len(method_choices) == 0:
            raise ValueError("method_choices must contain at least one solver method.")

        if matching_method_choices is None:
            matching_method_choices = ["hybrid", "one2one", "simulated"]
        matching_method_choices = list(dict.fromkeys(matching_method_choices))
        if len(matching_method_choices) == 0:
            raise ValueError("matching_method_choices must contain at least one matcher.")

        if max_in_flight is None:
            max_in_flight = max(1, min(total_samples, num_processes, 8))
        else:
            max_in_flight = max(1, min(total_samples, int(max_in_flight)))

        trial_csv_path = os.path.join(self.path_config.root_path, "optuna_refinement_trials.csv")
        summary_txt_path = os.path.join(self.path_config.root_path, "optuna_refinement_summary.txt")
        prune_after_samples = max(1, int(np.ceil(pruning_fraction * total_samples)))
        use_sequential_cuda_eval = self.device.type == "cuda"
        evaluation_backend = "sequential_cuda" if use_sequential_cuda_eval else "process_pool"

        if use_sequential_cuda_eval:
            num_processes = 1
            max_in_flight = 1
            print(
                "[Optuna]: CUDA detected. Evaluating refinement trials serially in the main "
                "process to avoid multiprocessing CUDA OOM."
            )

        def _record_trial_user_attrs(
            trial: optuna.Trial,
            success_count: int,
            evaluated_count: int,
            failure_count: int,
            start_time_s: float,
            pruned: bool = False
        ):
            elapsed_s = time.perf_counter() - start_time_s
            denom = max(1, evaluated_count)
            trial.set_user_attr("time_s", elapsed_s)
            trial.set_user_attr("time_per_sample_s", elapsed_s / denom)
            trial.set_user_attr("convergence_rate", success_count / denom)
            trial.set_user_attr("num_converged", success_count)
            trial.set_user_attr("num_evaluated", evaluated_count)
            trial.set_user_attr("num_total", total_samples)
            trial.set_user_attr("num_failures", failure_count)
            trial.set_user_attr("pruned", pruned)

        def objective(trial: optuna.Trial):
            t0 = time.perf_counter()
            hyperparams = {
                "method": trial.suggest_categorical(
                    "method",
                    method_choices
                ),
                "loss_type": "l2",
                "q_cutoff": trial.suggest_float(
                    "q_cutoff",
                    q_cutoff_min,
                    q_cutoff_max,
                    log=q_cutoff_log
                ),
                "maxiter": int(maxiter),
                "outer_loops": int(outer_loops),
                'matching_method': trial.suggest_categorical(
                    'matching_method',
                    matching_method_choices
                )
            }
            if hyperparams["method"] == "torch-lbfgs":
                hyperparams["torch_lbfgs_lr"] = trial.suggest_float(
                    "torch_lbfgs_lr",
                    torch_lbfgs_lr_min,
                    torch_lbfgs_lr_max,
                    log=torch_lbfgs_lr_log
                )
            else:
                hyperparams["torch_lbfgs_lr"] = 1.0

            success_count = 0
            trial_count = 0
            failure_count = 0
            if use_sequential_cuda_eval:
                pbar = tqdm(total=total_samples, desc=f"Optuna trial {trial.number}", leave=False)
                try:
                    for sample in samples:
                        try:
                            res = GixdRegressor._evaluate_sample(
                                sample,
                                hyperparams,
                                self.device,
                                self.feature_config,
                                self.validation_data_generators["uniform"],
                                self.target_config,
                                self.refinement_hyperparameters,
                                distance_tolerance,
                                angle_tolerance
                            )
                        except Exception:
                            res = 0
                            failure_count += 1

                        success_count += res
                        trial_count += 1
                        running_rate = success_count / trial_count
                        trial.report(running_rate, step=trial_count)
                        pbar.set_postfix(
                            convergence_rate=f"{running_rate:.3f}",
                            failures=failure_count
                        )
                        pbar.update(1)

                        prune_reason = None
                        if enable_pruning and trial_count >= prune_after_samples:
                            if success_count == 0:
                                prune_reason = (
                                    f"Pruned after {trial_count}/{total_samples} samples "
                                    f"with zero converged refinements."
                                )
                            elif (
                                (trial_count - prune_after_samples) % pruning_interval_samples == 0
                                and trial.should_prune()
                            ):
                                prune_reason = (
                                    f"Optuna pruner stopped trial at {trial_count}/{total_samples} "
                                    f"samples (running convergence {running_rate:.4f})."
                                )

                        if prune_reason is not None:
                            _record_trial_user_attrs(
                                trial,
                                success_count=success_count,
                                evaluated_count=trial_count,
                                failure_count=failure_count,
                                start_time_s=t0,
                                pruned=True
                            )
                            raise optuna.TrialPruned(prune_reason)

                        torch.cuda.empty_cache()
                finally:
                    pbar.close()
            else:
                executor = concurrent.futures.ProcessPoolExecutor(max_workers=num_processes)
                pbar = tqdm(total=total_samples, desc=f"Optuna trial {trial.number}", leave=False)
                pruned = False

                try:
                    sample_iter = iter(samples)
                    in_flight = set()

                    def _submit_next():
                        try:
                            sample = next(sample_iter)
                        except StopIteration:
                            return False
                        future = executor.submit(
                            GixdRegressor._evaluate_sample,
                            sample,
                            hyperparams,
                            self.device,
                            self.feature_config,
                            self.validation_data_generators["uniform"],
                            self.target_config,
                            self.refinement_hyperparameters,
                            distance_tolerance,
                            angle_tolerance
                        )
                        in_flight.add(future)
                        return True

                    for _ in range(min(max_in_flight, total_samples)):
                        if not _submit_next():
                            break

                    while in_flight:
                        done, _ = concurrent.futures.wait(
                            in_flight,
                            return_when=concurrent.futures.FIRST_COMPLETED
                        )
                        for future in done:
                            in_flight.remove(future)
                            try:
                                res = future.result()
                            except Exception:
                                res = 0
                                failure_count += 1

                            success_count += res
                            trial_count += 1
                            running_rate = success_count / trial_count
                            trial.report(running_rate, step=trial_count)
                            pbar.set_postfix(
                                convergence_rate=f"{running_rate:.3f}",
                                failures=failure_count
                            )
                            pbar.update(1)

                            prune_reason = None
                            if enable_pruning and trial_count >= prune_after_samples:
                                if success_count == 0:
                                    prune_reason = (
                                        f"Pruned after {trial_count}/{total_samples} samples "
                                        f"with zero converged refinements."
                                    )
                                elif (
                                    (trial_count - prune_after_samples) % pruning_interval_samples == 0
                                    and trial.should_prune()
                                ):
                                    prune_reason = (
                                        f"Optuna pruner stopped trial at {trial_count}/{total_samples} "
                                        f"samples (running convergence {running_rate:.4f})."
                                    )

                            if prune_reason is not None:
                                pruned = True
                                _record_trial_user_attrs(
                                    trial,
                                    success_count=success_count,
                                    evaluated_count=trial_count,
                                    failure_count=failure_count,
                                    start_time_s=t0,
                                    pruned=True
                                )
                                raise optuna.TrialPruned(prune_reason)

                            _submit_next()
                finally:
                    pbar.close()
                    executor.shutdown(wait=not pruned, cancel_futures=pruned)

            convergence_rate = success_count / total_samples
            _record_trial_user_attrs(
                trial,
                success_count=success_count,
                evaluated_count=trial_count,
                failure_count=failure_count,
                start_time_s=t0,
                pruned=False
            )
            return convergence_rate

        if enable_pruning:
            pruner = optuna.pruners.MedianPruner(
                n_startup_trials=pruner_startup_trials,
                n_warmup_steps=prune_after_samples,
                interval_steps=pruning_interval_samples
            )
        else:
            pruner = optuna.pruners.NopPruner()

        model_restore_device = None
        if use_sequential_cuda_eval and self.gixd_model is not None:
            model_restore_device = self.device
            self.gixd_model.to("cpu")
            torch.cuda.empty_cache()

        try:
            study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=sampler_seed, multivariate=True),
                pruner=pruner
            )
            study.optimize(objective, n_trials=n_trials)
        finally:
            if model_restore_device is not None:
                self.gixd_model.to(model_restore_device)
                torch.cuda.empty_cache()

        study.trials_dataframe().to_csv(trial_csv_path, index=False)

        with open(summary_txt_path, "w") as f:
            print("Refinement Hyperparameter Optimization", file=f)
            print("=" * 40, file=f)
            print(f"input_csv: {input_csv}", file=f)
            print(f"num_samples: {total_samples}", file=f)
            print(f"distance_tolerance: {distance_tolerance:.4e}", file=f)
            print(f"angle_tolerance: {angle_tolerance:.4e}", file=f)
            print(f"evaluation_backend: {evaluation_backend}", file=f)
            print(f"method_choices: {method_choices}", file=f)
            print(f"matching_method_choices: {matching_method_choices}", file=f)
            print(f"fixed_loss_type: l2", file=f)
            print(f"fixed_maxiter: {maxiter}", file=f)
            print(f"fixed_outer_loops: {outer_loops}", file=f)
            print(
                f"q_cutoff_range: [{q_cutoff_min:.6f}, {q_cutoff_max:.6f}] "
                f"(log_scale={q_cutoff_log})",
                file=f
            )
            print(
                f"torch_lbfgs_lr_range: [{torch_lbfgs_lr_min:.6f}, {torch_lbfgs_lr_max:.6f}] "
                f"(log_scale={torch_lbfgs_lr_log})",
                file=f
            )
            print(f"enable_pruning: {enable_pruning}", file=f)
            print(f"pruning_fraction: {pruning_fraction:.3f}", file=f)
            print(f"pruning_interval_samples: {pruning_interval_samples}", file=f)
            print(f"prune_after_samples: {prune_after_samples}", file=f)
            print(f"max_in_flight: {max_in_flight}", file=f)
            print("", file=f)
            print("=== All trials ===", file=f)
            for t in study.trials:
                time_per_sample = t.user_attrs.get("time_per_sample_s", float("nan"))
                convergence_rate = t.user_attrs.get("convergence_rate", 0.0)
                num_converged = t.user_attrs.get("num_converged", 0)
                num_evaluated = t.user_attrs.get("num_evaluated", t.user_attrs.get("num_total", total_samples))
                num_total = t.user_attrs.get("num_total", total_samples)
                num_failures = t.user_attrs.get("num_failures", 0)
                method = t.params.get("method")
                trial_outer_loops = t.params.get("outer_loops", outer_loops)
                loss_type = t.params.get("loss_type", "l2")
                matching_method = t.params.get("matching_method")
                q_cutoff = t.params.get("q_cutoff")
                torch_lbfgs_lr = t.params.get("torch_lbfgs_lr", 1.0)
                trial_maxiter = t.params.get("maxiter", maxiter)
                state = t.state.name
                rate_label = "partial_conv" if state == "PRUNED" else "conv"
                print(
                    f"Trial {t.number:2d} [{state:8s}]: method={method} maxiter={trial_maxiter} loops={trial_outer_loops} "
                    f"{loss_type}-loss {matching_method}-matched cutoff={q_cutoff:.6f} "
                    f"torch_lbfgs_lr={torch_lbfgs_lr:.6f} "
                    f"{rate_label}={100.0 * convergence_rate:7.5f}% ({num_converged}/{num_evaluated} evaluated, "
                    f"target={num_total}) "
                    f"failures={num_failures} time={time_per_sample:4.2f}s/sample",
                    file=f
                )

            completed_trials = [
                t for t in study.trials
                if t.state == optuna.trial.TrialState.COMPLETE
            ]
            if completed_trials:
                best_params = study.best_trial.params
                best_score = study.best_trial.value
            else:
                best_params = None
                best_score = None
            print("", file=f)
            print("Best hyperparameters:", best_params, file=f)
            print("Best convergence rate:", best_score, file=f)

        completed_trials = [
            t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ]
        if completed_trials:
            best_params = study.best_trial.params
            best_score = study.best_trial.value
            print("Best hyperparameters:", best_params)
            print("Best convergence rate:", best_score)
        else:
            best_params = None
            best_score = None
            print("No Optuna trial completed without pruning.")


        return {
            "best_params": best_params,
            "best_score": best_score,
            "input_csv": input_csv,
            "trials_csv": trial_csv_path,
            "summary_txt": summary_txt_path,
            "study": study
        }

    def predict_and_refine_unknown_benchmark(
        self,
        num_samples,
        output_csv,
        num_processes=None,
        q_cutoff_1 = 0.066,
        q_cutoff_2 = 0.066,
        probe_iterations = 10,
        q_loss_threshold = 1e-5,
        percent_matched_threshold = 0.95,
        save_result_report = False,
        save_result_reconstruction = False,
        refinement_backend = None,
        data_split = "testing"
    ):
        if data_split not in {"validation", "testing"}:
            raise ValueError("data_split must be validation or testing")
        generators = (self.validation_data_generators if data_split == "validation"
                      else self.test_data_generators)
        # ensure model loaded
        if self.gixd_model is None:
            self._load_default_checkpoint()

        #generators["uniform"].batch_size = 32

        self.gixd_model.eval()
        generators["uniform"].epoch_size = num_samples
        generators["uniform"].batch_size = num_samples
        loader = DataLoader(
            generators["uniform"],
            batch_size=None,
        )

        samples = []
        idx = 0
        num_orients = len(self.feature_config.miller_indices)

        with torch.no_grad():
            for ((images, top_layers), cell_targets, lattice_systems, q_xy, q_z) in loader:
                B = images.shape[0]
                images   = images.to(self.device)
                q_xy     = q_xy.to(self.device)
                q_z      = q_z.to(self.device)
                top_layers = top_layers.to(self.device)

                for b in range(B):
                    img_rep = images[b].unsqueeze(0) \
                                    .repeat(num_orients, 1, 1, 1)
                    top_idx = torch.arange(
                        num_orients,
                        device=self.device,
                        dtype=torch.int64
                    )
                    q_xy_b, q_z_b = GixdDataGenerator.filter_q_points(q_xy[b], q_z[b], 1e-6)
                    out = self.gixd_model([img_rep, top_idx]).cpu()
                    if isinstance(out, tuple):
                        means, _, weights = out
                        pred_norm_all = (means * weights.unsqueeze(-1)).sum(1)
                    else:
                        pred_norm_all = out
                    
                    pred_cell_deg = GixdUtility.denormalize_cell_deg(pred_norm_all) \
                                                .cpu().tolist()

                    samples.append({
                        'sample_index':   idx,
                        'pred_norm':      pred_norm_all.tolist(),
                        'pred_cell_deg':  pred_cell_deg,
                        'q_xy':           q_xy_b.cpu().numpy(),
                        'q_z':            q_z_b.cpu().numpy(),
                        'true_top_layer': int(top_layers[b].item()),
                        'true_cell_deg':  GixdUtility.denormalize_cell_deg(cell_targets[b]).cpu().tolist()
                    })
                    idx += 1

        import multiprocessing as mp
        from functools import partial

        from gidind_ai.batched_refinement import safe_process_count
        num_processes = safe_process_count(self.device, num_processes)
        pool = mp.Pool(processes=num_processes) if num_processes > 1 else None
        func = partial(
            _unknown_refine_worker,
            data_generator            = generators["uniform"],
            feature_config            = self.feature_config,
            device                    = self.device,
            q_cutoff_1                = q_cutoff_1,
            q_cutoff_2                = q_cutoff_2,
            probe_iterations          = probe_iterations,
            q_loss_threshold          = q_loss_threshold,
            percent_matched_threshold = percent_matched_threshold,
            return_only_converged     = False,
            return_debug_output       = save_result_reconstruction,
            refinement_backend        = refinement_backend or getattr(self, 'refinement_backend', 'auto')
        )

        results = []
        if save_result_report or save_result_reconstruction:
            if not os.path.exists(f"{self.path_config.root_path}/unknown_sanity_check"):
                os.mkdir(f"{self.path_config.root_path}/unknown_sanity_check")
        result_iter = pool.imap(func, samples) if pool is not None else map(func, samples)
        for sidx, res in enumerate(tqdm(result_iter, total=len(samples), desc="Refining unknown top layers")):
            #for res in pool.imap(func, samples):
            results.append(res)
            if save_result_report or save_result_reconstruction:
                if not os.path.exists(f"{self.path_config.root_path}/unknown_sanity_check/sample_{sidx:03d}"):
                    os.mkdir(f"{self.path_config.root_path}/unknown_sanity_check/sample_{sidx:03d}")
            if save_result_report:
                with open(f"{self.path_config.root_path}/unknown_sanity_check/sample_{sidx:03d}/report.txt","w") as fh:
                    print(f'True hkl = {res["true_hkl"]}', file=fh)
                    print(f'True unit cell = [{res["true_a"]},{res["true_b"]},{res["true_c"]},{res["true_alpha"]},{res["true_beta"]},{res["true_gamma"]}',file=fh)
                    #print(f"res = {res}")
                    for i in range(len(self.feature_config.miller_indices)):
                        print(f"Try hkl = {res[f'hkl_{i}']}", file=fh)
                        print(f"Refined unit cell = [{res[f'refined{i}_a']},{res[f'refined{i}_b']},"
                            f"{res[f'refined{i}_c']},{res[f'refined{i}_alpha']},{res[f'refined{i}_beta']},"
                            f"{res[f'refined{i}_gamma']}", file=fh)
                        print(f"Uncertainty unit cell = [{res[f'unc{i}_a']},{res[f'unc{i}_b']},"
                            f"{res[f'unc{i}_c']},{res[f'unc{i}_alpha']},{res[f'unc{i}_beta']},"
                            f"{res[f'unc{i}_gamma']}", file=fh) 
                        print(f"Is converged = {res[f'is_converged{i}']}", file=fh)
                        print(f"q loss = {res[f'q_loss{i}']}", file=fh)
                        print(f"Percent matched = {res[f'percent_matched{i}']}\n", file=fh)                 
            if save_result_reconstruction:      
                for i in range(len(self.feature_config.miller_indices)):

                    fig, axs = plt.subplots(1,2, figsize=(24, 14))
                    #refined_plot_dict = res[f"refined_plot_dict{i}"]
                    # right subplot: refined result
                    for idx, ax in enumerate(axs):
                        if idx==0: plot_dict = res[f"refined_plot_dict{i}"]
                        else: plot_dict = res[f"init_plot_dict{i}"]
                        ax.scatter(
                            plot_dict["sim_q"][:, 0],
                            plot_dict["sim_q"][:, 1],
                            s=180,
                            facecolors='none',
                            edgecolors='g',
                            marker='o',
                            label="simulated"
                        )
                        ax.scatter(
                            plot_dict["test_qxy"],
                            plot_dict["test_qz"],
                            s=180,
                            c="r",
                            marker="x",
                            label="observed"
                        )
                        ax.legend(fontsize=14)
                        # subplot title with unit cell and loss
                        ax.set_title(
                            "Refinement Fit Results" if idx==0 else "Initial Reconstruction",
                            fontsize=32
                        )
                        ax.set_xlabel(r"$q_{xy}\,/\,\mathrm{Å}$", fontsize=32)
                        ax.set_ylabel(r"$q_{z}$", fontsize=32)
                        ax.set_xlim(0, 2)
                        ax.set_ylim(0, 2)
                        # axs[1].grid(False)
                        ax.tick_params(axis='both', which='major', labelsize=18)
                    plt.savefig(f"{self.path_config.root_path}/unknown_sanity_check/sample_{sidx:03d}/reconstruction_{i}.png")
                    plt.close()
                    

        if pool is not None:
            pool.close()
            pool.join()

        df = pd.DataFrame(results)
        df["data_split"] = data_split
        df.to_csv(output_csv, index=False)
        acc = df['true_hkl_included'].mean()
        avg_num_hkl = df['number_of_hkl'].mean()
        print(f"Saved unknown-HKL results (with true labels) to {output_csv}")
        print(f"accuracy: {acc:.4f} average number: {avg_num_hkl}")
        return acc, avg_num_hkl
    
    def analyze_real_sample(self, sample_data_path:str, save_analysis_to:str,
                             q_scale=None, save_result_report=True,
                             save_result_reconstruction=True, num_processes=None,
                             refinement_backend=None):
        from gidind_ai.batched_refinement import safe_process_count
        num_processes = safe_process_count(self.device, num_processes)
        refinement_backend = refinement_backend or getattr(self, 'refinement_backend', 'auto')
        from load_peaks import load_peaks_qxy_qz
        #if sample_data_path == "mof":
        #    q_xy_test = np.array([0.578,0.817,1.157,0.581,0.822,1.16,
        #                          1.290,1.734, 1.295, 1.735])/q_scale
        #    q_z_test = np.array([0.0,0.0,0.0,0.651,0.65,0.65,
        #                         0.0, 0.0, 0.655, 0.649])/q_scale

        q_xy_test, q_z_test, q_scale_used = load_peaks_qxy_qz(
            sample_data=sample_data_path,
            feature_config=self.feature_config,
            q_scale=q_scale,
            max_fill=0.97
        )
        # Keep continuity with existing code paths that might read q_scale afterward
        q_scale = q_scale_used
        print(f"Using q_scale = {q_scale}")

        #filter values:
        q_is_not_too_large = np.logical_not(np.logical_or(q_xy_test >= self.feature_config.max_q_xy,
                                    q_z_test >= self.feature_config.max_q_z))
        q_xy_test = q_xy_test[q_is_not_too_large]
        q_z_test = q_z_test[q_is_not_too_large]
        #remove specular scan
        if not self.feature_config.use_specular_scan: 
            q_is_not_too_specular = np.logical_not(q_xy_test < self.feature_config.specular_cutoff)
            q_xy_test = q_xy_test[q_is_not_too_specular]
            q_z_test = q_z_test[q_is_not_too_specular]
        else:
            q_is_not_direct_beam = np.logical_not(np.logical_or(q_xy_test > 0.01, q_z_test > 0.01))
            q_xy_test = q_xy_test[q_is_not_direct_beam]
            q_z_test = q_z_test[q_is_not_direct_beam]

        #for idx in range(len(q_xy_test)): print(f"[{q_xy_test[idx]} | {q_z_test[idx]}]")

        if self.gixd_model is None:
            self.predict_and_save_csv(num_samples=0, unknown_top_layer=True)

        q_xy_test = torch.from_numpy(q_xy_test).unsqueeze(0).to(torch.float64)
        q_z_test = torch.from_numpy(q_z_test).unsqueeze(0).to(torch.float64)

        self.gixd_model.eval()

        num_orients = len(self.feature_config.miller_indices)
        images = self.test_data_generators["uniform"].calc_images_from_q_points(q_xy_test, q_z_test)

        with torch.no_grad():
            images   = images.to(self.device)

            img_rep = images.repeat(num_orients, 1, 1, 1)
            top_idx = torch.arange(
                num_orients,
                device=self.device,
                dtype=torch.int64
            )
            out = self.gixd_model([img_rep, top_idx]).cpu()
            if isinstance(out, tuple):
                means, _, weights = out
                pred_norm_all = (means * weights.unsqueeze(-1)).sum(1)
            else:
                pred_norm_all = out
            
            pred_cell_deg = GixdUtility.denormalize_cell_deg(pred_norm_all) \
                                        .cpu().tolist()

            sample_input = {
                'sample_index':   0,
                'pred_norm':      pred_norm_all.tolist(),
                'pred_cell_deg':  pred_cell_deg,
                'q_xy':           q_xy_test.cpu().numpy(),
                'q_z':            q_z_test.cpu().numpy()
            }

        import multiprocessing as mp
        from functools import partial

        results = []

        if save_result_report or save_result_reconstruction:
            if not os.path.exists(save_analysis_to):
                os.mkdir(save_analysis_to)

        # Handle multiprocessing - use Pool only if num_processes > 1
        # to avoid pickling issues with PyTorch objects
        if num_processes is not None and num_processes > 1:
            pool = mp.Pool(processes=num_processes)
            func = partial(
                _unknown_refine_worker,
                data_generator            = self.test_data_generators["uniform"],
                feature_config            = self.feature_config,
                device                    = self.device,
                q_cutoff_1                = 0.06,
                q_cutoff_2                = 0.06,
                matching_method           = 'simulated_fractional',
                probe_iterations          = 20,
                q_loss_threshold          = 1e-5,
                percent_matched_threshold = 0.90,
                return_only_converged     = False,
                return_debug_output       = save_result_reconstruction,
                refinement_backend        = refinement_backend
            )
            result_iter = pool.imap(func, [sample_input])
        else:
            # Single process - call worker directly
            result_iter = [_unknown_refine_worker(
                sample_input,
                data_generator            = self.test_data_generators["uniform"],
                feature_config            = self.feature_config,
                device                    = self.device,
                q_cutoff_1                = 0.06,
                q_cutoff_2                = 0.06,
                matching_method           = 'simulated_fractional',
                probe_iterations          = 20,
                q_loss_threshold          = 1e-5,
                percent_matched_threshold = 0.90,
                return_only_converged     = False,
                return_debug_output       = save_result_reconstruction,
                refinement_backend        = refinement_backend
            )]

        for sidx, res in enumerate(tqdm(result_iter, total=len([sample_input]), desc="Refining unknown top layers")):
            results.append(res)
            if save_result_report:
                with open(f"{save_analysis_to}/report.txt","w") as fh:
                    for i in range(len(self.feature_config.miller_indices)):
                        print(f"Try hkl = {res[f'hkl_{i}']}", file=fh)
                        print(f"NN unit cell = {sample_input['pred_cell_deg'][i]/np.array(3*[q_scale]+3*[1.0])}", file=fh)
                        print(f"Refined unit cell = [{res[f'refined{i}_a']/q_scale},{res[f'refined{i}_b']/q_scale},"
                            f"{res[f'refined{i}_c']/q_scale},{res[f'refined{i}_alpha']},{res[f'refined{i}_beta']},"
                            f"{res[f'refined{i}_gamma']}]", file=fh)
                        print(f"Uncertainty unit cell = [{res[f'unc{i}_a']/q_scale},{res[f'unc{i}_b']/q_scale},"
                            f"{res[f'unc{i}_c']/q_scale},{res[f'unc{i}_alpha']},{res[f'unc{i}_beta']},"
                            f"{res[f'unc{i}_gamma']}]", file=fh) 
                        print(f"Is converged = {res[f'is_converged{i}']}", file=fh)
                        print(f"q loss = {res[f'q_loss{i}']}", file=fh)
                        print(f"Percent matched = {res[f'percent_matched{i}']}\n", file=fh)                 
            if save_result_reconstruction:      
                for i in range(len(self.feature_config.miller_indices)):
                    fig, axs = plt.subplots(1,3, figsize=(3*6*self.feature_config.max_q_xy, 6*self.feature_config.max_q_z))
                    
                    # right subplot: refined result
                    for idx, ax in enumerate(axs):
                        if idx==0:
                            plot_dict = res[f"init_plot_dict{i}"]
                        else:
                            plot_dict = res[f"refined_plot_dict{i}"]
                        if idx < 2:
                            ax.scatter(
                                plot_dict["sim_q"][:, 0]*q_scale,
                                plot_dict["sim_q"][:, 1]*q_scale,
                                s=180,
                                facecolors='none',
                                edgecolors='g',
                                marker='o',
                                label="simulated"
                            )
                            ax.scatter(
                                plot_dict["test_qxy"]*q_scale,
                                plot_dict["test_qz"]*q_scale,
                                s=180,
                                c="r",
                                marker="x",
                                label="observed"
                            )
                        
                            ax.legend(fontsize=14)
                            #ax.set_title("Refinement Fit Results",fontsize=32)
                            ax.set_xlabel(r"$q_{xy}\,/\,\mathrm{Å}$", fontsize=32)
                            ax.set_ylabel(r"$q_{z}$", fontsize=32)
                            ax.set_xlim(0, self.feature_config.max_q_xy*q_scale)
                            ax.set_ylim(0, self.feature_config.max_q_z*q_scale)
                        else:
                            ax.imshow(images[0,:,:,0])
                        # axs[1].grid(False)
                        ax.tick_params(axis='both', which='major', labelsize=18)
                    plt.savefig(f"{save_analysis_to}/reconstruction_{i}.png")
                    plt.close()
                    
        # Only close and join pool if it was created
        if num_processes is not None and num_processes > 1:
            pool.close()
            pool.join()
        return res
    
    def analyze_real_sample_partial_phase(self, sample_data_path:str, save_analysis_to:str,
                             q_scale=1.0, save_result_report=True,
                             save_result_reconstruction=True, num_processes=None):
        if self.device.type == 'cuda':
            raise ValueError("Legacy partial/multiphase refinement uses process pools; run it on CPU")
        #FIT FIRST PHASE AGAIN:
        first_phase_match_qxy = None
        first_phase_match_qz = None
        peak_list_df = pd.read_csv(sample_data_path)
        q_xy_test = peak_list_df["q_xy_list"].apply(ast.literal_eval).apply(np.array)[0]/q_scale
        q_z_test = peak_list_df["q_z_list"].apply(ast.literal_eval).apply(np.array)[0]/q_scale
      
        init_cell_deg_1 = np.array([5.919275484181106,7.539978733260092,15.540952523181643,
                                    84.1461516402646,87.32325720814515,90.04981154592718])
        init_cell_deg_2 = np.array([6.070807587802225,7.887053716654455,14.879115469137323,
                                    96.71017379835584,100.51406772791033,94.1703447893983])
        
        """
        7.900000095367432,6.059999942779541,16.010000228881836,
        101.9000015258789,112.5999984741211,85.80000305175781,

        5.920000076293945,7.539999961853027,15.630000114440918,
        81.5,87.19999694824219,89.9000015258789"""

        #filter values:
        q_is_not_too_large = np.logical_not(np.logical_or(q_xy_test > self.feature_config.max_q_xy,
                                    q_z_test > self.feature_config.max_q_z))
        q_xy_test = q_xy_test[q_is_not_too_large]
        q_z_test = q_z_test[q_is_not_too_large]
        #remove specular scan
        if not self.feature_config.use_specular_scan: 
            q_is_not_too_specular = np.logical_not(q_xy_test < self.feature_config.specular_cutoff)
            q_xy_test = q_xy_test[q_is_not_too_specular]
            q_z_test = q_z_test[q_is_not_too_specular]
        else:
            q_is_not_direct_beam = np.logical_not(np.logical_or(q_xy_test > 0.01, q_z_test > 0.01))
            q_xy_test = q_xy_test[q_is_not_direct_beam]
            q_z_test = q_z_test[q_is_not_direct_beam]

        if self.gixd_model is None:
            self.predict_and_save_csv(num_samples=0, unknown_top_layer=True)

        q_xy_test = torch.from_numpy(q_xy_test).unsqueeze(0).to(torch.float64)
        q_z_test = torch.from_numpy(q_z_test).unsqueeze(0).to(torch.float64)

        self.gixd_model.eval()

        num_orients = len(self.feature_config.miller_indices)
        images = self.test_data_generators["uniform"].calc_images_from_q_points(q_xy_test, q_z_test)

        with torch.no_grad():
            pred_cell_deg = torch.from_numpy(np.repeat(np.expand_dims(init_cell_deg_2,0),
                                                       repeats=num_orients, axis=0)).to(images.dtype).tolist()
            pred_norm_all = torch.from_numpy(np.repeat(np.expand_dims(GixdUtility.normalize_cell_deg(init_cell_deg_2),0),
                                                       repeats=num_orients, axis=0)).to(images.dtype).tolist()
           
            sample_input = {
                'sample_index':   0,
                'pred_norm':      pred_norm_all,
                'pred_cell_deg':  pred_cell_deg,
                'q_xy':           q_xy_test.cpu().numpy(),
                'q_z':            q_z_test.cpu().numpy()
            }

        import multiprocessing as mp
        from functools import partial

        pool = mp.Pool(processes=(num_processes or mp.cpu_count()))
        func = partial(
            _unknown_refine_worker,
            data_generator            = self.test_data_generators["uniform"],
            feature_config            = self.feature_config,
            device                    = self.device,
            q_cutoff_1                = 0.01,
            q_cutoff_2                = 0.01,
            matching_method           = 'one2one',
            probe_iterations          = 20,
            q_loss_threshold          = 1e-5,
            percent_matched_threshold = 0.90,
            return_only_converged     = False,
            return_debug_output       = save_result_reconstruction
        )
        results = []

        if save_result_report or save_result_reconstruction:
            if not os.path.exists(save_analysis_to):
                os.mkdir(save_analysis_to)

        for sidx, res in enumerate(tqdm(pool.imap(func, [sample_input]), total=len([sample_input]), desc="Refining unknown top layers")):
            results.append(res)
            if save_result_report:
                with open(f"{save_analysis_to}/report.txt","w") as fh:
                    for i in range(len(self.feature_config.miller_indices)):
                        print(f"Try hkl = {res[f'hkl_{i}']}", file=fh)
                        print(f"NN unit cell = {sample_input['pred_cell_deg'][i]/np.array(3*[q_scale]+3*[1.0])}", file=fh)
                        print(f"Refined unit cell = [{res[f'refined{i}_a']/q_scale},{res[f'refined{i}_b']/q_scale},"
                            f"{res[f'refined{i}_c']/q_scale},{res[f'refined{i}_alpha']},{res[f'refined{i}_beta']},"
                            f"{res[f'refined{i}_gamma']}]", file=fh)
                        print(f"Uncertainty unit cell = [{res[f'unc{i}_a']/q_scale},{res[f'unc{i}_b']/q_scale},"
                            f"{res[f'unc{i}_c']/q_scale},{res[f'unc{i}_alpha']},{res[f'unc{i}_beta']},"
                            f"{res[f'unc{i}_gamma']}]", file=fh) 
                        print(f"Is converged = {res[f'is_converged{i}']}", file=fh)
                        print(f"q loss = {res[f'q_loss{i}']}", file=fh)
                        print(f"Percent matched = {res[f'percent_matched{i}']}\n", file=fh)                 
            if save_result_reconstruction:      
                for i in range(len(self.feature_config.miller_indices)):
                    fig, axs = plt.subplots(1,3, figsize=(3*6*self.feature_config.max_q_xy, 6*self.feature_config.max_q_z))
                    
                    # right subplot: refined result
                    for idx, ax in enumerate(axs):
                        if idx==0:
                            plot_dict = res[f"init_plot_dict{i}"]
                        else:
                            plot_dict = res[f"refined_plot_dict{i}"]
                        if idx < 2:
                            if idx == 1 and i == 0:
                                first_phase_match_qxy = plot_dict['sim_q'][:, 0]
                                first_phase_match_qz = plot_dict['sim_q'][:, 1]
                            ax.scatter(
                                plot_dict["sim_q"][:, 0]*q_scale,
                                plot_dict["sim_q"][:, 1]*q_scale,
                                s=180,
                                facecolors='none',
                                edgecolors='g',
                                marker='o',
                                label="simulated"
                            )
                            ax.scatter(
                                plot_dict["test_qxy"]*q_scale,
                                plot_dict["test_qz"]*q_scale,
                                s=180,
                                c="r",
                                marker="x",
                                label="observed"
                            )
                        
                            ax.legend(fontsize=14)
                            #ax.set_title("Refinement Fit Results",fontsize=32)
                            ax.set_xlabel(r"$q_{xy}\,/\,\mathrm{Å}$", fontsize=32)
                            ax.set_ylabel(r"$q_{z}$", fontsize=32)
                            ax.set_xlim(0, self.feature_config.max_q_xy*q_scale)
                            ax.set_ylim(0, self.feature_config.max_q_z*q_scale)
                        else:
                            ax.imshow(images[0,:,:,0])
                        # axs[1].grid(False)
                        ax.tick_params(axis='both', which='major', labelsize=18)
                    plt.savefig(f"{save_analysis_to}/01reconstruction_{i}.png", transparent=False)
                    plt.close()
                    
        pool.close()
        pool.join()

        if first_phase_match_qz is None or first_phase_match_qxy is None:
            raise RuntimeError(f"Could not match first phase")
        #FIT SECOND PHASE:
        q_test = np.vstack([q_xy_test, q_z_test]).T
        q_sim_first = np.vstack([first_phase_match_qxy, first_phase_match_qz]).T

        #Now remove all points from q_test which are part of the other phase
        tolerance = 1e-3
        mask = np.ones(q_test.shape[0], dtype=bool)
        for i, pt in enumerate(q_test):
            if np.any(np.linalg.norm(q_sim_first-pt, axis=1) <= tolerance):
                mask[i] = False
        q_test = q_test[mask]
        q_xy_test = q_test[:,0]
        q_z_test = q_test[:,1]
        
        #filter values:
        q_is_not_too_large = np.logical_not(np.logical_or(q_xy_test > self.feature_config.max_q_xy,
                                    q_z_test > self.feature_config.max_q_z))
        q_xy_test = q_xy_test[q_is_not_too_large]
        q_z_test = q_z_test[q_is_not_too_large]
        #remove specular scan
        if not self.feature_config.use_specular_scan: 
            q_is_not_too_specular = np.logical_not(q_xy_test < self.feature_config.specular_cutoff)
            q_xy_test = q_xy_test[q_is_not_too_specular]
            q_z_test = q_z_test[q_is_not_too_specular]
        else:
            q_is_not_direct_beam = np.logical_not(np.logical_or(q_xy_test > 0.01, q_z_test > 0.01))
            q_xy_test = q_xy_test[q_is_not_direct_beam]
            q_z_test = q_z_test[q_is_not_direct_beam]

        #for idx in range(len(q_xy_test)): print(f"[{q_xy_test[idx]} | {q_z_test[idx]}]")

        if self.gixd_model is None:
            self.predict_and_save_csv(num_samples=0, unknown_top_layer=True)

        q_xy_test = torch.from_numpy(q_xy_test).unsqueeze(0).to(torch.float64)
        q_z_test = torch.from_numpy(q_z_test).unsqueeze(0).to(torch.float64)

        self.gixd_model.eval()

        num_orients = len(self.feature_config.miller_indices)
        images = self.test_data_generators["uniform"].calc_images_from_q_points(q_xy_test, q_z_test)

        with torch.no_grad():
            images   = images.to(self.device)

            img_rep = images.repeat(num_orients, 1, 1, 1)
            top_idx = torch.arange(
                num_orients,
                device=self.device,
                dtype=torch.int64
            )
            out = self.gixd_model([img_rep, top_idx]).cpu()
            if isinstance(out, tuple):
                means, _, weights = out
                pred_norm_all = (means * weights.unsqueeze(-1)).sum(1)
            else:
                pred_norm_all = out
            
            pred_cell_deg = GixdUtility.denormalize_cell_deg(pred_norm_all) \
                                        .cpu().tolist()
           
            sample_input = {
                'sample_index':   0,
                'pred_norm':      pred_norm_all.tolist(),
                'pred_cell_deg':  pred_cell_deg,
                'q_xy':           q_xy_test.cpu().numpy(),
                'q_z':            q_z_test.cpu().numpy()
            }


        pool = mp.Pool(processes=(num_processes or mp.cpu_count()))
        func = partial(
            _unknown_refine_worker,
            data_generator            = self.test_data_generators["uniform"],
            feature_config            = self.feature_config,
            device                    = self.device,
            q_cutoff_1                = 0.1,
            q_cutoff_2                = 0.1,
            matching_method           = 'one2one', #simulated_fractional
            probe_iterations          = 20,
            q_loss_threshold          = 1e-5,
            percent_matched_threshold = 0.90,
            return_only_converged     = False,
            return_debug_output       = save_result_reconstruction
        )
        results = []

        if save_result_report or save_result_reconstruction:
            if not os.path.exists(save_analysis_to):
                os.mkdir(save_analysis_to)

        for sidx, res in enumerate(tqdm(pool.imap(func, [sample_input]), total=len([sample_input]), desc="Refining unknown top layers")):
            results.append(res)
            if save_result_report:
                with open(f"{save_analysis_to}/report2.txt","w") as fh:
                    for i in range(len(self.feature_config.miller_indices)):
                        print(f"Try hkl = {res[f'hkl_{i}']}", file=fh)
                        print(f"NN unit cell = {sample_input['pred_cell_deg'][i]/np.array(3*[q_scale]+3*[1.0])}", file=fh)
                        print(f"Refined unit cell = [{res[f'refined{i}_a']/q_scale},{res[f'refined{i}_b']/q_scale},"
                            f"{res[f'refined{i}_c']/q_scale},{res[f'refined{i}_alpha']},{res[f'refined{i}_beta']},"
                            f"{res[f'refined{i}_gamma']}]", file=fh)
                        print(f"Uncertainty unit cell = [{res[f'unc{i}_a']/q_scale},{res[f'unc{i}_b']/q_scale},"
                            f"{res[f'unc{i}_c']/q_scale},{res[f'unc{i}_alpha']},{res[f'unc{i}_beta']},"
                            f"{res[f'unc{i}_gamma']}]", file=fh) 
                        print(f"Is converged = {res[f'is_converged{i}']}", file=fh)
                        print(f"q loss = {res[f'q_loss{i}']}", file=fh)
                        print(f"Percent matched = {res[f'percent_matched{i}']}\n", file=fh)                 
            if save_result_reconstruction:      
                for i in range(len(self.feature_config.miller_indices)):
                    fig, axs = plt.subplots(1,3, figsize=(3*6*self.feature_config.max_q_xy, 6*self.feature_config.max_q_z))
                    
                    # right subplot: refined result
                    for idx, ax in enumerate(axs):
                        if idx==0:
                            plot_dict = res[f"init_plot_dict{i}"]
                        else:
                            plot_dict = res[f"refined_plot_dict{i}"]
                        if idx < 2:
                            ax.scatter(
                                plot_dict["sim_q"][:, 0]*q_scale,
                                plot_dict["sim_q"][:, 1]*q_scale,
                                s=180,
                                facecolors='none',
                                edgecolors='g',
                                marker='o',
                                label="simulated"
                            )
                            ax.scatter(
                                plot_dict["test_qxy"]*q_scale,
                                plot_dict["test_qz"]*q_scale,
                                s=180,
                                c="r",
                                marker="x",
                                label="observed"
                            )
                        
                            ax.legend(fontsize=14)
                            #ax.set_title("Refinement Fit Results",fontsize=32)
                            ax.set_xlabel(r"$q_{xy}\,/\,\mathrm{Å}$", fontsize=32)
                            ax.set_ylabel(r"$q_{z}$", fontsize=32)
                            ax.set_xlim(0, self.feature_config.max_q_xy*q_scale)
                            ax.set_ylim(0, self.feature_config.max_q_z*q_scale)
                        else:
                            ax.imshow(images[0,:,:,0])
                        # axs[1].grid(False)
                        ax.tick_params(axis='both', which='major', labelsize=18)
                    plt.savefig(f"{save_analysis_to}/02_reconstruction_{i}.png", transparent=False)
                    plt.close()
                    
        pool.close()
        pool.join()
        return res
    

    def optimize_unknown_refinement(
        self,
        num_trials: int = 16,
        num_samples: int = 128,
        top_k: int = 3,
        num_processes: int = None,
        seed: int = 42
    ) -> optuna.Study:
        """
        Tune candidate acceptance on validation data only.

        The objective is 100 * true-plane inclusion rate - mean candidate count / 10;
        it is not unit-cell recovery, a top-k metric, or a percentage. The top_k
        argument remains for call compatibility and does not affect this objective.

        We reseed Python, NumPy and PyTorch before every trial so that
        the validation generator (and DataLoader) always produce identical
        batches across trials.
        """
        def _seed_everything():
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        def objective(trial: optuna.Trial) -> float:
            # Suggest hyperparameters
            q1 = 0.1#trial.suggest_float('matching_q_cutoff_1', 0.01, 1.0, log=True)
            q2 = 0.066 #(found before) trial.suggest_float('matching_q_cutoff_2', 0.05, 1.0)
            start_time = time.perf_counter()
            # Reset dedicated Generator streams too; global seeding alone is insufficient.
            _seed_everything()
            generator = self.validation_data_generators["uniform"]
            generator.rng = np.random.default_rng(seed)
            generator.unit_cell_generator.rng = np.random.default_rng(seed + 1)

            # Call your existing routine—make sure it returns the top-k accuracy
            acc, avg_num_hkl = self.predict_and_refine_unknown_benchmark(
                data_split       = "validation",
                num_samples      = num_samples,
                output_csv       = f"/tmp/optuna_out.csv",
                num_processes    = num_processes,
                q_cutoff_1       = q1,
                q_cutoff_2       = q2,
                probe_iterations = trial.suggest_int("probe_iterations", 1, 10),
                q_loss_threshold = trial.suggest_float("q_loss_threshold", 1e-15, 1.0, log=True),
                percent_matched_threshold= trial.suggest_float("percent_matched_threshold", 0.0, 1.0)
            )
            time_elapsed_s = time.perf_counter()-start_time
            trial.set_user_attr('time_s', time_elapsed_s)
            trial.set_user_attr('avg_num_hkl', avg_num_hkl)
            trial.set_user_attr('acc', acc)
            return 100*acc-avg_num_hkl/10

        search_space = {"probe_iterations": [10],
                        "percent_matched_threshold": [0.9, 0.95],
                        "q_loss_threshold": [1e-4, 1e-5, 1e-6]}
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.GridSampler(search_space))
        #study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=num_trials)

        if True:
            with open(f"{self.path_config.root_path}/optimizer_tuning_unknown_hkl.txt", "a") as f:
                print("\n=== All trials ===", file=f)
                for trial in study.trials:
                    time_s = trial.user_attrs['time_s']
                    avg_num_hkl = trial.user_attrs['avg_num_hkl']
                    percent_th = trial.params['percent_matched_threshold']
                    probe_iter = trial.params["probe_iterations"]
                    print(f"Trial {trial.number:2d}: probe iter {probe_iter} %th = {percent_th:.2f}"
                        f"objective={trial.value:7.5f} time={time_s/num_samples:.2f} s num_hkl = {avg_num_hkl:.2f}", file=f)

                best_params = study.best_trial.params
                best_score = study.best_trial.value
                print("Best hyperparameters:", best_params, file=f)
                print("Best validation objective:", best_score, file=f)

        print("Best params:", study.best_params)
        print(f"Best validation objective: {study.best_value:.4f}")
        return study

def _observed_peak_count(qxy):
    """Return the number of peaks for batched and unbatched q-coordinate arrays."""
    return int(qxy.shape[-1])


def _unknown_refine_worker(
    sample,
    data_generator,
    feature_config,
    device,
    q_cutoff_1,
    q_cutoff_2,
    matching_method="one2one",
    probe_iterations=20,
    maxiter=50,
    outer_loops=10,
    q_loss_threshold=1e-3,
    percent_matched_threshold=0.9,
    return_only_converged=False,
    return_debug_output=False,
    refinement_backend="auto"
):
    """
    sample: dict with keys
      'sample_index', 'pred_norm' (6,), 'pred_cell_deg', 'q_xy', 'q_z',
      'true_top_layer' (int), 'true_cell_deg' (normalized 6,)
    Returns a flat dict containing:
      - sample_index
      - true_hkl, true_{a,b,c,alpha,beta,gamma}
      - true_in_topk (bool)
      - best{i}_hkl, best{i}_pct_match, best{i}_loss  for i=1..top_k
      - refined{i}_{a,b,c,alpha,beta,gamma}, unc{i}_{a,b,c,alpha,beta,gamma} for i=1..top_k
      - final_hkl, refined_is_converged
    """
    from gidind_ai.batched_refinement import Geometry, refine_batch, resolve_backend, result_rows
    import time
    backend = resolve_backend(refinement_backend, device)
    print_debug = False
    # restore tensors
    qxy    = torch.tensor(sample['q_xy'],    dtype=torch.float64, device=device)
    qz     = torch.tensor(sample['q_z'],     dtype=torch.float64, device=device)

    converged_hkls = []
    refined_cells = []
    refined_uncs = []
    q_loss_list = []
    percent_matched_list = []
    is_converged_list = []
    if return_debug_output:
        plot_dict_refined_list = []
        plot_dict_init_list = []

    out = {'sample_index': sample['sample_index']}
    

    if print_debug: print(f"Sample[{sample['sample_index']}]: ")

    batch_rows = None
    timings = {}
    if backend == 'batched':
        observations = torch.stack((qxy.reshape(-1), qz.reshape(-1)), -1)
        batch_result = refine_batch(
            torch.as_tensor(sample['pred_cell_deg'], device=device, dtype=torch.float64),
            observations, torch.as_tensor(feature_config.miller_indices, device=device),
            Geometry.from_generator(data_generator, device),
            matching_method=matching_method, q_cutoff=q_cutoff_1,
            maxiter=maxiter, outer_loops=outer_loops, return_uncertainties=True)
        batch_rows = result_rows(batch_result, observations)
        timings = batch_result['timings']
    refinement_start = time.perf_counter()
    iteration_counts = []
    loss_histories = []
    # The loop below serializes reporting only for the batched backend.
    # The legacy CG calls and acceptance rules remain the serial reference.
    # iterate all orientations
    for hkl_idx, hkl in enumerate(tqdm(
        feature_config.miller_indices,
        desc=f"Sample {sample['sample_index']} full loop",
        leave=False
    )):
        result = batch_rows[hkl_idx] if batch_rows is not None else direct_inversion_pytorch(
            first_predicted_cell_deg = torch.tensor(sample['pred_cell_deg'][hkl_idx], dtype=torch.float64, device=device),
            observed_qxy            = qxy,
            observed_qz             = qz,
            trial_top_layer         = hkl,
            data_generator          = data_generator,
            loss_type               = 'l2',
            method                  = 'cg',
            q_cutoff                = q_cutoff_1,
            maxiter                 = maxiter,
            outer_loops             = outer_loops,
            matching_method         = matching_method,
            torch_lbfgs_lr          = 1.0,
            return_uncertainties    = True
        )

        iteration_counts.append(result.get('iterations'))
        loss_histories.append(result.get('loss_history', []))
        uncertainties = result["refined_uncertainty_cell_deg"].cpu().numpy()
        if np.any(np.isnan(uncertainties)) and return_only_converged:
            continue

        if (np.any(uncertainties[:3]>0.1) or np.any(uncertainties[3:]>1.0)) and return_only_converged:
            continue

        q_loss = result["refined_q_loss"]
        refined_plot_dict = result["refined_plot_dict"]
        init_plot_dict = result["init_plot_dict"]
        num_matches = len(refined_plot_dict["indices_match"])
        # ``test_qxy`` is one-dimensional for real peak lists and may carry a
        # leading batch dimension for synthetic data.  The final axis is the
        # peak axis in both cases.
        observed_peaks = _observed_peak_count(refined_plot_dict["test_qxy"])
        pct_p       = num_matches / observed_peaks if observed_peaks > 0 else 0.0

        if (not np.isfinite(float(q_loss)) or q_loss >= q_loss_threshold or pct_p < percent_matched_threshold) and return_only_converged:
            continue

        cell_full = result['refined_predicted_cell_deg'].cpu().tolist()
        unc_full  = result['refined_uncertainty_cell_deg'].cpu().tolist()

        # record this converged orientation
        converged_hkls.append(tuple(hkl))
        refined_cells.append(cell_full)
        refined_uncs.append(unc_full)
        q_loss_list.append(q_loss)
        percent_matched_list.append(pct_p)
        if return_debug_output:
            plot_dict_refined_list.append(refined_plot_dict)
            plot_dict_init_list.append(init_plot_dict)

        if np.any(np.isnan(uncertainties)) or np.any(uncertainties[:3]>0.1) or np.any(uncertainties[3:]>1.0) or\
            not np.isfinite(float(q_loss)) or q_loss >= q_loss_threshold or pct_p < percent_matched_threshold:
            is_converged_list.append(False)
        else:
            is_converged_list.append(True)

        if print_debug:
            idx = sample['sample_index']
            print(f"[{idx}]: try {hkl} / {feature_config.miller_indices[sample['true_top_layer']]}: {result['convergence_status']}\n"
                  f"[{idx}]: uncertainty = {uncertainties}\n"
                  f"[{idx}]: q_loss = {q_loss} pct_match = {pct_p}\n")


    

    # true cell: denormalize normalized input
    if sample.get('true_cell_deg') is not None:
        true_norm = torch.tensor(sample['true_cell_deg'], dtype=torch.float32, device=device)
        true_cell = GixdUtility.denormalize_cell_deg(true_norm).tolist()
        a,b,c,alpha,beta,gamma = true_cell
        true_hkl = tuple(feature_config.miller_indices[sample['true_top_layer']])
        out.update({
            'true_hkl':                  true_hkl,
            'true_a':                    a, 'true_b': b, 'true_c': c,
            'true_alpha':                alpha, 'true_beta': beta, 'true_gamma': gamma
        })

        out['true_hkl_included'] = true_hkl in converged_hkls
        out["number_of_hkl"] = len(converged_hkls)

    # store each converged result
    for i, (hkl, cell, unc, q_loss, percent_matched) in enumerate(zip(
            converged_hkls, refined_cells, refined_uncs, q_loss_list, percent_matched_list)):
        out[f'hkl_{i}'] = hkl
        out.update({
            f'refined{i}_a': cell[0],
            f'refined{i}_b': cell[1],
            f'refined{i}_c': cell[2],
            f'refined{i}_alpha': cell[3],
            f'refined{i}_beta': cell[4],
            f'refined{i}_gamma': cell[5],
            f'unc{i}_a': unc[0],
            f'unc{i}_b': unc[1],
            f'unc{i}_c': unc[2],
            f'unc{i}_alpha': unc[3],
            f'unc{i}_beta': unc[4],
            f'unc{i}_gamma': unc[5],
            f'is_converged{i}': is_converged_list[i],
            f'q_loss{i}': q_loss,
            f'percent_matched{i}':percent_matched
        })
        if return_debug_output:
            out.update({
                f"refined_plot_dict{i}": plot_dict_refined_list[i],
                f"init_plot_dict{i}": plot_dict_init_list[i]
            })

    out['refinement_metadata'] = {
        'backend': backend, 'optimizer': 'levenberg_marquardt' if backend == 'batched' else 'cg',
        'iterations': iteration_counts, 'loss_history': loss_histories,
        'timings': timings if batch_rows is not None else {
            'total_ms': 1000 * (time.perf_counter() - refinement_start)
        }
    }
    return out




def global_refine_sample(sample_dict, data_generator, target_config, feature_config,
                         return_uncertainties=True, refinement_backend="auto"):
    """
    Helper function to run direct_inversion on one sample.
    Converts stored JSON strings for q_xy and q_z back to arrays,
    constructs the first_prediction, and calls direct_inversion.
    """
    device = torch.device(data_generator.device)
    # Convert stored q_xy and q_z strings back to torch tensors.
    qxy = torch.tensor(json.loads(sample_dict.get("q_xy")), dtype=torch.float64).to(device)
    qz = torch.tensor(json.loads(sample_dict.get("q_z")), dtype=torch.float64).to(device)
    qxy, qz = GixdDataGenerator.filter_q_points(qxy, qz, 1e-6)

    first_predicted_cell_deg = torch.tensor([
        float(sample_dict.get("pred_a")), 
        float(sample_dict.get("pred_b")), 
        float(sample_dict.get("pred_c")),
        float(sample_dict.get("pred_alpha")), 
        float(sample_dict.get("pred_beta")), 
        float(sample_dict.get("pred_gamma"))
    ], dtype=torch.float64).to(device)

    # Call the direct_inversion function (assumed imported from polishing_fit).
    results = direct_inversion_pytorch(
        first_predicted_cell_deg=first_predicted_cell_deg,
        observed_qxy=qxy,
        observed_qz=qz,
        trial_top_layer=feature_config.miller_indices[sample_dict.get("true_top_layer")],
        data_generator=data_generator,            
        method='cg', #lm, trf
        loss_type='l2',
        q_cutoff=0.056, # 0.066
        maxiter=500,
        outer_loops=20,
        matching_method='hybrid',
        torch_lbfgs_lr=1.0,
        return_uncertainties=return_uncertainties,
        refinement_backend=refinement_backend
    )

    refined_cell = results.get("refined_predicted_cell_deg").detach().cpu()
    refined_uncertainty = results.get("refined_uncertainty_cell_deg").detach().cpu()
    refined_q_loss = results.get("refined_q_loss")
    if isinstance(refined_q_loss, torch.Tensor):
        refined_q_loss = refined_q_loss.detach().cpu().item()
    
    return {
        "refined_a": refined_cell[0].item(),
        "refined_b": refined_cell[1].item(),
        "refined_c": refined_cell[2].item(),
        "refined_alpha": refined_cell[3].item(),
        "refined_beta": refined_cell[4].item(),
        "refined_gamma": refined_cell[5].item(),
        "debug_init_plot_dict": results.get("init_plot_dict"),
        "debug_refined_plot_dict": results.get("refined_plot_dict"),
        "debug_second_cell_deg": refined_cell.numpy(),
        "debug_first_cell_deg": first_predicted_cell_deg.detach().cpu().numpy(),
        "debug_true_cell_deg": [
            sample_dict.get("true_a"), sample_dict.get("true_b"), sample_dict.get("true_c"),
            sample_dict.get("true_alpha"), sample_dict.get("true_beta"), sample_dict.get("true_gamma")
        ],
        "refined_q_loss": refined_q_loss,
        "refined_uncertainty_a": refined_uncertainty[0].item(),
        "refined_uncertainty_b": refined_uncertainty[1].item(),
        "refined_uncertainty_c": refined_uncertainty[2].item(),
        "refined_uncertainty_alpha": refined_uncertainty[3].item(),
        "refined_uncertainty_beta": refined_uncertainty[4].item(),
        "refined_uncertainty_gamma": refined_uncertainty[5].item(),
        "debug_uncertainty_cell_deg": refined_uncertainty.numpy(),
    }
