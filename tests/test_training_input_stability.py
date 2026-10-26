"""Input-scale and scheduler checks for the synthetic training path."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from torchvision.models import EfficientNet_V2_S_Weights

from gixd_data_generator import GixdDataGenerator
from gixd_utility import GixdUtility
import gixd_regressor


class CaptureImages(nn.Module):
    def forward(self, images, orientations):
        return images


def _module(monkeypatch):
    monkeypatch.setattr(gixd_regressor, "GixdPointModel", lambda *args, **kwargs: CaptureImages())
    return gixd_regressor.GixdModule(
        num_phases=1, use_gmm=False, num_gmm=2, learning_rate=1e-3,
        training_batch_size=512, validation_batch_size=256, test_batch_size=256,
        simulation_batch_size=1024, max_epochs=817,
        training_samples_per_epoch=100000, validation_samples_per_epoch=5000,
        test_samples_per_epoch=5000, training_data_generator={},
        validation_data_generator={}, test_data_generator={},
    )


@pytest.mark.parametrize("legacy_batch_dimension", [False, True])
def test_forward_keeps_raster_pixels_and_does_not_mutate_input(monkeypatch, legacy_batch_dimension):
    module = _module(monkeypatch)
    images = torch.ones(2, 4, 5, 3)
    images[:, -1, 0, :] = 0  # already excluded by the rasterizer
    original = images.clone()
    inputs = images.unsqueeze(0) if legacy_batch_dimension else images
    actual = module((inputs, torch.tensor([0, 1])))
    weights_transform = EfficientNet_V2_S_Weights.DEFAULT.transforms()
    means = torch.tensor(weights_transform.mean).view(1, 3, 1, 1)
    stds = torch.tensor(weights_transform.std).view(1, 3, 1, 1)
    expected = (original.permute(0, 3, 1, 2) - means) / stds
    torch.testing.assert_close(actual, expected)
    assert torch.equal(images, original)


def test_salt_overlap_keeps_binary_intensity(monkeypatch):
    generator = object.__new__(GixdDataGenerator)
    generator.data_mode = "training"
    generator.image_size = 8
    generator.num_channels = 3
    generator.q_xy_max = generator.q_z_max = 2.0
    generator.max_intensity = 1.0
    generator.feature_config = SimpleNamespace(
        training_extra_peak_rate=1.0, training_missing_peak_rate=0.0,
        max_intensity=1.0,
    )
    # Salt covers the existing peak as well as background. Pepper removes none.
    draws = iter((0.0, 1.0, 1.0, 0.0))
    monkeypatch.setattr(
        torch, "rand", lambda *shape, device=None: torch.full(shape, next(draws), device=device)
    )
    images = generator.calc_images_from_q_points(
        torch.tensor([[0.5]], dtype=torch.float64), torch.tensor([[0.5]], dtype=torch.float64)
    )
    assert images.max().item() == 1.0
    assert torch.equal(images[0, 5, 2], torch.ones(3))
    assert torch.count_nonzero(images[0, -1, 0]) == 0


def test_lr_warmup_peak_has_no_upward_discontinuity(monkeypatch):
    module = _module(monkeypatch)
    module.gixd_model = nn.Linear(1, 1)
    optimizers, schedulers = module.configure_optimizers()
    optimizer, scheduler = optimizers[0], schedulers[0]["scheduler"]
    rates = []
    for _ in range(module.max_epochs):
        rates.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    assert module.get_optimizer_schedule_config()["warmup_epochs"] == 41
    assert rates[40] == pytest.approx(1e-3)
    assert max(rates) == pytest.approx(1e-3)
    assert 0 < rates[40] - rates[39] <= 1e-3 / 41 * (1 + 1e-10)
    assert rates[41] <= rates[40]


def test_cell_targets_use_radians_and_normalize_to_unit_range(monkeypatch):
    config = SimpleNamespace(min_distance=4.0, max_distance=30.0, min_angle=60.0, max_angle=120.0)
    monkeypatch.setattr(GixdUtility, "config", config)
    generator = object.__new__(GixdDataGenerator)
    cells = np.array([[4, 4, 4, 60, 60, 60], [30, 30, 30, 120, 120, 120]], dtype=np.float32)
    parameters = generator.extract_cell_parameters(cells, 2)
    targets = generator.calc_target_tensor(*parameters)
    np.testing.assert_allclose(
        GixdUtility.normalize_cell(targets), [[0]*6, [1]*6], rtol=0, atol=1e-6
    )
