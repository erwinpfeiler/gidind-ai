"""Regression cases for invalid CUDA pixel indices during synthetic training."""

from types import SimpleNamespace

import pytest
import torch

from gixd_data_generator import GixdDataGenerator


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("use_specular", [True, False])
def test_noise_rejects_either_negative_coordinate_before_rasterization(
    monkeypatch, device, use_specular
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    generator = object.__new__(GixdDataGenerator)
    generator.device = torch.device(device)
    generator.data_mode = "training"
    generator.image_size = 256
    generator.num_channels = 3
    generator.max_intensity = 1.0
    generator.q_xy_max = generator.q_z_max = 2.0
    generator.laue_indices_tensor = torch.tensor([[1.0, 0.0, 1.0]] * 8, device=device)
    generator.laue_indices_tensor[4] = torch.tensor([0.0, 0.0, 1.0], device=device)
    generator.feature_config = SimpleNamespace(
        use_specular_scan=use_specular,
        specular_cutoff=0.05,
        training_gaussian_noise_sigma=0.005,
    )
    # Noise can put a single coordinate below zero near the specular axis.
    # Include both-negative, axis, valid-interior, and upper-bound cases too.
    draws = iter((
        torch.tensor([[-0.001, 0.5, -0.001, 0.5, 0.0, 0.5, 2.0, 0.5]], device=device),
        torch.tensor([[0.5, -0.01, -0.01, 0.5, 0.5, 0.0, 0.5, 2.0]], device=device),
    ))
    monkeypatch.setattr(torch, "normal", lambda **kwargs: next(draws))
    qxy, qz = generator.calc_q_points(torch.eye(3, device=device).unsqueeze(0))
    assert torch.equal(qxy, torch.tensor(
        [[0.0, 0.0, 0.0, 0.5, 0.0, 0.5, 0.0, 0.0]], device=device
    ))
    assert torch.equal(qz, torch.tensor(
        [[0.0, 0.0, 0.0, 0.5, 0.5 if use_specular else 0.0, 0.0, 0.0, 0.0]],
        device=device,
    ))
    # Exercise the actual rasterizer, without stochastic training augmentation.
    generator.data_mode = "validation"
    images = generator.calc_images_from_q_points(qxy, qz)
    assert torch.count_nonzero(images).item() == (9 if use_specular else 6)
    assert torch.equal(images[0, 191, 64], torch.ones(3, device=device))
    assert torch.equal(images[0, 255, 64], torch.ones(3, device=device))


def test_in_range_peaks_at_image_edges_remain_visible():
    generator = object.__new__(GixdDataGenerator)
    generator.data_mode = "validation"
    generator.image_size = 256
    generator.num_channels = 3
    generator.max_intensity = 1.0
    generator.q_xy_max = generator.q_z_max = 2.0
    qxy = torch.tensor([0.0, 1.999, 0.5], dtype=torch.float64)
    qz = torch.tensor([0.5, 0.5, 0.0], dtype=torch.float64)
    images = generator.calc_images_from_q_points(qxy, qz)
    assert torch.count_nonzero(images).item() == 9
    assert images[0, 191, 0].sum() == 3
    assert images[0, 191, 255].sum() == 3
    assert images[0, 255, 64].sum() == 3


@pytest.mark.parametrize("data_mode", ["training", "validation", "testing"])
@pytest.mark.parametrize("use_specular", [True, False])
def test_specular_scan_has_only_vertical_position_noise(monkeypatch, data_mode, use_specular):
    generator = object.__new__(GixdDataGenerator)
    generator.device = torch.device("cpu")
    generator.data_mode = data_mode
    generator.q_xy_max = generator.q_z_max = 2.0
    generator.laue_indices_tensor = torch.zeros((5, 3), dtype=torch.float64)
    generator.feature_config = SimpleNamespace(
        use_specular_scan=use_specular, specular_cutoff=0.05,
        training_gaussian_noise_sigma=0.005, testing_gaussian_noise_sigma=0.005,
    )
    # Exact axis, numerical rotation residual, nearby off-axis, and ordinary
    # off-axis reflections. Nearby reflections must not be snapped to the axis.
    noiseless_q = torch.tensor([[
        [0.0, 0.0, 0.5], [0.999e-4, 0.0, 0.7], [1e-4, 0.0, 0.9],
        [1.001e-4, 0.0, 1.1], [0.5, 0.0, 1.3],
    ]], dtype=torch.float64)
    monkeypatch.setattr(
        GixdDataGenerator, "get_scattering_vector", lambda *args: noiseless_q
    )
    monkeypatch.setattr(
        torch, "rand", lambda *args, **kwargs: torch.full(args, 0.5, **kwargs)
    )
    offsets = iter((torch.tensor([[-2.0, 2.0, 2.0, 2.0, 2.0]]), -1.0))
    monkeypatch.setattr(torch, "normal", lambda *, mean, std: mean + std * next(offsets))
    qxy, qz = generator.calc_q_points(torch.eye(3).unsqueeze(0))
    sigma = 0.005 if data_mode == "testing" else 0.0025
    if use_specular:
        assert torch.equal(qxy[0, :3], torch.zeros(3, dtype=torch.float64))
        assert torch.allclose(qz[0, :3], noiseless_q[0, :3, 2] - sigma)
        assert qxy[0, 3].item() == pytest.approx(1.001e-4 + 2 * sigma)
    else:
        assert torch.count_nonzero(qxy[0, :4]) == 0
        assert torch.count_nonzero(qz[0, :4]) == 0
    assert qxy[0, 4].item() == pytest.approx(0.5 + 2 * sigma)
    assert qz[0, 4].item() == pytest.approx(1.3 - sigma)
