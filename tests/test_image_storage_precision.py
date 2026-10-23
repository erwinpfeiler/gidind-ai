"""Image-storage precision must not alter geometry, augmentation, or filtering."""

from types import SimpleNamespace

import pytest
import torch

from gixd_data_generator import GixdDataGenerator


def _reference_images(generator, qxy, qz):
    # Independent scalar rasterizer retaining the previous image dtype.
    columns = (generator.image_size * qxy / generator.q_xy_max).floor().long()
    rows = ((generator.image_size - 1) * (generator.q_z_max - qz) / generator.q_z_max).round().long()
    images = torch.zeros(
        qxy.shape[0], generator.image_size, generator.image_size, generator.num_channels,
        dtype=qxy.dtype, device=qxy.device,
    )
    for batch in range(qxy.shape[0]):
        for peak in range(qxy.shape[1]):
            images[batch, rows[batch, peak], columns[batch, peak], :] = generator.max_intensity
    if generator.data_mode == "training":
        shape = (qxy.shape[0], generator.image_size, generator.image_size, 1)
        rates_shape = (qxy.shape[0], 1, 1, 1)
        salt_draw = torch.rand(shape, device=qxy.device)
        salt_rate = torch.rand(rates_shape, device=qxy.device) * generator.feature_config.training_extra_peak_rate
        pepper_draw = torch.rand(shape, device=qxy.device)
        pepper_rate = torch.rand(rates_shape, device=qxy.device) * generator.feature_config.training_missing_peak_rate
        images = images * (pepper_draw >= pepper_rate).to(images.dtype)
        images = torch.where(salt_draw < salt_rate, generator.max_intensity, images)
    images[:, -1, 0, :] = 0
    return images


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("data_mode", ["training", "validation", "testing"])
def test_float32_images_preserve_pixel_values_rng_and_filtering(device, dtype, data_mode):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    generator = object.__new__(GixdDataGenerator)
    generator.data_mode = data_mode
    generator.image_size = 256
    generator.num_channels = 3
    generator.q_xy_max = generator.q_z_max = 2.0
    generator.max_intensity = 1.0
    generator.feature_config = SimpleNamespace(
        training_extra_peak_rate=0.00012207, training_missing_peak_rate=0.30,
        max_intensity=1.0,
    )
    # Include degenerate peaks, direct beam, specular axis, and peaks on
    # opposite sides of a pixel boundary distinguishable only in float64.
    qxy = torch.tensor([
        [0.5 - 1e-12, 0.5 + 1e-12, 0.0, 1.999, 0.0],
        [0.4, 0.4, 0.8, 1.1, 0.0],
    ], dtype=dtype, device=device)
    qz = torch.tensor([
        [0.5, 0.5, 0.5, 0.5, 0.0], [0.5, 0.5, 0.8, 1.1, 0.0],
    ], dtype=dtype, device=device)
    original_xy, original_z = qxy.clone(), qz.clone()
    torch.manual_seed(42)
    expected = _reference_images(generator, qxy, qz)
    expected_rng = torch.cuda.get_rng_state() if device == "cuda" else torch.get_rng_state()
    torch.manual_seed(42)
    actual = generator.calc_images_from_q_points(qxy, qz)
    actual_rng = torch.cuda.get_rng_state() if device == "cuda" else torch.get_rng_state()
    assert actual.dtype == torch.float32
    assert torch.equal(actual, expected.float())
    assert torch.equal(actual_rng, expected_rng)
    assert torch.equal(qxy, original_xy) and torch.equal(qz, original_z)
    old_counts = expected.mean(dim=-1).sum(dim=(1, 2))
    new_counts = actual.mean(dim=-1).sum(dim=(1, 2))
    assert torch.equal(new_counts.double(), old_counts.double())
    assert torch.equal((new_counts >= 5) & (new_counts <= 500), (old_counts >= 5) & (old_counts <= 500))
    assert actual.numel() * actual.element_size() == expected.numel() * 4
    if dtype == torch.float64:
        assert actual.element_size() * 2 == expected.element_size()
