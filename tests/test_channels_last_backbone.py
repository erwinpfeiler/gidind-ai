"""Exercise CNN memory layout, backward, and old checkpoint/optimizer loading."""

import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import neural_network


class SmallBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 4, 3, padding=1), nn.BatchNorm2d(4), nn.SiLU(),
            nn.Conv2d(4, 6, 3, padding=1), nn.SiLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(nn.Linear(6, 10))

    def forward(self, images):
        return self.classifier(self.features(images).flatten(1))


def _model(monkeypatch, device):
    monkeypatch.setattr(neural_network, "efficientnet_v2_s", lambda **kwargs: SmallBackbone())
    monkeypatch.setattr(
        neural_network, "FeatureConfig", lambda: SimpleNamespace(miller_indices=[(0, 0, 1), (1, 0, 0)])
    )
    return neural_network.GixdPointModel(num_phases=1, multi_head=True).to(device)


def _contiguous_forward(model, images, ids):
    features = model.backbone(images.contiguous())
    return torch.cat([
        model.heads[head_id](features[i:i + 1]) for i, head_id in enumerate(ids.tolist())
    ])


def test_simulator_layout_needs_no_extra_input_copy(monkeypatch):
    model = _model(monkeypatch, "cpu").eval()
    # Match the simulator's contiguous NHWC image followed by NCHW permute.
    images = torch.randn(2, 12, 12, 3).permute(0, 3, 1, 2)
    assert images.is_contiguous(memory_format=torch.channels_last)
    pointers = []
    hook = model.backbone.register_forward_pre_hook(
        lambda module, inputs: pointers.append(inputs[0].data_ptr())
    )
    with torch.no_grad():
        model(images, torch.tensor([0, 1]))
    hook.remove()
    assert pointers == [images.data_ptr()]


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("mixed_precision", [False, True])
def test_channels_last_preserves_training_and_resumes_old_checkpoint(
    monkeypatch, device, mixed_precision
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(42)
    model = _model(monkeypatch, device)
    reference = copy.deepcopy(model).to(memory_format=torch.contiguous_format)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    reference_optimizer = torch.optim.Adam(reference.parameters(), lr=1e-3)
    images = torch.randn(4, 3, 12, 12, device=device)
    ids = torch.tensor([1, 0, 0, 1], device=device)
    targets = torch.randn(4, 6, device=device)

    # Populate an old contiguous checkpoint's Adam state, then resume it into
    # the new layout. Parameter names and optimizer identities must stay valid.
    initial = _contiguous_forward(reference, images, ids)
    (initial - targets).square().mean().backward()
    reference_optimizer.step()
    reference_optimizer.zero_grad(set_to_none=True)
    model.load_state_dict(reference.state_dict(), strict=True)
    optimizer.load_state_dict(copy.deepcopy(reference_optimizer.state_dict()))
    assert model.state_dict().keys() == reference.state_dict().keys()
    for parameter in model.backbone.parameters():
        if parameter.ndim == 4:
            assert parameter.is_contiguous(memory_format=torch.channels_last)

    seen_layout = []
    hook = model.backbone.register_forward_pre_hook(
        lambda module, inputs: seen_layout.append(inputs[0].is_contiguous(memory_format=torch.channels_last))
    )
    with torch.autocast(device, dtype=torch.bfloat16, enabled=mixed_precision):
        actual = model(images, ids)
        expected = _contiguous_forward(reference, images, ids)
    hook.remove()
    assert seen_layout == [True]
    rtol, atol = (0.03, 0.002) if mixed_precision else (1e-4, 1e-6)
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    (actual.float() - targets).square().mean().backward()
    (expected.float() - targets).square().mean().backward()
    for a, b in zip(model.parameters(), reference.parameters()):
        assert (a.grad is None) == (b.grad is None)
        if a.grad is not None:
            torch.testing.assert_close(a.grad, b.grad, rtol=rtol, atol=atol)
    optimizer.step()
    reference_optimizer.step()
    for a, b in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(a, b, rtol=rtol, atol=atol)
