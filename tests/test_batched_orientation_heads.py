"""Verify batched routing against the original per-sample training computation."""

import copy

import pytest
import torch
from torch import nn

from neural_network import GixdPointModel


def _small_model(device):
    # Exercise the production forward method without pretrained downloads or
    # the large image backbone. Heads have the same Linear/SiLU structure.
    model = GixdPointModel.__new__(GixdPointModel)
    nn.Module.__init__(model)
    model.multi_head = True
    model.backbone = nn.Linear(5, 7)
    model.heads = nn.ModuleList([
        nn.Sequential(nn.Linear(7, 8), nn.SiLU(), nn.Linear(8, 3))
        for _ in range(4)
    ])
    return model.to(device)


def _original_forward(model, inputs, orientations):
    feats = model.backbone(inputs)
    return torch.cat([
        model.heads[idx.item()](feats[i].unsqueeze(0))
        for i, idx in enumerate(orientations.view(-1).long())
    ])


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("orientations", [[2], [1, 1, 1], [3, 0, 3, 2, 0, 2]])
def test_routing_preserves_outputs_gradients_and_adam_update(device, orientations):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(42)
    grouped = _small_model(device)
    reference = copy.deepcopy(grouped)
    x = torch.randn(len(orientations), 5, device=device, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_()
    ids = torch.tensor(orientations, device=device).view(-1, 1)
    target = torch.randn(len(orientations), 3, device=device)
    optimizer = torch.optim.Adam(grouped.parameters(), lr=1e-3)
    reference_optimizer = torch.optim.Adam(reference.parameters(), lr=1e-3)

    # Two micro-batches also verify gradient accumulation through the routing.
    for _ in range(2):
        actual = grouped(x, ids)
        expected = _original_forward(reference, reference_x, ids)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        ((actual - target).square().mean() / 2).backward()
        ((expected - target).square().mean() / 2).backward()

    torch.testing.assert_close(x.grad, reference_x.grad, rtol=1e-5, atol=1e-6)
    assert grouped.state_dict().keys() == reference.state_dict().keys()
    for actual, expected in zip(grouped.parameters(), reference.parameters()):
        # Unused heads must keep grad=None, so Adam does not update them.
        assert (actual.grad is None) == (expected.grad is None)
        if actual.grad is not None:
            torch.testing.assert_close(actual.grad, expected.grad, rtol=1e-5, atol=1e-6)
    optimizer.step()
    reference_optimizer.step()
    for actual, expected in zip(grouped.parameters(), reference.parameters()):
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_existing_checkpoint_loads_without_parameter_conversion():
    model = _small_model("cpu")
    restored = _small_model("cpu")
    restored.load_state_dict(model.state_dict(), strict=True)
    x = torch.randn(4, 5)
    ids = torch.tensor([3, 1, 0, 3])
    torch.testing.assert_close(restored(x, ids), _original_forward(model, x, ids))


def test_bfloat16_autocast_preserves_routing_and_backward():
    torch.manual_seed(42)
    model = _small_model("cpu")
    reference = copy.deepcopy(model)
    x = torch.randn(6, 5)
    ids = torch.tensor([3, 0, 3, 2, 0, 2])
    target = torch.randn(6, 3)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = model(x, ids)
        expected = _original_forward(reference, x, ids)
    assert actual.dtype == expected.dtype == torch.bfloat16
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.004)
    (actual.float() - target).square().mean().backward()
    (expected.float() - target).square().mean().backward()
    for a, b in zip(model.parameters(), reference.parameters()):
        assert (a.grad is None) == (b.grad is None)
        if a.grad is not None:
            torch.testing.assert_close(a.grad, b.grad, rtol=0.02, atol=0.001)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('batch', [1, 3])
def test_all_orientation_inference_matches_existing_forward(device, batch):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    torch.manual_seed(931)
    model = _small_model(device).eval()
    x = torch.randn(batch, 5, device=device)
    calls = []
    handle = model.backbone.register_forward_hook(lambda *args: calls.append(1))
    with torch.no_grad():
        actual = model.forward_all_orientations(x)
        assert len(calls) == 1
        expected = model(x.repeat_interleave(4, 0), torch.arange(4,device=device).repeat(batch))
    handle.remove()
    torch.testing.assert_close(actual, expected.reshape(batch,4,3), rtol=1e-5, atol=1e-6)


def test_all_orientation_requires_eval_and_no_grad():
    model = _small_model('cpu')
    x = torch.randn(1,5)
    with torch.no_grad(), pytest.raises(RuntimeError): model.forward_all_orientations(x)
    model.eval()
    with pytest.raises(RuntimeError): model.forward_all_orientations(x)
