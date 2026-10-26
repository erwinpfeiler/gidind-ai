"""FIFO and sample-generation regression checks for tensor-chunk buffering."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from gixd_data_stream import FilteredBatchStream


class SyntheticChunks:
    def __init__(self, device):
        self.device = torch.device(device)
        self.batch_size = 99
        self.epoch_size = 1000
        self.next_id = 0
        self.calls = []

    def __getitem__(self, index):
        self.calls.append(self.batch_size)
        ids = np.arange(self.next_id, self.next_id + self.batch_size)
        self.next_id += self.batch_size
        # Reject the entire first chunk; later chunks contain peaks below,
        # within, and above the accepted range, with identifiable payloads.
        counts = np.zeros(len(ids), dtype=int) if len(self.calls) == 1 else ids % 5
        images = torch.zeros(len(ids), 4, 4, 3, dtype=torch.float64, device=self.device)
        for row, count in enumerate(counts):
            images[row].view(16, 3)[:count] = 1
        tops = torch.as_tensor(ids % 7, dtype=torch.float32, device=self.device)
        targets = np.repeat(ids[:, None], 6, axis=1).astype(np.float32)
        lattice = torch.as_tensor(ids % 3, dtype=torch.int64, device=self.device)
        qxy = torch.as_tensor(np.repeat(ids[:, None], 3, axis=1), device=self.device)
        qz = qxy + 1000
        return (images, tops), targets, lattice, qxy, qz


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("batch_size,chunk_size,samples", [(4, 6, 13), (7, 2, 15), (1, 16, 9), (8, 16, 1)])
def test_chunks_preserve_filtering_order_fields_and_generation_count(
    monkeypatch, device, batch_size, chunk_size, samples
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    monkeypatch.setattr(
        "gixd_data_stream.FeatureConfig", lambda: SimpleNamespace(max_intensity=1.0)
    )
    generator = SyntheticChunks(device)
    stream = FilteredBatchStream(
        generator, batch_size=batch_size, chunk_size=chunk_size,
        samples_per_epoch=samples, min_peaks=2, max_peaks=3,
    )
    batches = list(stream)
    expected_ids = []
    expected_calls = 1
    cursor = chunk_size
    while len(expected_ids) < samples:
        expected_ids.extend(i for i in range(cursor, cursor + chunk_size) if i % 5 in (2, 3))
        cursor += chunk_size
        expected_calls += 1
    ids = torch.tensor(expected_ids[:samples], device=device)
    assert len(batches) == len(stream) == (samples + batch_size - 1) // batch_size
    assert generator.calls == [chunk_size] * expected_calls
    assert generator.batch_size == 99
    assert [batch[1].shape[0] for batch in batches] == [
        min(batch_size, samples - start) for start in range(0, samples, batch_size)
    ]
    images = torch.cat([batch[0][0] for batch in batches])
    tops = torch.cat([batch[0][1] for batch in batches])
    targets = torch.cat([batch[1] for batch in batches])
    lattice = torch.cat([batch[2] for batch in batches])
    qxy = torch.cat([batch[3] for batch in batches])
    qz = torch.cat([batch[4] for batch in batches])
    assert torch.equal(tops, (ids % 7).float().view(-1, 1))
    assert targets.dtype == images.dtype == torch.float64
    assert lattice.dtype == qxy.dtype == qz.dtype == torch.int64
    assert torch.equal(targets, ids[:, None].expand(-1, 6).double())
    assert torch.equal(lattice, ids % 3)
    assert torch.equal(qxy, ids[:, None].expand(-1, 3))
    assert torch.equal(qz, qxy + 1000)
    assert torch.equal(images.mean(dim=-1).sum(dim=(1, 2)), (ids % 5).double())


def test_generation_exception_restores_original_chunk_setting(monkeypatch):
    monkeypatch.setattr(
        "gixd_data_stream.FeatureConfig", lambda: SimpleNamespace(max_intensity=1.0)
    )

    class FailingGenerator:
        device = torch.device("cpu")
        batch_size = 99
        epoch_size = 10

        def __getitem__(self, index):
            raise RuntimeError("simulation failed")

    generator = FailingGenerator()
    stream = FilteredBatchStream(generator, batch_size=2, chunk_size=4)
    with pytest.raises(RuntimeError, match="simulation failed"):
        next(iter(stream))
    assert generator.batch_size == 99
