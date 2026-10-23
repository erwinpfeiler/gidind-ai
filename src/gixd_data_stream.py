import torch
from torch.utils.data import IterableDataset
from collections import deque

from config_loader import FeatureConfig

class FilteredBatchStream(IterableDataset):
    """
    Wraps a GixdDataGenerator that produces (big) variable-quality batches and
    yields fixed-size batches after filtering by number of observed peaks.
    """
    def __init__(self, base_generator, batch_size: int,
                 min_peaks: int = 5, max_peaks: int = 300,
                 chunk_size: int = None, buffer_cap: int = None,
                 samples_per_epoch: int = None):
        super().__init__()
        self.feature_config = FeatureConfig()
        self.generator = base_generator  # GixdDataGenerator should inherit from torch.utils.data.Dataset
        self.batch_size = int(batch_size)
        self.samples_per_epoch = int(samples_per_epoch if samples_per_epoch is not None
                                     else getattr(base_generator, "epoch_size", self.batch_size))
        self.min_peaks = int(min_peaks)
        self.max_peaks = int(max_peaks)
        self.info = {} # extra info dict to pass extra information for debuggin/analysis
        # how many candidates to generate per chunk (default 1.5× batch)
        self.chunk = int(chunk_size or 1.5 * self.batch_size)
        # upper bound for queue length (in samples) to avoid unbounded memory
        self.buffer_cap = int(buffer_cap or 3 * self.batch_size)

    def __len__(self):
        if self.samples_per_epoch <= 0:
            return 0
        return (self.samples_per_epoch + self.batch_size - 1) // self.batch_size

    @torch.no_grad()
    def __iter__(self):
        device = self.generator.device
        buf = deque()  # each entry holds a filtered tensor chunk (on GPU)
        buffered_samples = 0
        yielded_samples = 0

        def _to_device(value, ref_device, dtype=None):
            if torch.is_tensor(value):
                tensor = value
                if dtype is not None:
                    tensor = tensor.to(dtype=dtype)
                return tensor.to(ref_device)
            return torch.as_tensor(value, device=ref_device, dtype=dtype)

        def _append_valid(images, top_idx, targets, lattice, qxy, qz):
            """
            Determine keep mask using image intensity:
            peak_proxy = sum_{h,w}( mean_c images[h,w,c] ) / max_intensity
            """
            nonlocal buffered_samples
            imgs = images
            if imgs.dim() == 4:  # [N,H,W,C]
                if imgs.size(-1) > 1:
                    imgs_scalar = imgs.mean(dim=-1)         # [N,H,W]
                else:
                    imgs_scalar = imgs[..., 0]              # [N,H,W]
            elif imgs.dim() == 3:  # already [N,H,W]
                imgs_scalar = imgs
            else:
                raise ValueError(f"Unexpected images shape: {imgs.shape}")

            # sum over spatial dims and normalize
            num_peaks = imgs_scalar.sum(dim=(1, 2)) / self.feature_config.max_intensity  # [N]

            # keep within bounds (treating proxy as 'number of peaks')
            keep = (num_peaks >= self.min_peaks) & (num_peaks <= self.max_peaks)

            kidx = torch.nonzero(keep, as_tuple=False).view(-1)
            count = kidx.numel()
            if count:
                buf.append(tuple(value[kidx] for value in (
                    images, top_idx, targets, lattice, qxy, qz
                )))
                buffered_samples += count
            return count, int(images.shape[0])

        while yielded_samples < self.samples_per_epoch:
            current_batch_size = min(self.batch_size, self.samples_per_epoch - yielded_samples)
            while buffered_samples < current_batch_size and buffered_samples < self.buffer_cap:
                old_bs = self.generator.batch_size
                self.generator.batch_size = self.chunk
                try:
                    (imgs, top_layers), targets, lattice, qxy, qz = self.generator.__getitem__(0)
                    imgs = _to_device(imgs, device)
                    targets = _to_device(targets, imgs.device, dtype=imgs.dtype)
                    top_layers = _to_device(top_layers, imgs.device)
                    lattice = _to_device(lattice, imgs.device)
                    qxy = _to_device(qxy, imgs.device)
                    qz = _to_device(qz, imgs.device)
                    _append_valid(imgs, top_layers, targets, lattice, qxy, qz)
                finally:
                    self.generator.batch_size = old_bs
            if buffered_samples >= current_batch_size:
                parts = [[] for _ in range(6)]
                remaining = current_batch_size
                while remaining:
                    chunk = buf.popleft()
                    take = min(remaining, chunk[0].shape[0])
                    for field_parts, value in zip(parts, chunk):
                        field_parts.append(value[:take])
                    if take < chunk[0].shape[0]:
                        buf.appendleft(tuple(value[take:] for value in chunk))
                    remaining -= take
                # Slicing one chunk avoids a copy; concatenate only when a
                # batch spans chunks. No per-sample Python objects or stacks.
                images, tops, targets, lattice_systems, qxy, qz = (
                    field[0] if len(field) == 1 else torch.cat(field, dim=0)
                    for field in parts
                )
                tops = tops.view(-1, 1)
                buffered_samples -= current_batch_size
                yielded_samples += current_batch_size
                yield ( (images, tops), targets, lattice_systems, qxy, qz )
            else:
                break
