"""Convert a trusted legacy Lightning checkpoint to a safe tensor-only checkpoint."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def convert(input_path: Path, output_path: Path) -> dict[str, object]:
    """Return and save a tensor-only checkpoint from an explicitly trusted input."""
    import torch

    legacy = torch.load(input_path, map_location="cpu", weights_only=False)
    if not isinstance(legacy, dict) or not isinstance(legacy.get("state_dict"), dict):
        raise TypeError("Expected a Lightning checkpoint containing a state_dict mapping.")

    converted = {
        "state_dict": legacy["state_dict"],
        "epoch": int(legacy.get("epoch", 0)),
        "global_step": int(legacy.get("global_step", 0)),
        "pytorch-lightning_version": str(legacy.get("pytorch-lightning_version", "unknown")),
        "hyper_parameters": dict(legacy.get("hyper_parameters", {})),
        "gidind_ai_conversion": {
            "format": "tensor_only_lightning_checkpoint_v1",
            "source_filename": input_path.name,
            "source_sha256": checksum(input_path),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(converted, output_path)
    return converted["gidind_ai_conversion"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    source = args.input.expanduser().resolve()
    destination = args.output.expanduser().resolve()
    if not source.is_file():
        raise SystemExit(f"Input checkpoint does not exist: {source}")
    if source == destination:
        raise SystemExit("--output must be different from --input.")
    metadata = convert(source, destination)
    print(f"Wrote {destination}")
    print(f"Source SHA-256: {metadata['source_sha256']}")
    print(f"Output SHA-256: {checksum(destination)}")


if __name__ == "__main__":
    main()
