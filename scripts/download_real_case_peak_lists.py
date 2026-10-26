#!/usr/bin/env python3
"""Download the pinned experimental peak lists using only Python's standard library."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import urllib.error
import urllib.request

REPOSITORY = "epfeiler/gidind-ai-paper-assets"
REVISION = "9c2aef88ad916c6d353e2664c4185fdebe159216"
MANIFEST_SHA256 = "a2a1958b5310522c5279244d94e42857b699a9cf872c2769c28d392da70297fc"
CASES = ("pq", "cu_bod_mof", "mapbbr3", "naib_s11")
BASE_URL = f"https://huggingface.co/datasets/{REPOSITORY}/resolve/{REVISION}/real_cases"


def download(filename: str) -> bytes:
    headers = {"User-Agent": "gidind-ai-peak-list-downloader"}
    token = os.environ.get("HF_TOKEN")
    if not token:
        cache = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache")))
        home = Path(os.environ.get("HF_HOME", str(cache / "huggingface")))
        token_path = Path(os.environ.get("HF_TOKEN_PATH", str(home / "token")))
        if token_path.is_file():
            token = token_path.read_text().strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(f"{BASE_URL}/{filename}", headers=headers)
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def verify(data: bytes, expected: str, filename: str) -> None:
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError(f"SHA-256 mismatch for {filename}; no output files were replaced.")


def download_peak_lists(output_dir: Path, *, include_originals: bool = False) -> list[Path]:
    """Verify every payload before saving; refuse to overwrite differing local files."""
    manifest_bytes = download("manifest.json")
    verify(manifest_bytes, MANIFEST_SHA256, "manifest.json")
    manifest = json.loads(manifest_bytes)
    cases = {case["id"]: case for case in manifest["cases"]}
    payloads = {"manifest.json": manifest_bytes}
    for case_id in CASES:
        case = cases[case_id]
        derived = case["without_specular"]
        selected = [(derived["filename"], derived["sha256"])]
        if include_originals:
            selected.append((case["filename"], derived["source_sha256"]))
        for filename, checksum in selected:
            if Path(filename).name != filename or not filename.endswith(".csv"):
                raise ValueError(f"Invalid filename in release manifest: {filename!r}")
            data = download(filename)
            verify(data, checksum, filename)
            payloads[filename] = data

    output_dir = output_dir.expanduser().resolve()
    for filename, data in payloads.items():
        destination = output_dir / filename
        if destination.exists() and (
            not destination.is_file() or destination.read_bytes() != data
        ):
            raise ValueError(
                f"Refusing to overwrite different existing file: {destination}. "
                "Choose a new --output-dir."
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    # Stage on the same filesystem, then atomically install each verified file.
    with tempfile.TemporaryDirectory(prefix=".peak-lists-", dir=output_dir) as temporary:
        staging = Path(temporary)
        for filename, data in payloads.items():
            (staging / filename).write_bytes(data)
        for filename in payloads:
            if not (output_dir / filename).exists():
                (staging / filename).replace(output_dir / filename)
    return [output_dir / filename for filename in payloads if filename.endswith(".csv")]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("gidind_peak_lists"))
    parser.add_argument(
        "--include-originals", action="store_true",
        help="Download all eight CSVs instead of only the four specular-free lists.",
    )
    args = parser.parse_args(argv)
    try:
        paths = download_peak_lists(args.output_dir, include_originals=args.include_originals)
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            parser.exit(
                1, "Hugging Face access denied. Log in with `hf auth login` or set HF_TOKEN "
                "to an authorized read token.\n"
            )
        parser.exit(1, f"Hugging Face download failed (HTTP {exc.code}).\n")
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"Download failed: {exc}\n")
    print(f"Verified {len(paths)} CSVs from {REPOSITORY}@{REVISION}")
    for path in paths:
        print(path)
    print(f"Manifest: {paths[0].parent / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
