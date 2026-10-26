#!/usr/bin/env python3
"""Prepare a fresh Niggli-deduplicated COD split without fitting GMMs or changing paper assets."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--cod-tables', type=Path, nargs='+', help='Parsed COD pickle shards (trusted local files)')
    source.add_argument('--cif-root', type=Path, help='Root of a locally downloaded COD CIF archive')
    parser.add_argument('--output-dir', type=Path, required=True, help='New empty directory; never the paper-asset folder')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args(argv)
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        parser.error('output directory must be empty; existing datasets are never overwritten')
    inputs = [p.expanduser().resolve() for p in args.cod_tables] if args.cod_tables else None
    cif_root = args.cif_root.expanduser().resolve() if args.cif_root else None
    if inputs and not all(p.is_file() for p in inputs):
        parser.error('one or more COD tables are missing')
    if cif_root and not cif_root.is_dir():
        parser.error('CIF root is not a directory')
    from read_crystal_data_2 import filter_cod_dataframes, load_cod_to_pickle
    from unit_cell_generator import UnitCellGenerator
    output.mkdir(parents=True, exist_ok=True)
    if cif_root:
        load_cod_to_pickle(cif_root, output / 'parsed_cod')
        inputs = [output / 'parsed_cod.pkl']
    data = filter_cod_dataframes(inputs)
    previous = Path.cwd()
    try:
        os.chdir(output)
        generator = object.__new__(UnitCellGenerator)
        generator.lattice_types = list(data)
        generator.seed = args.seed
        generator.load_data(data)
    finally:
        os.chdir(previous)
    manifest_path = output / 'training_unit_cells/split_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['cod_sources'] = [{'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
                               for p in inputs]
    manifest['coordinate_storage'] = ('Legacy-named system.npy tables store conventional coordinates for GMM fitting; '
                                      'system_niggli.npy companions store the corresponding unrounded Niggli cells.')
    manifest['development_validation'] = ('GMM fitting reserves a deterministic 20% of the non-test partition '
                                          'and saves it in validation_crystals for NN/refinement development.')
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Prepared fresh COD split: {output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
