"""Manifest-driven eight-input benchmark with prepared-input deduplication."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import time
from .config import Settings
from .inputs import read_peaks, seed_coordinates
from .runner import atomic_json, write_outputs
from .supervision import index_file


def copy_run_artifacts(source, destination):
    for name in ('port_source_snapshot.zip','solution_events.jsonl','candidate_timing.json',
                 'solution_records.jsonl','supervision.json'):
        if (Path(source)/name).is_file():
            shutil.copyfile(Path(source)/name,Path(destination)/name)


def prepare_input(source, case, variant, name, settings):
    """Shared validation, mode selection and dedup key for serial/array runs."""
    if Path(name).name != name:
        raise ValueError('Manifest filename must be a basename')
    path = source/name
    if variant == 'original':
        # Inspect actual positive-qz specular rows without assuming original means with-specular.
        try:
            read_peaks(path, 'specular', settings)
            mode, unavailable = 'specular', None
        except ValueError as exc:
            if 'requires a positive-q_z peak' not in str(exc):
                raise
            mode = 'without-specular'
            unavailable = 'No measured specular peak in original input'
    else:
        mode, unavailable = 'without-specular', None
    peaks = read_peaks(path, mode, settings)
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    expected = case['without_specular']['sha256' if variant != 'original' else 'source_sha256']
    if checksum != expected:
        raise ValueError(f'Manifest checksum mismatch: {name}')
    key = (mode, peaks.provenance['prepared_sha256'], json.dumps(settings.to_dict(),sort_keys=True))
    # Explicit seed rows are file-specific; do not reuse different row layouts.
    if settings.seed_peak_rows is not None:
        key += (tuple(peaks.source_rows),)
    return path, mode, unavailable, peaks, key


def reused_result(original, original_dir, peaks, mode, settings):
    """Alias one measured computation, remapping file-specific source rows."""
    result = copy.deepcopy(original)
    result['input'] = peaks.provenance
    seed, rows = seed_coordinates(peaks, mode, settings)
    result['input']['seed_source_rows'] = rows.tolist()
    result['input']['seed_coordinates'] = seed.tolist()
    result['deduplication'] = {'reused_from':str(original_dir/'results.json'),
                               'independent_timing_measurement':False,
                               'note':'Timings belong to reused canonical computation, not an independent run.'}
    if result.get('candidate_storage',{}).get('backend')=='sqlite':
        # Aliases are sibling output directories. Retain one potentially large
        # database and a relocatable reference, not another physical copy.
        result['candidate_storage']['path'] = str(Path('..')/original_dir.name/
                                                result['candidate_storage']['path'])
    for sol in result['solutions']:
        for assignment, row in zip(sol['assignments'], peaks.source_rows):
            assignment['source_row'] = int(row)
    return result


def benchmark(input_dir, output_dir, settings=None):
    settings = (settings or Settings()).validate()
    source, out = Path(input_dir).resolve(), Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out/'benchmark.json').exists():
        raise FileExistsError(f'Refusing to overwrite {out}/benchmark.json')
    manifest_path = source/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    results, cache = [], {}
    started = time.perf_counter()
    summary = {'schema_version':1, 'manifest':str(manifest_path), 'manifest_sha256':manifest_sha,
               'effective_settings':settings.to_dict(), 'jobs':results, 'status':'running'}
    for case in manifest['cases']:
        for variant, name in (('original', case['filename']),
                              ('without_specular',case['without_specular']['filename'])):
            path, mode, unavailable, peaks, key = prepare_input(source, case, variant, name, settings)
            job_dir = out/f"{case['id']}_{variant}"
            if key in cache:
                original_dir, original = cache[key]
                result = reused_result(original, original_dir, peaks, mode, settings)
                job_dir.mkdir(parents=True, exist_ok=False)
                copy_run_artifacts(original_dir,job_dir)
                write_outputs(job_dir, result)
            else:
                result = index_file(path, mode, job_dir, settings)
                cache[key] = (job_dir, result)
            results.append({'case':case['id'], 'variant':variant, 'mode':mode,
                            'specular_comparison_unavailable':unavailable,
                            'status':result['status'], 'num_solutions':len(result['solutions']),
                            'results':str(job_dir/'results.json'),
                            'reused': 'deduplication' in result,
                            'search_work': result.get('search_work')})
            atomic_json(out/'benchmark.json', summary)
            if result['status'] == 'interrupted':
                summary['status'] = 'interrupted'; break
        if summary['status'] == 'interrupted':
            break
    if summary['status'] != 'interrupted':
        summary['status'] = 'finished_with_failures' if any(j['status']=='failure' for j in results) else 'finished'
    summary['total_wall_seconds'] = time.perf_counter()-started
    summary['unique_computations'] = len(cache)
    atomic_json(out/'benchmark.json', summary)
    return summary
