"""Public API, provenance, atomic checkpoints and portable benchmark outputs."""
import csv
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
import zipfile
import numpy as np
from . import IMPLEMENTATION, UPSTREAM_COMMIT, __version__
from .config import Settings
from .finalize import finalize
from .inputs import Peaks, read_peaks, seed_coordinates
from .state import State, SearchLimit


def atomic_json(path, data):
    path = Path(path)
    temp = path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


def port_fingerprint():
    h = hashlib.sha256()
    for p in sorted(Path(__file__).parent.glob('*.py')):
        h.update(p.name.encode()); h.update(p.read_bytes())
    return h.hexdigest()


def environment():
    from scipy import __version__ as scipy_version
    import openpyxl
    try:
        cpu = next(l.split(':', 1)[1].strip() for l in Path('/proc/cpuinfo').read_text().splitlines()
                   if l.startswith('model name'))
    except (OSError, StopIteration):
        cpu = platform.processor()
    return {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
            "cpu": cpu, "logical_cpu_count": os.cpu_count(), "numpy": np.__version__,
            "scipy": scipy_version, "openpyxl": openpyxl.__version__,
            "requested_threads": 1,
            "thread_environment": {k: os.environ.get(k) for k in
                                   ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS')},
            "numpy_loaded_before_port": "Thread count is guaranteed for CLI; restart Python before importing the API if NumPy/BLAS was already initialized."}


def write_outputs(output_dir, result, *, write_result=True):
    output_dir = Path(output_dir)
    solutions = result.get('solutions', [])
    for i, sol in enumerate(solutions, 1):
        sol['id'] = i
        sol['assignment_file'] = f'assignments/solution_{i:05d}.csv'
        dest = output_dir / sol['assignment_file']
        dest.parent.mkdir(parents=True, exist_ok=True)
        assignments = sol['assignments']
        temp = dest.with_suffix('.csv.tmp')
        with temp.open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(assignments[0]))
            w.writeheader(); w.writerows(assignments)
        temp.replace(dest)
    fields = ['id','a','b','c','alpha','beta','gamma','volume_angstrom3','u','v','w',
              'rmse_q_xy','rmse_q_z','rmse_q_xyz','specular_reflection_order','specular_error']
    temp = output_dir / 'solutions.csv.tmp'
    with temp.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for i, sol in enumerate(solutions, 1):
            record = {'id': i, **sol['cell'], 'volume_angstrom3': sol['volume_angstrom3'],
                      **dict(zip(('u','v','w'), sol['contact_plane'])),
                      **{'rmse_'+k: v for k,v in sol['rmse'].items()},
                      'specular_reflection_order': sol['specular_reflection_order'],
                      'specular_error': sol['specular_error']}
            w.writerow(record)
    temp.replace(output_dir/'solutions.csv')
    if result.get('search_work'):
        fields = ['stage','work_unit','attempted_units','possible_units','possible_units_scope',
                  'enumeration_fraction','observed_stage_wall_seconds','amortized_wall_seconds_per_attempt',
                  'attempts_per_wall_second','estimated_exhaustive_stage_wall_seconds',
                  'estimated_remaining_stage_wall_seconds']
        temp = output_dir/'search_work.csv.tmp'
        with temp.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
            writer.writeheader(); writer.writerows(result['search_work']['phases'])
        temp.replace(output_dir/'search_work.csv')
    if write_result:
        atomic_json(output_dir/'results.json', result)


def index_file(input_path, mode, output_dir, settings=None):
    with ExitStack() as resources:
        return _index_file(input_path, mode, output_dir, settings, resources)


def _index_file(input_path, mode, output_dir, settings, resources):
    """Run a headless search and return the exact result written to results.json.

    Nonempty output folders containing an existing result/checkpoint are refused.
    Failure and partial-search statuses are explicit; no candidate implies neither
    success nor exhaustion when a time/solution limit was reached.
    """
    settings = (settings or Settings()).validate()
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any((out/name).exists() for name in ('results.json', 'checkpoint.json','solution_events.jsonl')):
        raise FileExistsError(f'Refusing to replace existing results: {out}')
    started, cpu_started = time.perf_counter(), time.process_time()
    state = State(settings,started=started,candidate_path=out/'native_candidates.sqlite3')
    resources.callback(state.pending.close)
    executor = None
    result = {'schema_version': 1, 'implementation': IMPLEMENTATION,
              'upstream_repository': 'https://github.com/m-kainz/GIDInd',
              'upstream_commit': UPSTREAM_COMMIT, 'port_version': __version__,
              'port_source_sha256': port_fingerprint(),
              'original_executable_equivalence': 'unverified',
              'algorithm_variant': settings.algorithm,
              'algorithm_note': ('Corrected bounded tail search; distinct from upstream GIDInd'
                  if settings.algorithm == 'corrected' else
                  'Previous Python translation of upstream choices; not certified exact MATLAB execution'),
              'mode': mode, 'effective_settings': settings.to_dict(),
              'input': {'path': str(Path(input_path).resolve())}, 'environment': environment(),
              'status': 'running', 'solutions': [],
              'parameter_uncertainty': None,
              'uncertainty_note': 'No parameter uncertainty estimator is present in the ported upstream routines.',
              'adaptations_document': str(Path(__file__).resolve().parents[1]/'PROVENANCE.md')}
    # Preserve the implementation used by this run, not just a hash of code
    # which may subsequently be edited during this local development project.
    port = Path(__file__).resolve().parents[1]
    with zipfile.ZipFile(out/'port_source_snapshot.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted((port/'gidind_reference').glob('*.py')):
            archive.write(path, path.relative_to(port))
        for name in ('README.md','PROVENANCE.md','ALGORITHM_VARIANTS.md','TIMING_AND_EVALUATION.md','RUNTIME_REPAIR.md','CANDIDATE_STORAGE.md','LICENSE','pyproject.toml'):
            if (port/name).is_file():
                archive.write(port/name, name)
    result['port_source_snapshot'] = 'port_source_snapshot.zip'
    event_path = out/'solution_events.jsonl'
    event_stream = event_path.open('x',encoding='utf-8')
    record_stream = (out/'solution_records.jsonl').open('x',encoding='utf-8')
    def record_event(event):
        from .records import encode_record
        record_stream.write(encode_record(event,state.solutions[event['solution_slot']]))
        record_stream.flush()
        event_stream.write(json.dumps(event,allow_nan=False)+'\n')
        event_stream.flush()
    state.on_solution_event = record_event
    result['solution_event_log'] = event_path.name
    result['solution_records'] = 'solution_records.jsonl'
    def phase(name):
        with (out/'execution_events.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(phase=name,wall_seconds=time.perf_counter()-started,
                stage=state.stage,pending=len(state.pending),solutions=len(state.solutions)))+'\n')
    result['solution_event_scope'] = ('Reference-blind acceptance/replacement events during post-search '
        'finalization. Candidate discovery is coordinator commit time, not worker completion or validated availability.')

    saved_pending_revision = -1
    def snapshot(current,final=False):
        nonlocal saved_pending_revision
        snapshot_start = time.perf_counter()
        phase('checkpoint_start')
        current.pending.flush()
        result['candidate_storage'] = current.pending.metadata()
        # Raw candidates are immutable during finalization. Save them at search
        # checkpoints and once at the end (including final enumeration order).
        if current.pending and len(current.pending)<=settings.candidate_snapshot_max_rows and (final or (current.stage!='correction_reduction_discrete_projection'
                and saved_pending_revision!=current.pending_revision)):
            temp = out/'intermediate_candidates.npz.tmp'
            with temp.open('wb') as f:
                np.savez_compressed(f,
                    reciprocal_metrics=np.asarray([c.metric for c in current.pending]),
                    hkl=np.asarray([c.hkl for c in current.pending]),
                    planes=np.asarray([np.full(3, np.nan) if c.plane is None else c.plane for c in current.pending]),
                    projections=np.asarray([np.full(3, np.nan) if c.projection is None else c.projection for c in current.pending]))
            temp.replace(out/'intermediate_candidates.npz')
            atomic_json(out/'intermediate_scores.json', [c.native_scores for c in current.pending])
            atomic_json(out/'candidate_timing.json',[
                dict(pending_index=i,timing=c.timing or current.candidate_timings.get(id(c)))
                for i,c in enumerate(current.pending)])
            saved_pending_revision=current.pending_revision
        elif len(current.pending)>settings.candidate_snapshot_max_rows:
            # Database is the complete native-candidate checkpoint. Never load
            # the growing pool into arrays merely to rewrite an NPZ checkpoint.
            for name in ('intermediate_candidates.npz','intermediate_scores.json','candidate_timing.json'):
                (out/name).unlink(missing_ok=True)
        result.update(status='running', current_stage=current.stage,
                      counts=dict(current.counts), pending_candidates=len(current.pending),
                      partial_reasons=list(current.partial_reasons), solutions=current.solutions,
                      elapsed_seconds=time.perf_counter()-started)
        from .work_metrics import search_work
        live_stages = dict(current.stages)
        live_stages[current.stage] = live_stages.get(current.stage, 0.) + time.perf_counter()-current.stage_started
        result['search_work'] = search_work(result, live_stages)
        # Full fits are journaled once on acceptance/replacement. Rewriting all
        # reflection assignments every few seconds made checkpoint cost grow
        # quadratically and could consume almost the entire time budget.
        checkpoint=dict(result,solutions=[],validated_solution_count=len(current.solutions),
            solutions_storage='solution_records.jsonl; geometry timeline in solution_events.jsonl',
            checkpoint_schema_version=2)
        atomic_json(out/'checkpoint.json', checkpoint)
        elapsed = time.perf_counter()-snapshot_start
        current.stages['checkpoint_io'] = current.stages.get('checkpoint_io', 0.) + elapsed
        if current.stage not in ('initialization', 'writing_outputs'):
            current.checkpoint_compute_io += elapsed
        phase('checkpoint_end')

    state.checkpoint = snapshot
    try:
        peaks = read_peaks(input_path, mode, settings)
        result['input'] = peaks.provenance
        seed, seed_rows = seed_coordinates(peaks, mode, settings)
        result['input']['seed_source_rows'] = seed_rows.tolist()
        result['input']['seed_coordinates'] = seed.tolist()
        # Computational order matches INITIALIZE_GUI / updateplot. Keep source
        # row IDs and restore file order in saved reflection assignments.
        coords = peaks.coordinates
        order = (np.argsort(coords[:, 0], kind='stable') if mode == 'specular' else
                 np.lexsort((coords[:, 1], coords[:, 0], np.linalg.norm(coords, axis=1))))
        peaks = Peaks(coords[order], peaks.source_rows[order], peaks.provenance)
        result['input']['computational_order'] = 'q_xy stable ascending' if mode == 'specular' else 'q_xyz,q_xy,q_z ascending'
        if mode == 'specular':
            from .specular import search
        else:
            from .without_specular import search
        from .parallel import OrderedExecutor
        from .geometry import hkl_grid
        data = {"coordinates": peaks.coordinates, "seed": seed, "settings": settings,
                "qspec": peaks.provenance.get("selected_specular_q")}
        if mode == 'without-specular':
            data.update(seed_hkls=hkl_grid(tuple(settings.without_specular_seed_hkl), sorted_norm=True),
                        indexing=hkl_grid(tuple(settings.without_specular_index_hkl)))
        executor = state.executor = OrderedExecutor(data, settings.workers)
        result['execution'] = executor.info
        result['execution']['cpu_affinity'] = sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None
        phase('search_start')
        search(peaks, seed, settings, state)
        phase('pool_shutdown_start')
        executor.close(abort=bool(state.partial_reasons))
        phase('pool_shutdown_end')
        phase('finalization_start')
        finalize(peaks, mode, settings, state)
        phase('finalization_end')
        result['status'] = ('time_limited' if any('time_limit' in r for r in state.partial_reasons)
                            else 'search_limited' if state.partial_reasons
                            else 'completed' if state.solutions else 'no_solution')
    except KeyboardInterrupt:
        state.interrupted = True
        state.partial('interrupted')
        result['status'] = 'interrupted'
    except SearchLimit as exc:
        state.partial(str(exc)); result['status'] = 'time_limited'
    except Exception as exc:
        result['status'] = 'failure'
        result['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        try:
            if executor is not None:
                executor.close(abort=True)
        finally:
            event_stream.close()
            record_stream.close()
    state.set_stage('writing_outputs')
    # Always preserve completed native batches, including on failure/interruption.
    status = result['status']
    snapshot(state,final=True)
    result.update(status=status, partial_reasons=state.partial_reasons, counts=dict(state.counts),
                  solution_event_count=state.solution_event_count,
                  solution_event_log_sha256=hashlib.sha256(event_path.read_bytes()).hexdigest(),
                  solutions=state.solutions, pending_candidates=len(state.pending),
                  search_complete=status in ('completed','no_solution'),
                  timing={'stage_seconds': state.stages,
                          'compute_wall_seconds': max(0., sum(v for k,v in state.stages.items()
                                                      if k not in ('initialization','checkpoint_io'))
                                                      -state.checkpoint_compute_io),
                          'first_validated_solution_wall_seconds': state.first_solution_wall_seconds,
                          'first_solution_timing_scope': 'First accepted solution during post-search finalization; this port does not stream final solutions during seed enumeration.',
                          'cpu_seconds_before_output': time.process_time()-cpu_started,
                          'wall_seconds_before_output': time.perf_counter()-started})
    from .work_metrics import search_work
    result['search_work'] = search_work(result)
    phase('output_start')
    write_outputs(out, result,write_result=False)
    result['timing']['total_wall_seconds'] = time.perf_counter()-started
    result['timing']['total_cpu_seconds'] = time.process_time()-cpu_started
    timing = result['timing']
    timing['coordinator_cpu_seconds'] = timing['total_cpu_seconds']
    child_cpu = executor.info.get('child_process_cpu_seconds') if executor else None
    timing['child_process_cpu_seconds'] = child_cpu
    timing['aggregate_cpu_seconds'] = (timing['coordinator_cpu_seconds']+child_cpu
                                        if child_cpu is not None else None)
    timing['cpu_measurement_note'] = ('Legacy total_cpu_seconds and cpu_seconds_before_output measure '
        'the coordinator only. aggregate_cpu_seconds includes reaped child processes where available; '
        'committed_worker_task_cpu_seconds excludes startup and discarded tasks. Execution overhead '
        'wall measurements are included in stage/total wall times and must not be added again.')
    atomic_json(out/'results.json', result)
    atomic_json(out/'checkpoint.json',dict(result,solutions=[],
        validated_solution_count=len(state.solutions),checkpoint_schema_version=2,
        solutions_storage='solution_records.jsonl'))
    phase('output_end')
    print(f"{mode}: {result['status']}; {len(state.solutions)} solutions; "
          f"{result['timing']['total_wall_seconds']:.2f}s; algorithm={settings.algorithm} -> {out}", flush=True)
    return result
