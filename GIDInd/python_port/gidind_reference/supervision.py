"""Per-run subprocess watchdog for CLI/benchmark jobs, independent of Pool locks.

The solver and candidate decisions run unchanged in the child. This supervisor
never retries a solver, uses a reference, or turns a killed run into no_solution.
"""
import faulthandler
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time


def _signal_group(process,sig):
    try:
        if os.name=='posix':os.killpg(process.pid,sig)
        elif sig==signal.SIGTERM:process.terminate()
        else:process.kill()
    except ProcessLookupError:
        pass


def _stop(process):
    _signal_group(process,signal.SIGTERM)
    try:process.wait(timeout=3.)
    except subprocess.TimeoutExpired:pass
    # Also kill descendants if the coordinator exited before its workers.
    _signal_group(process,signal.SIGKILL if os.name=='posix' else signal.SIGTERM)
    process.wait(timeout=5.)


def index_file(input_path,mode,output_dir,settings=None,*,_command=None,_grace=None):
    from .config import Settings
    from .runner import atomic_json,port_fingerprint,environment
    from .inputs import read_peaks
    settings=(settings or Settings()).validate()
    out=Path(output_dir).resolve();out.mkdir(parents=True,exist_ok=True)
    if any((out/name).exists() for name in ('results.json','checkpoint.json','solution_events.jsonl','supervision.json')):
        raise FileExistsError(f'Refusing to replace existing results: {out}')
    source=Path(input_path).resolve()
    config=out/'supervised_settings.json'
    atomic_json(config,settings.to_dict())
    grace=max(30.,min(180.,.1*settings.time_limit_seconds)) if _grace is None else _grace
    limit=settings.time_limit_seconds+grace
    command=_command or [sys.executable,'-m','gidind_reference.supervision','--child',
                        str(source),mode,str(out),str(config)]
    started=time.monotonic()
    process=subprocess.Popen(command,start_new_session=(os.name=='posix'))
    interrupted=False;timed_out=False;stopped=False
    previous_term=None
    def request_stop(signum,frame):
        nonlocal interrupted
        interrupted=True
    if os.name=='posix' and threading.current_thread() is threading.main_thread():
        previous_term=signal.signal(signal.SIGTERM,request_stop)
    try:
        while process.poll() is None:
            if interrupted:break
            if time.monotonic()-started>=limit:
                timed_out=True;break
            try:time.sleep(.1)
            except KeyboardInterrupt:interrupted=True
        if timed_out or interrupted:
            if interrupted:
                try:process.send_signal(signal.SIGINT)
                except ProcessLookupError:pass
                try:process.wait(timeout=min(10.,grace))
                except subprocess.TimeoutExpired:
                    _stop(process);stopped=True
            else:
                _stop(process);stopped=True
    finally:
        if previous_term is not None:signal.signal(signal.SIGTERM,previous_term)
        # A native coordinator crash can leave its Pool descendants alive.
        # Clean the isolated process group even when its leader already exited.
        if not stopped and (process.poll() is None or process.returncode!=0):_stop(process)
    elapsed=time.monotonic()-started
    supervision=dict(backend='isolated-subprocess-watchdog',child_pid=process.pid,
        child_exit_code=process.returncode,
        child_signal=(signal.Signals(-process.returncode).name
                      if os.name=='posix' and process.returncode<0 else None),
        diagnostic_policy='fatal-current-thread-only; no timed or signal-triggered stack walks',
        hard_wall_limit_seconds=limit,
        output_shutdown_allowance_seconds=grace,total_supervised_wall_seconds=elapsed,
        timed_out=timed_out,interrupted=interrupted,
        timing_scope='Includes child startup, numerical work, checkpoints, final output and process exit. '
            'Legacy total_wall_seconds and event timestamps retain their child-coordinator scope.')
    path=out/'results.json';result=None
    if path.exists():
        try:result=json.loads(path.read_text())
        except (OSError,ValueError):pass
    failed=(timed_out or interrupted or result is None or
            (process.returncode!=0 and result.get('status') not in ('failure','interrupted')))
    if failed:
        checkpoint=out/'checkpoint.json';saved={}
        if checkpoint.exists():
            try:saved=json.loads(checkpoint.read_text())
            except (OSError,ValueError):pass
        from . import IMPLEMENTATION,UPSTREAM_COMMIT,__version__
        result=result or dict(saved)
        result.update(schema_version=1,implementation=IMPLEMENTATION,
            upstream_repository='https://github.com/m-kainz/GIDInd',upstream_commit=UPSTREAM_COMMIT,
            original_executable_equivalence='unverified',port_version=__version__,
            port_source_sha256=port_fingerprint(),mode=mode,algorithm_variant=settings.algorithm,
            effective_settings=settings.to_dict(),input=read_peaks(source,mode,settings).provenance,
            environment=result.get('environment') or environment(),
            status='interrupted' if interrupted else 'failure',search_complete=False,
            parameter_uncertainty=None,
            error='Supervisor stopped a non-returning run' if timed_out else
                  'Run interrupted' if interrupted else f'Indexing child exited with code {process.returncode} without a normal result')
        # Preserve complete results if the child stalled only during process exit.
        result.setdefault('solutions',[])
        result.setdefault('counts',{})
        result.setdefault('partial_reasons',[])
        reason='supervisor_timeout' if timed_out else 'interrupted' if interrupted else 'child_process_failure'
        if reason not in result['partial_reasons']:result['partial_reasons'].append(reason)
        if not result.get('timing',{}).get('total_wall_seconds'):
            result['timing']=dict(total_wall_seconds=elapsed,aggregate_cpu_seconds=None,
                cpu_measurement_note='Unavailable after forced process-group shutdown; coordinator CPU is not total CPU.')
        result['salvage_note']='Complete validated fits remain recoverable from solution_records.jsonl. '
        result['salvage_note']+='The checkpoint and journals are execution evidence, not a completed search.'
        event=out/'solution_events.jsonl'
        if event.exists():
            result['solution_event_log']=event.name
            result['solution_event_log_sha256']=hashlib.sha256(event.read_bytes()).hexdigest()
    result['supervision']=supervision
    atomic_json(out/'supervision.json',supervision)
    atomic_json(path,result)
    if failed:
        atomic_json(out/'checkpoint.json',dict(result,solutions=[],checkpoint_schema_version=2,
            validated_solution_count=len(result['solutions']),solutions_storage='solution_records.jsonl'))
    return result


def _child(args):
    global _DIAGNOSTIC_STREAM
    from .config import Settings
    from .runner import index_file as run
    input_path,mode,output_dir,config=args
    out=Path(output_dir)
    # Retain the descriptor through process-exit finalizers. Never walk other
    # threads from a timer/signal: CPython issue #116008 documents crashes while
    # doing that without the GIL. ESRF's 3.12.3 coordinator exited with SIGSEGV
    # during the first timed dump. Causation remains unverified without a core.
    # Phase journals + the external watchdog provide progress/hang evidence.
    _DIAGNOSTIC_STREAM=(out/'diagnostic_stacks.log').open('a')
    faulthandler.enable(file=_DIAGNOSTIC_STREAM,all_threads=False)
    result=run(input_path,mode,output_dir,Settings.load(config))
    return 1 if result['status']=='failure' else 130 if result['status']=='interrupted' else 0


if __name__=='__main__':
    if len(sys.argv)!=6 or sys.argv[1]!='--child':raise SystemExit('Internal supervised indexing child')
    raise SystemExit(_child(sys.argv[2:]))
