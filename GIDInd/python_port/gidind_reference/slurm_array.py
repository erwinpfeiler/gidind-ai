"""Prepare immutable unique-case plans, run one case, and collect eight entries.

No scheduler calls are made here. Shell launchers submit the prepared work.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
from dataclasses import replace
from .batch import prepare_input, reused_result, copy_run_artifacts
from .config import Settings
from .runner import atomic_json, port_fingerprint, write_outputs
from .supervision import index_file


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(input_dir, run_dir, settings):
    settings.validate()
    source, out = Path(input_dir).resolve(), Path(run_dir).resolve()
    manifest = source/'manifest.json'
    data = json.loads(manifest.read_text())
    entries, tasks, cache = [], [], {}
    for case in data['cases']:
        if not isinstance(case['id'], str) or Path(case['id']).name != case['id'] or case['id'] in ('.','..'):
            raise ValueError('Case ID must be a basename')
        for variant, name in (('original',case['filename']), ('without_specular',case['without_specular']['filename'])):
            path, mode, unavailable, peaks, key = prepare_input(source,case,variant,name,settings)
            from .inputs import seed_coordinates
            seed_coordinates(peaks,mode,settings)
            output_name=f"{case['id']}_{variant}"
            if any(e['output_name']==output_name for e in entries):
                raise ValueError(f'Duplicate case output: {output_name}')
            if key not in cache:
                cache[key]=len(tasks)
                tasks.append({'task_id':len(tasks),'case':case['id'],'variant':variant,
                              'mode':mode,'input_path':str(path),'input_sha256':_sha(path),
                              'output_name':output_name})
            entries.append({'case':case['id'],'variant':variant,'mode':mode,
                            'input_path':str(path),'input_sha256':_sha(path),
                            'output_name':output_name,'task_id':cache[key],
                            'specular_comparison_unavailable':unavailable})
    plan={'schema_version':1,'input_dir':str(source),'manifest':str(manifest),
          'manifest_sha256':_sha(manifest),'effective_settings':settings.to_dict(),
          'port_source_sha256':port_fingerprint(),'tasks':tasks,'entries':entries}
    if not tasks:
        raise ValueError('No benchmark tasks')
    out.mkdir(parents=True,exist_ok=False)
    atomic_json(out/'array_plan.json',plan)
    for task in tasks:
        print(f"Task {task['task_id']}: {task['case']} / {task['variant']} / {task['mode']}")
    print(f"Prepared {len(tasks)} unique tasks for {len(entries)} entries: {out}")
    return plan


def prepare_ladder(input_dir,run_dir,settings,budgets):
    from .budgets import validate_budgets,budget_name
    budgets=validate_budgets(budgets)
    root=Path(run_dir).resolve();root.mkdir(parents=True,exist_ok=False)
    runs=[]
    for budget in budgets:
        path=root/budget_name(budget)
        plan=prepare(input_dir,path,replace(settings,time_limit_seconds=budget))
        runs.append(dict(budget_seconds=budget,run_dir=str(path)))
    plan=dict(plan,kind='budget_ladder',budget_runs=runs)
    atomic_json(root/'array_plan.json',plan)
    print(f'Prepared budget ladder: {budgets}; budgets stay sequential within each array task')
    return plan


def load_plan(run_dir):
    root=Path(run_dir).resolve()
    plan=json.loads((root/'array_plan.json').read_text())
    if port_fingerprint()!=plan['port_source_sha256']:
        raise ValueError('Port code changed after array preparation; keep the checkout unchanged while the array runs.')
    if _sha(plan['manifest'])!=plan['manifest_sha256']:
        raise ValueError('Input manifest changed after array preparation')
    return root,plan,Settings(**plan['effective_settings']).validate()


def run_task(run_dir, task_id):
    root,plan,settings=load_plan(run_dir)
    if task_id < 0 or task_id >= len(plan['tasks']):
        raise ValueError('Array task ID is outside prepared task range')
    allocated=os.environ.get('SLURM_CPUS_PER_TASK')
    if allocated is not None and int(allocated)!=settings.workers:
        raise ValueError(f'Plan requires {settings.workers} workers, allocation supplies {allocated} CPUs')
    if plan.get('kind')=='budget_ladder':
        results=[]
        for run in plan['budget_runs']:
            result=run_task(run['run_dir'],task_id)
            results.append(result)
            if result['status'] in ('failure','interrupted'):break
        statuses=[r['status'] for r in results]
        status=next((s for s in ('failure','interrupted','time_limited','search_limited') if s in statuses),'completed')
        return dict(status=status,budget_statuses=statuses)
    task=plan['tasks'][task_id]
    if _sha(task['input_path'])!=task['input_sha256']:
        raise ValueError('Task input changed after array preparation')
    out=root/task['output_name']
    # Exclusive directory creation prevents duplicate/requeued tasks overwriting work.
    out.mkdir(exist_ok=False)
    receipt={'task_id':task_id,'plan':str(root/'array_plan.json'),
             'slurm':{k:os.environ.get(k) for k in
                ('SLURM_JOB_ID','SLURM_ARRAY_JOB_ID','SLURM_ARRAY_TASK_ID','SLURM_CPUS_PER_TASK','SLURMD_NODENAME')}}
    atomic_json(out/'array_task.json',receipt)
    result=index_file(task['input_path'],task['mode'],out,settings)
    return result


def _validate_result(result,task,plan):
    if (result.get('mode')!=task['mode'] or result.get('input',{}).get('sha256')!=task['input_sha256'] or
        result.get('effective_settings')!=plan['effective_settings'] or
        result.get('port_source_sha256')!=plan['port_source_sha256']):
        raise ValueError(f"Result does not match prepared task: {task['output_name']}")


def collect(run_dir):
    root,plan,settings=load_plan(run_dir)
    if plan.get('kind')=='budget_ladder':
        from .budgets import ladder_summary
        runs=[(run['budget_seconds'],run['run_dir'],collect(run['run_dir']))
              for run in plan['budget_runs']]
        summary=ladder_summary(runs,settings)
        atomic_json(root/'benchmark.json',summary)
        print(f"Ladder collection: {summary['status']} -> {root/'benchmark.json'}")
        return summary
    summary={'schema_version':1,'manifest':plan['manifest'],'manifest_sha256':plan['manifest_sha256'],
             'effective_settings':plan['effective_settings'],'array_plan':str(root/'array_plan.json'),
             'unique_computations':len(plan['tasks']),'jobs':[],'status':'finished',
             'timing_note':'Independent array tasks may overlap and run on different CPU models. '
                           'Per-case measured timings are retained; collection time is not indexing time.'}
    missing=False;failures=False;interrupted=False
    for entry in plan['entries']:
        task=plan['tasks'][entry['task_id']]
        original_dir=root/task['output_name'];result_path=original_dir/'results.json'
        alias=entry['output_name']!=task['output_name']
        job={k:entry[k] for k in ('case','variant','mode','task_id','specular_comparison_unavailable')}
        job['reused']=alias
        if not result_path.is_file():
            missing=True
            job.update(status='missing_result',num_solutions=None,results=None,
                       canonical_results=str(result_path),
                       checkpoint=str(original_dir/'checkpoint.json') if (original_dir/'checkpoint.json').exists() else None)
        else:
            if _sha(entry['input_path'])!=entry['input_sha256']:
                raise ValueError('Input changed after array preparation')
            result=json.loads(result_path.read_text())
            _validate_result(result,task,plan)
            if alias:
                source=Path(entry['input_path'])
                if _sha(source)!=entry['input_sha256']:
                    raise ValueError('Alias input changed after array preparation')
                from .inputs import read_peaks
                peaks=read_peaks(source,entry['mode'],settings)
                dest=root/entry['output_name']
                canonical_sha=_sha(result_path)
                if dest.exists():
                    prior=json.loads((dest/'results.json').read_text())
                    _validate_result(prior,entry,plan)
                    if (prior.get('deduplication',{}).get('canonical_results_sha256')!=canonical_sha or
                        prior.get('deduplication',{}).get('reused_from')!=str(result_path) or
                        prior.get('input',{}).get('sha256')!=entry['input_sha256']):
                        raise ValueError(f'Refusing to replace an unrelated or stale alias: {dest}')
                    result=prior
                else:
                    result=reused_result(result,original_dir,peaks,entry['mode'],settings)
                    result['deduplication']['canonical_results_sha256']=canonical_sha
                    dest.mkdir(exist_ok=False)
                    copy_run_artifacts(original_dir,dest)
                    write_outputs(dest,result)
                result_path=dest/'results.json'
            job.update(status=result['status'],num_solutions=len(result['solutions']),
                       results=str(result_path),search_work=result.get('search_work'))
            failures |= result['status']=='failure'
            interrupted |= result['status']=='interrupted'
        summary['jobs'].append(job)
    summary['status']='incomplete' if missing else 'finished_with_failures' if failures else 'interrupted' if interrupted else 'finished'
    atomic_json(root/'benchmark.json',summary)
    print(f"Array collection: {summary['status']}; {len(summary['jobs'])} entries, {len(plan['tasks'])} unique tasks -> {root/'benchmark.json'}")
    return summary


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--input-dir',type=Path,required=True)
    p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--workers',type=int,default=32);p.add_argument('--time-limit-seconds',type=float,default=7200.)
    p.add_argument('--algorithm',choices=('corrected','upstream'))
    p.add_argument('--budgets-seconds',type=float,nargs='+')
    p=sub.add_parser('run');p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--task-id',type=int,required=True)
    p=sub.add_parser('collect');p.add_argument('--run-dir',type=Path,required=True)
    args=parser.parse_args(argv)
    try:
        if args.command=='prepare':
            settings=Settings.load(args.config,workers=args.workers,
                time_limit_seconds=args.time_limit_seconds,algorithm=args.algorithm)
            if args.budgets_seconds:
                prepare_ladder(args.input_dir,args.run_dir,settings,args.budgets_seconds)
            else:prepare(args.input_dir,args.run_dir,settings)
            return 0
        if args.command=='run':
            result=run_task(args.run_dir,args.task_id)
            return 1 if result['status']=='failure' else 130 if result['status']=='interrupted' else 0
        result=collect(args.run_dir)
        return 0 if result['status']=='finished' else 1
    except (ValueError,TypeError,OSError) as exc:
        parser.exit(1,f'{type(exc).__name__}: {exc}\n')


if __name__=='__main__':
    raise SystemExit(main())
