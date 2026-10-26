"""Independent fixed-budget runs; no reference-dependent stopping or reuse."""
from dataclasses import replace
import math
from pathlib import Path

DEFAULT_BUDGETS = (120.,600.,3600.,7200.)


def validate_budgets(values):
    values=list(values)
    if (not values or any(isinstance(v,bool) or not isinstance(v,(int,float))
            or not math.isfinite(v) or v <= 0 for v in values)
            or values != sorted(set(values))):
        raise ValueError('Budgets must be positive, finite, unique and increasing')
    return [float(v) for v in values]


def budget_name(value):
    return 'budget_'+format(float(value),'.17g').replace('.','p')+'s'


def ladder_summary(runs, settings):
    jobs=[]
    for budget,path,summary in runs:
        jobs.extend(dict(job,budget_seconds=budget) for job in summary['jobs'])
    statuses=[s['status'] for _,_,s in runs]
    status=('incomplete' if 'incomplete' in statuses else
            'finished_with_failures' if 'finished_with_failures' in statuses else
            'interrupted' if 'interrupted' in statuses else 'finished')
    return dict(schema_version=1,kind='budget_ladder',status=status,
        effective_settings=settings.to_dict(),budgets_seconds=[b for b,_,_ in runs],
        effective_settings_scope='Base settings; each budget overrides time_limit_seconds in its own results.',
        unique_computations=sum(s['unique_computations'] for _,_,s in runs),jobs=jobs,
        budget_runs=[dict(budget_seconds=b,benchmark=str(Path(p)/'benchmark.json'),status=s['status'])
                     for b,p,s in runs],
        timing_note='Fresh independent runs at each budget. No reference-based stopping. '
            'Budgets are not checkpoints of one continuous run; timings are per computation.')


def benchmark_ladder(input_dir, output_dir, settings, budgets):
    from .batch import benchmark
    from .runner import atomic_json
    budgets=validate_budgets(budgets)
    root=Path(output_dir).resolve();root.mkdir(parents=True,exist_ok=False)
    runs=[]
    for budget in budgets:
        path=root/budget_name(budget)
        summary=benchmark(input_dir,path,replace(settings,time_limit_seconds=budget))
        runs.append((budget,path,summary))
        result=ladder_summary(runs,settings)
        atomic_json(root/'benchmark.json',result)
        if summary['status']=='interrupted':break
    return result
