"""Post-run reference evaluation. Never imported by the search or its workers."""
from bisect import bisect_left, insort
import csv
import hashlib
from itertools import product
import json
import math
from pathlib import Path
import time
import numpy as np

CELL_KEYS=('a','b','c','alpha','beta','gamma')


def cell_geometry(cell):
    values=np.asarray([cell[k] for k in CELL_KEYS] if isinstance(cell,dict) else cell,dtype=float)
    if values.shape!=(6,) or not np.isfinite(values).all() or np.any(values[:3]<=0) or np.any(values[3:]<=0) or np.any(values[3:]>=180):
        raise ValueError('Invalid cell: require three positive lengths and three angles in degrees')
    a,b,c=values[:3];al,be,ga=np.deg2rad(values[3:])
    metric=np.array([[a*a,a*b*np.cos(ga),a*c*np.cos(be)],
                     [a*b*np.cos(ga),b*b,b*c*np.cos(al)],
                     [a*c*np.cos(be),b*c*np.cos(al),c*c]])
    try:np.linalg.cholesky(metric)
    except np.linalg.LinAlgError as exc:raise ValueError('Cell metric must be positive definite') from exc
    return values,metric


def plane_vector(plane):
    if (not isinstance(plane,list) or len(plane)!=3 or
            any(type(v) is not int for v in plane) or not any(plane)):
        raise ValueError('contact_plane must contain three nonzero-together integer coordinates')
    return np.asarray(plane,dtype=float)


def validate_criteria(criteria):
    for key in ('length_relative_tolerance','volume_relative_tolerance'):
        value=criteria[key]
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not 0<value<1:
            raise ValueError(f'{key} must lie strictly between 0 and 1')
    for key in ('angle_absolute_tolerance_degrees','orientation_tolerance_degrees'):
        value=criteria[key]
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not 0<value<90:
            raise ValueError(f'{key} must lie strictly between 0 and 90 degrees')
    if type(criteria['basis_limit']) is not int or criteria['basis_limit']<1:
        raise ValueError('basis_limit must be a positive integer')
    cuts=criteria['rmse_cutoffs']
    if len(cuts)!=3 or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<=0 for v in cuts):
        raise ValueError('rmse_cutoffs must contain three positive finite values for qxy,qz,|q|')
    return criteria


def match_reference(solution,reference,criteria):
    """Bounded unimodular direct-basis matching plus reciprocal-plane matching.

    No solver geometry/reduction code is reused. A miss means not matched within
    the declared comparison domain, not a proof of crystallographic inequivalence.
    """
    candidate,dc=cell_geometry(solution['cell'])
    ref,dr=cell_geometry(reference['cell'])
    pc=plane_vector(solution['contact_plane']);pr=plane_vector(reference['contact_plane'])
    errors=np.array([solution['rmse'][k] for k in ('q_xy','q_z','q_xyz')],dtype=float)
    if not np.isfinite(errors).all() or np.any(errors<0):raise ValueError('Invalid solution residuals')
    if np.any(errors>criteria['rmse_cutoffs']):return dict(matched=False,reason='residual_cutoff')
    if abs(np.sqrt(np.linalg.det(dc)/np.linalg.det(dr))-1)>criteria['volume_relative_tolerance']:
        return dict(matched=False,reason='volume_tolerance')
    reciprocal=np.linalg.inv(dr)
    length_tol=criteria['length_relative_tolerance'];angle_tol=criteria['angle_absolute_tolerance_degrees']
    def check(transform):
        metric=transform.T@dr@transform
        lengths=np.sqrt(np.diag(metric))
        angles=np.rad2deg([np.arccos(np.clip(metric[j,k]/(lengths[j]*lengths[k]),-1,1))
                          for j,k in ((1,2),(0,2),(0,1))])
        if np.any(np.abs(candidate[:3]/lengths-1)>length_tol+1e-12) or np.any(np.abs(candidate[3:]-angles)>angle_tol+1e-10):return None
        p=np.linalg.solve(transform.T,pc)
        cosine=abs(pr@reciprocal@p)/np.sqrt((pr@reciprocal@pr)*(p@reciprocal@p))
        angle=float(np.rad2deg(np.arccos(np.clip(cosine,-1,1))))
        if angle>criteria['orientation_tolerance_degrees']+1e-10:return None
        return dict(matched=True,basis_transform=transform.astype(int).tolist(),orientation_error_degrees=angle)
    direct=check(np.eye(3,dtype=int))
    if direct is not None:return direct
    bound=criteria['basis_limit']
    vectors=np.asarray([v for v in product(range(-bound,bound+1),repeat=3) if any(v)],dtype=int)
    lengths=np.sqrt(np.einsum('ni,ij,nj->n',vectors,dr,vectors))
    choices=[vectors[np.abs(candidate[i]/lengths-1)<=length_tol+1e-12] for i in range(3)]
    for a in choices[0]:
        for b in choices[1]:
            na=np.sqrt(a@dr@a);nb=np.sqrt(b@dr@b)
            gamma=np.rad2deg(np.arccos(np.clip((a@dr@b)/(na*nb),-1,1)))
            if abs(gamma-candidate[5])>angle_tol+1e-10:continue
            for c in choices[2]:
                transform=np.column_stack((a,b,c))
                if abs(round(float(np.linalg.det(transform))))!=1:continue
                result=check(transform)
                if result is not None:return result
    return dict(matched=False,reason='not_matched_in_bounded_unimodular_domain')


def rank_key(solution):
    return (solution['rmse']['q_xyz'],solution['rmse']['q_z'],solution['rmse']['q_xy'],
            tuple(solution['contact_plane']),tuple(solution['cell'][k] for k in CELL_KEYS))


def read_events(folder,result):
    name=result.get('solution_event_log')
    if not name:return None,False
    if Path(name).name!=name:raise ValueError('Invalid solution-event filename')
    path=Path(folder)/name
    if not path.exists():return None,False
    raw=path.read_bytes()
    if result.get('solution_event_log_sha256') and hashlib.sha256(raw).hexdigest()!=result['solution_event_log_sha256']:
        raise ValueError('Solution-event checksum mismatch')
    events=[];truncated=False;previous=-1.
    lines=raw.splitlines()
    for i,line in enumerate(lines):
        try:event=json.loads(line)
        except json.JSONDecodeError:
            if i!=len(lines)-1 or raw.endswith(b'\n'):raise
            truncated=True;break
        if (event['sequence']!=len(events) or not math.isfinite(event['wall_seconds']) or
                event['wall_seconds']<previous or event['wall_seconds']<0):
            raise ValueError('Invalid event ordering/timestamps')
        previous=event['wall_seconds'];events.append(event)
    if result.get('solution_event_count') is not None and len(events)!=result['solution_event_count']:
        truncated=True
    total=result.get('timing',{}).get('total_wall_seconds')
    if total is not None and events and events[-1]['wall_seconds']>total+1e-6:
        raise ValueError('Solution event occurs after the recorded run end')
    return events,truncated


def evaluate_result(result,folder,reference,criteria,top_k):
    total=result.get('timing',{}).get('total_wall_seconds')
    report=dict(status=result['status'],total_wall_seconds=total,
        aggregate_cpu_seconds=result.get('timing',{}).get('aggregate_cpu_seconds'),
        algorithm_variant=result.get('algorithm_variant','upstream'),
        input_sha256=result.get('input',{}).get('sha256'),
        port_source_sha256=result.get('port_source_sha256'),
        first_any_validated_seconds=result.get('timing',{}).get('first_validated_solution_wall_seconds'),
        first_correct_validated_seconds=None,first_correct_top_k_seconds=None,
        first_correct_observed=None,first_correct_top_k_observed=None,censoring_wall_seconds=None,
        retrospective_correct_candidate_discovery_seconds=None,
        final_best_match_rank=None,final_top_k_success=None,any_final_match=None,
        work_counts=result.get('counts',{}),
        timing_scope='Actual post-search validation events; not a streaming search convergence time. '
            'Candidate discovery is retrospective coordinator-commit time and excludes later validation.',
        reference_used_by_search=False)
    if reference is None:
        report.update(reference_status='reference_unavailable',timeline_status='not_evaluated')
        return report
    cache={}
    def matched(solution):
        key=json.dumps(solution,sort_keys=True)
        if key not in cache:cache[key]=match_reference(solution,reference,criteria)['matched']
        return cache[key]
    final=sorted(result.get('solutions',[]),key=rank_key)
    ranks=[i+1 for i,s in enumerate(final) if matched(s)]
    report.update(reference_status='evaluated',
                  reference_orientation_status=reference.get('orientation_status','unspecified'),
                  any_final_match=bool(ranks),
                  final_best_match_rank=min(ranks) if ranks else None,
                  final_top_k_success=any(r<=top_k for r in ranks))
    events,truncated=read_events(folder,result)
    if events is None:
        report['timeline_status']='unavailable_for_historical_result'
        return report
    report['timeline_status']='incomplete' if truncated else 'recorded'
    active={};keys={};ranked=[];first=None;first_top=None;births=[]
    for event in events:
        slot=event['solution_slot'];solution=event['solution'];key=(*rank_key(solution),slot)
        if slot in keys:
            index=bisect_left(ranked,keys[slot]);ranked.pop(index)
        keys[slot]=key;insort(ranked,key);active[slot]=matched(solution)
        if active[slot]:
            if first is None:first=event['wall_seconds']
            timing=event.get('candidate_timing')
            if timing and timing.get('discovered_wall_seconds') is not None:
                birth=timing['discovered_wall_seconds']
                if not math.isfinite(birth) or not 0<=birth<=event['wall_seconds']:raise ValueError('Invalid candidate discovery time')
                births.append(birth)
        if first_top is None and any(active[k[-1]] for k in ranked[:top_k]):first_top=event['wall_seconds']
    report.update(first_correct_validated_seconds=first,first_correct_top_k_seconds=first_top,
        retrospective_correct_candidate_discovery_seconds=min(births) if births else None,
        first_correct_observed=first is not None,first_correct_top_k_observed=first_top is not None,
        censoring_wall_seconds=total if result['status'] not in ('failure','interrupted') else None,
        censoring_note='Missing success times are not observed by run end; failures/interruptions are separate technical outcomes.')
    return report


def evaluate_run(run_dir,references_path,output_dir,*,top_k=None,case=None):
    from .runner import atomic_json
    started=time.perf_counter();root=Path(run_dir).resolve()
    raw=Path(references_path).read_bytes();config=json.loads(raw)
    criteria=validate_criteria(config['criteria']);top_k=config.get('top_k',5) if top_k is None else top_k
    if type(top_k) is not int or top_k<1:raise ValueError('top_k must be a positive integer')
    for reference in config['cases'].values():
        cell_geometry(reference['cell']);plane_vector(reference['contact_plane'])
        validate_criteria(dict(criteria,**reference.get('criteria',{})))
        if not isinstance(reference.get('source'),str) or not reference['source'].strip():
            raise ValueError('Each reference needs its provenance in source')
    if (root/'benchmark.json').exists():
        summary=json.loads((root/'benchmark.json').read_text())
        if summary['status']=='running':raise ValueError('Evaluate after benchmark collection completes')
        jobs=summary['jobs']
    else:
        if not case:raise ValueError('--case is required for a single results.json directory')
        jobs=[dict(case=case,variant='single',results=str(root/'results.json'),reused=False)]
    rows=[]
    for job in jobs:
        row={k:job.get(k) for k in ('case','variant','mode','budget_seconds','reused')}
        if not job.get('results'):
            rows.append(dict(row,status='missing_result',reference_status='not_evaluated'));continue
        path=Path(job['results']);result=json.loads(path.read_text())
        reference=config['cases'].get(job['case'])
        effective_criteria=dict(criteria,**reference.get('criteria',{})) if reference else criteria
        evaluation=evaluate_result(result,path.parent,reference,effective_criteria,top_k)
        rows.append(dict(row,mode=result['mode'],results=str(path),reference_criteria=effective_criteria,
                         reference_criteria_status=config.get('criteria_status','unspecified'),
                         **evaluation))
        rows[-1]['reused']=bool(job.get('reused') or 'deduplication' in result)
    groups={}
    for row in rows:
        key=(row['case'],row['variant'],row.get('mode'),row.get('algorithm_variant'))
        groups.setdefault(key,[]).append(row)
    minimum=[]
    for key,points in groups.items():
        successes=[p['budget_seconds'] for p in points if p.get('budget_seconds') is not None
            and p.get('status') in ('completed','time_limited','search_limited')
            and p.get('final_top_k_success')]
        minimum.append(dict(zip(('case','variant','mode','algorithm_variant'),key),
            smallest_tested_successful_budget_seconds=min(successes) if successes else None,
            interpretation='Smallest successful tested budget, not the first-passage time of a continuous run.'))
    missing_references=sorted({r['case'] for r in rows if r.get('reference_status')=='reference_unavailable'})
    missing_results=sum(r['status']=='missing_result' for r in rows)
    report=dict(schema_version=1,status='incomplete' if missing_references or missing_results else 'finished',
        missing_reference_cases=missing_references,missing_results=missing_results,
        reference_sha256=hashlib.sha256(raw).hexdigest(),reference_config=config,
        evaluator_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        top_k=top_k,reference_used_by_search=False,jobs=rows,minimum_tested_budgets=minimum,
        evaluation_wall_seconds=time.perf_counter()-started,
        matching_scope='Bounded unimodular basis equivalence, reciprocal-plane direction up to sign, and declared residual cutoffs.')
    out=Path(output_dir).resolve();out.mkdir(parents=True,exist_ok=False)
    atomic_json(out/'evaluation.json',report)
    fields=['case','variant','mode','algorithm_variant','budget_seconds','reused','status','reference_status',
        'reference_orientation_status','reference_criteria_status',
        'timeline_status','total_wall_seconds','aggregate_cpu_seconds','first_any_validated_seconds',
        'first_correct_validated_seconds','first_correct_top_k_seconds',
        'retrospective_correct_candidate_discovery_seconds','final_best_match_rank','final_top_k_success']
    with (out/'evaluation.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore',lineterminator='\n')
        writer.writeheader();writer.writerows(rows)
    print(f"Reference evaluation: {report['status']}; {len(rows)} entries; "
          f"{len(missing_references)} missing reference cases -> {out/'evaluation.json'}")
    return report
