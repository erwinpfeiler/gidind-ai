"""Auditable search-work denominators and explicitly conditional extrapolations.

No timings here constitute a worst-case runtime bound or a guarantee of indexing.
Counts refer to this port's seed enumeration, not all possible peak assignments.
"""
from math import comb, prod
from itertools import combinations


def _phase(name, unit, count_key, total, total_scope, result, stages):
    attempted = int(result.get('counts', {}).get(count_key, 0))
    wall = stages.get(name)
    mean = wall/attempted if wall is not None and attempted else None
    estimate = (mean*total if total is not None and mean is not None else
                wall if total == 0 and wall is not None else None)
    if total is not None and attempted == total and wall is not None:
        # Exhausted integer work has exactly zero work remaining. Avoid a
        # positive roundoff remainder from wall/attempted*total - wall.
        estimate = wall
    return {
        'stage': name, 'work_unit': unit, 'count_key': count_key,
        'attempted_units': attempted, 'possible_units': total, 'possible_units_scope': total_scope,
        'enumeration_fraction': attempted/total if total else (1. if total == 0 else None),
        'observed_stage_wall_seconds': wall,
        'amortized_wall_seconds_per_attempt': mean,
        'attempts_per_wall_second': attempted/wall if wall and attempted else None,
        'estimated_exhaustive_stage_wall_seconds': estimate,
        'estimated_remaining_stage_wall_seconds': max(0., estimate-wall) if estimate is not None else None,
    }


def search_work(result, stage_seconds=None):
    settings = result['effective_settings']
    counts = result.get('counts', {})
    stages = stage_seconds if stage_seconds is not None else result.get('timing', {}).get('stage_seconds', {})
    seeds = len(result.get('input', {}).get('seed_coordinates', []))
    reasons = result.get('partial_reasons', [])
    report = {
        'metric_version': 2, 'workers': settings.get('workers', 1),
        'algorithm_variant': settings.get('algorithm', 'upstream'),
        'exhaustion_observed': bool(result.get('search_complete', False)),
        'guarantees_correct_indexing': False, 'is_worst_case_runtime_bound': False,
        'estimated_exhaustive_search_wall_seconds': None,
        'estimated_remaining_search_wall_seconds': None,
        'estimate_scope': 'Native seed search only; excludes input setup, final correction/reduction/discrete projection and output writing.',
        'interpretation': 'Measured stage wall / committed attempted systems, including singular/rejected systems. '
            'Wall includes pool and checkpoint overhead and any speculative/discarded work. '
            'This is aggregate throughput at the recorded worker count, not individual solve latency. '
            'Linear extrapolation assumes the observed rejection/refit mix stays representative; '
            'ordered search prefixes may be unrepresentative. Time/candidate/solution caps can prevent exhaustion. '
            'Exhaustion covers only the configured seed enumeration and algorithmic filters; it does not prove '
            'all physical cells or peak assignments were considered, or guarantee a correct solution.',
        'phases': [],
    }
    if seeds == 0:
        report['estimate_unavailable_reason'] = 'Eligible seed selection was not completed.'
        return report
    if result['mode'] == 'without-specular':
        grid = prod(2*n+1 for n in settings['without_specular_seed_hkl'])
        total = grid**4
        phase = _phase('without_specular_triclinic_seeds',
                       'Ordered assignment of four HKL triples to the native seed system',
                       'without_specular_hkl_quadruples', total, 'exact_configured_seed_domain', result, stages)
        phase['domain_formula'] = '((2*h_max+1)*(2*k_max+1)*(2*l_max+1))**4; includes zero/singular combinations'
        phase['hkl_triples_per_seed'] = grid
        phase['seed_peaks'] = seeds
        phase['four_peak_combinations_per_attempt'] = comb(seeds-2, 4)
        phase['inner_work_note'] = 'Each attempt also selects fifth/sixth HKLs and may index/refit every retained peak; those costs vary with rejection gates.'
        if settings.get('algorithm') == 'corrected':
            seed_coords = result['input']['seed_coordinates']
            eligible = sum(any(seed_coords[i][1] > 0 for i in indices)
                for indices in combinations(range(seeds-2),4))
            phase['excluded_all_in_plane_seed_combinations'] = comb(seeds-2,4)-eligible
            phase['four_peak_combinations_per_attempt'] = eligible
            phase['tail_hkl_candidates_per_peak'] = settings['tail_hkl_candidates']
            phase['tail_systems_evaluated'] = counts.get('corrected_tail_systems',0)
            phase['tail_search_scope'] = ('Best observed tail-peak pair; up to K nonzero HKLs per peak '
                'and K squared tail systems per usable quartet. This is not exhaustive six-HKL enumeration.')
        report['phases'].append(phase)
        full_domains_known = True
    else:
        uv = (2*settings['specular_uv']+1)**2
        triples = comb(seeds, 3)
        first_total = uv*triples*(2*settings['specular_seed_hk']+1)**6
        first = _phase('specular_abgamma', 'One (u,v), seed-peak triple and HK sextuple system',
                       'specular_stage1_systems', first_total, 'exact_configured_seed_domain', result, stages)
        first['domain_formula'] = '(2*uv_max+1)**2 * C(actual_seed_peaks,3) * (2*seed_hk_max+1)**6'
        first['seed_peaks'] = seeds
        per_row = settings['redundancy_hk']**3*(2*settings['specular_l']+1)**3*(2*settings['specular_w']+1)
        rows = int(counts.get('specular_abg_candidates', 0))
        stage1_complete = (first['attempted_units'] == first_total and
                           result.get('current_stage') != 'specular_abgamma' and
                           not any(r in reasons for r in ('abg_candidate_limit', 'specular_stage1_time_limit')))
        # All ABG rows share the same observed-qz eligibility check.
        skipped = bool(counts.get('insufficient_distinct_qz', 0))
        retained_total = 0 if skipped else rows*per_row
        second = _phase('specular_alphabetac', 'One HK-redundancy/L/w system for a retained ABG row',
                        'specular_stage2_systems', retained_total,
                        'exact_for_retained_abg_rows_only' if not stage1_complete else 'exact_for_completed_stage1_output',
                        result, stages)
        second.update(retained_abg_rows=rows, systems_per_eligible_abg_row=per_row,
                      domain_formula='retained_abg_rows * redundancy_hk**3 * (2*l_max+1)**3 * (2*w_max+1); zero if observed qz eligibility fails',
                      upstream_stage_complete=stage1_complete)
        report['phases'].extend((first, second))
        full_domains_known = stage1_complete
    estimates = [p['estimated_exhaustive_stage_wall_seconds'] for p in report['phases']]
    if full_domains_known and all(e is not None for e in estimates):
        report['estimated_exhaustive_search_wall_seconds'] = sum(estimates)
        report['estimated_remaining_search_wall_seconds'] = sum(p['estimated_remaining_stage_wall_seconds'] for p in report['phases'])
    else:
        report['estimate_unavailable_reason'] = ('Full specular stage-2 domain is unknown until stage 1 finishes without truncating ABG output.'
            if not full_domains_known else 'No measured throughput for at least one required search stage.')
    return report
