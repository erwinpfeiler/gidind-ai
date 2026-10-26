"""Work-count formulas and honest conditional runtime extrapolation."""
from pathlib import Path
import csv
import json
import tempfile
import unittest
from gidind_reference.config import Settings
from gidind_reference.runner import write_outputs
from gidind_reference.work_metrics import search_work


def result(mode='without-specular', seeds=8, **changes):
    value={'effective_settings':Settings().to_dict(), 'input':{'seed_coordinates':[[1.,1.]]*seeds},
           'counts':{},'mode':mode,'partial_reasons':[],'search_complete':False,
           'timing':{'stage_seconds':{}}}
    value.update(changes)
    return value


class WorkMetricTests(unittest.TestCase):
    def test_exhausted_phase_has_exact_zero_remaining_despite_float_roundoff(self):
        for wall in (.1,.3,33.28184844,63.2):
            data=result(counts={'without_specular_hkl_quadruples':45**4},
                timing={'stage_seconds':{'without_specular_triclinic_seeds':wall}})
            phase=search_work(data)['phases'][0]
            self.assertEqual(phase['estimated_exhaustive_stage_wall_seconds'],wall)
            self.assertEqual(phase['estimated_remaining_stage_wall_seconds'],0.)

    def test_corrected_work_separates_tail_systems_and_excludes_in_plane_quartets(self):
        data=result(counts={'without_specular_hkl_quadruples':1000,'corrected_tail_systems':6400})
        data['input']['seed_coordinates'][:4]=[[1.,0.]]*4
        report=search_work(data);phase=report['phases'][0]
        self.assertEqual(report['algorithm_variant'],'corrected')
        self.assertEqual(phase['possible_units'],45**4)
        self.assertEqual(phase['four_peak_combinations_per_attempt'],14)
        self.assertEqual(phase['tail_systems_evaluated'],6400)
        self.assertEqual(phase['tail_hkl_candidates_per_peak'],8)

    def test_no_specular_exact_domain_and_extrapolation(self):
        data=result(counts={'without_specular_hkl_quadruples':1000},
                    timing={'stage_seconds':{'without_specular_triclinic_seeds':2.}})
        report=search_work(data);phase=report['phases'][0]
        self.assertEqual(phase['possible_units'],45**4)
        self.assertEqual(phase['four_peak_combinations_per_attempt'],15)
        self.assertEqual(phase['amortized_wall_seconds_per_attempt'],.002)
        self.assertEqual(phase['estimated_exhaustive_stage_wall_seconds'],8201.25)
        self.assertEqual(report['estimated_remaining_search_wall_seconds'],8199.25)
        self.assertFalse(report['is_worst_case_runtime_bound'])
        self.assertFalse(report['guarantees_correct_indexing'])
        self.assertFalse(report['exhaustion_observed'])

    def test_actual_seed_count_and_specular_stage_domains(self):
        data=result('specular',seeds=4)
        total=25*4*7**6
        data['counts']={'specular_stage1_systems':total, 'specular_abg_candidates':3,
                        'specular_stage2_systems':1000}
        data['timing']['stage_seconds']={'specular_abgamma':10.,'specular_alphabetac':2.}
        report=search_work(data);first,second=report['phases']
        self.assertEqual(first['possible_units'],11764900)
        self.assertEqual(second['possible_units'],3*4**3*13**3*7)
        self.assertTrue(second['upstream_stage_complete'])
        self.assertAlmostEqual(report['estimated_exhaustive_search_wall_seconds'],10.+.002*second['possible_units'])
        data['input']['seed_coordinates'].pop()
        data['counts']['specular_stage1_systems']=25*7**6
        self.assertEqual(search_work(data)['phases'][0]['possible_units'],2941225)

    def test_specular_partial_upstream_never_invents_full_downstream_size(self):
        for reason in ('specular_stage1_time_limit','abg_candidate_limit',None):
            data=result('specular',seeds=3,partial_reasons=[reason] if reason else [],
                        current_stage='specular_abgamma' if reason is None else 'writing_outputs',
                        counts={'specular_stage1_systems':25*7**6,'specular_abg_candidates':2,'specular_stage2_systems':100},
                        timing={'stage_seconds':{'specular_abgamma':10.,'specular_alphabetac':2.}})
            report=search_work(data)
            self.assertIsNone(report['estimated_exhaustive_search_wall_seconds'])
            self.assertFalse(report['phases'][1]['upstream_stage_complete'])
            self.assertEqual(report['phases'][1]['possible_units_scope'],'exact_for_retained_abg_rows_only')
            self.assertIn('unknown',report['estimate_unavailable_reason'])

    def test_missing_throughput_and_zero_downstream_are_explicit(self):
        report=search_work(result())
        self.assertIsNone(report['estimated_exhaustive_search_wall_seconds'])
        self.assertIsNone(report['phases'][0]['amortized_wall_seconds_per_attempt'])
        data=result('specular',seeds=3,counts={'specular_stage1_systems':25*7**6},search_complete=True,
                    timing={'stage_seconds':{'specular_abgamma':10.,'specular_alphabetac':.1}})
        report=search_work(data)
        self.assertEqual(report['phases'][1]['possible_units'],0)
        self.assertEqual(report['estimated_exhaustive_search_wall_seconds'],10.1)
        self.assertEqual(report['estimated_remaining_search_wall_seconds'],0.)
        self.assertTrue(report['exhaustion_observed'])
        self.assertFalse(report['guarantees_correct_indexing'])
        self.assertIsNone(search_work(result(seeds=0))['estimated_exhaustive_search_wall_seconds'])

    def test_ineligible_qz_rows_have_no_stage2_systems(self):
        data=result('specular',seeds=3,counts={'specular_stage1_systems':25*7**6,
                    'specular_abg_candidates':2,'insufficient_distinct_qz':2},
                    timing={'stage_seconds':{'specular_abgamma':10.,'specular_alphabetac':.1}})
        self.assertEqual(search_work(data)['phases'][1]['possible_units'],0)

    def test_csv_exports_even_when_no_solutions_found(self):
        data=result(counts={'without_specular_hkl_quadruples':1000},
                    timing={'stage_seconds':{'without_specular_triclinic_seeds':2.}})
        data.update(solutions=[],search_work=search_work(data))
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);write_outputs(folder,data)
            with (folder/'search_work.csv').open() as f:
                rows=list(csv.DictReader(f))
            self.assertEqual(int(rows[0]['possible_units']),4100625)
            self.assertEqual(json.loads((folder/'results.json').read_text())['search_work'],data['search_work'])


if __name__=='__main__':
    unittest.main()
