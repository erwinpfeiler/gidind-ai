"""Independent geometry criteria and retrospective timing, without oracle stopping."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from gidind_reference.evaluation import cell_geometry,match_reference,evaluate_result,evaluate_run
from gidind_reference.state import State,Candidate
from gidind_reference.config import Settings
from gidind_reference.batch import copy_run_artifacts


CRITERIA=dict(length_relative_tolerance=.02,angle_absolute_tolerance_degrees=2.,
    volume_relative_tolerance=.05,orientation_tolerance_degrees=2.,
    rmse_cutoffs=[.05,.05,.05],basis_limit=2)
REFERENCE=dict(cell=[5.,7.,9.,90.,90.,90.],contact_plane=[0,0,1],source='synthetic known cell')


def solution(cell=None,plane=None,error=.01):
    return dict(cell=dict(zip(('a','b','c','alpha','beta','gamma'),cell or REFERENCE['cell'])),
                contact_plane=plane or [0,0,1],rmse=dict(q_xy=error,q_z=error,q_xyz=error),
                volume_angstrom3=315.,max_combined_peak_error=error)


def result(folder,events,solutions=None,status='time_limited'):
    raw=''.join(json.dumps(e)+'\n' for e in events).encode()
    (folder/'solution_events.jsonl').write_bytes(raw)
    return dict(status=status,mode='without-specular',solutions=solutions or [],algorithm_variant='corrected',
        solution_event_log='solution_events.jsonl',solution_event_count=len(events),
        solution_event_log_sha256=hashlib.sha256(raw).hexdigest(),
        timing=dict(total_wall_seconds=10.,first_validated_solution_wall_seconds=1.),counts={})


class EvaluationTests(unittest.TestCase):
    def test_cu_reference_axis_and_plane_permutation_are_consistent(self):
        path=Path(__file__).resolve().parents[1]/'configs/reference_evaluation.experimental.json'
        config=json.loads(path.read_text());ref=config['cases']['cu_bod_mof']
        self.assertIn('User-confirmed',ref['orientation_status'])
        self.assertIn('Tentative',config['cases']['naib_s11']['orientation_status'])
        # Short-axis normal: (100) in the table, (010) after exchanging a/b.
        candidate=solution([10.87,9.666,10.87,90.,90.,90.],[0,1,0])
        self.assertTrue(match_reference(candidate,ref,config['criteria'])['matched'])
        candidate['contact_plane']=[0,0,1]
        self.assertFalse(match_reference(candidate,ref,config['criteria'])['matched'])

    def test_transient_success_does_not_end_the_run_or_imply_final_top_k_success(self):
        correct=solution(error=.01);wrong=solution([5.3,7.,9.,90.,90.,90.],error=.001)
        events=[dict(sequence=0,wall_seconds=1.,solution_slot=0,operation='accepted',solution=correct,candidate_timing=None),
                dict(sequence=1,wall_seconds=8.,solution_slot=1,operation='accepted',solution=wrong,candidate_timing=None)]
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);r=result(root,events,[wrong,correct])
            evaluated=evaluate_result(r,root,REFERENCE,CRITERIA,1)
            self.assertEqual(evaluated['first_correct_top_k_seconds'],1.)
            self.assertFalse(evaluated['final_top_k_success'])
            self.assertEqual(evaluated['total_wall_seconds'],10.)

    def test_equivalent_unimodular_basis_sign_and_plane(self):
        ref=dict(cell=[5.,7.,9.,84.,97.,106.],contact_plane=[1,0,2])
        _,metric=cell_geometry(ref['cell'])
        transform=np.array([[1,1,0],[0,1,0],[0,0,-1]])
        d=transform.T@metric@transform;lengths=np.sqrt(np.diag(d))
        angles=np.rad2deg([np.arccos(d[j,k]/(lengths[j]*lengths[k])) for j,k in ((1,2),(0,2),(0,1))])
        plane=(transform.T@ref['contact_plane']).tolist()
        candidate=solution(np.r_[lengths,angles].tolist(),plane)
        self.assertTrue(match_reference(candidate,ref,CRITERIA)['matched'])
        candidate['contact_plane']=[-v for v in plane]
        self.assertTrue(match_reference(candidate,ref,CRITERIA)['matched'])
        candidate['contact_plane']=[0,1,0]
        self.assertFalse(match_reference(candidate,ref,CRITERIA)['matched'])
        self.assertFalse(match_reference(solution([10.,7.,9.,90.,90.,90.]),REFERENCE,CRITERIA)['matched'])
        candidate=solution();candidate['rmse']['q_xy']=.1
        self.assertFalse(match_reference(candidate,REFERENCE,CRITERIA)['matched'])

    def test_distinct_discovery_validation_and_top_k_times(self):
        wrong=solution([5.3,7.,9.,90.,90.,90.],error=.001)
        first=solution(error=.01);better=solution(error=.0005)
        events=[dict(sequence=0,wall_seconds=1.,solution_slot=0,operation='accepted',solution=wrong,candidate_timing=None),
                dict(sequence=1,wall_seconds=4.,solution_slot=1,operation='accepted',solution=first,
                     candidate_timing=dict(discovered_wall_seconds=2.)),
                dict(sequence=2,wall_seconds=6.,solution_slot=0,operation='replaced',solution=better,
                     candidate_timing=dict(discovered_wall_seconds=3.))]
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);r=result(root,events,[better,first])
            value=evaluate_result(r,root,REFERENCE,CRITERIA,1)
            self.assertEqual(value['first_correct_validated_seconds'],4.)
            self.assertEqual(value['first_correct_top_k_seconds'],6.)
            self.assertEqual(value['retrospective_correct_candidate_discovery_seconds'],2.)
            self.assertTrue(value['final_top_k_success'])
            self.assertEqual(evaluate_result(r,root,REFERENCE,CRITERIA,5)['first_correct_top_k_seconds'],4.)
            (root/'solution_events.jsonl').write_text('corrupt')
            with self.assertRaisesRegex(ValueError,'checksum'):evaluate_result(r,root,REFERENCE,CRITERIA,5)

    def test_no_success_censoring_missing_reference_and_legacy_timeline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);r=result(root,[])
            value=evaluate_result(r,root,REFERENCE,CRITERIA,5)
            self.assertIsNone(value['first_correct_validated_seconds'])
            self.assertFalse(value['first_correct_observed'])
            self.assertEqual(value['censoring_wall_seconds'],10.)
            self.assertIsNone(evaluate_result(r,root,None,CRITERIA,5)['final_top_k_success'])
            r.pop('solution_event_log');r['solutions']=[solution()]
            value=evaluate_result(r,root,REFERENCE,CRITERIA,5)
            self.assertTrue(value['final_top_k_success'])
            self.assertEqual(value['timeline_status'],'unavailable_for_historical_result')
            self.assertIsNone(value['first_correct_validated_seconds'])

    def test_candidate_replacement_keeps_both_birth_times_and_search_continues(self):
        st=State(Settings(),started=0.)
        a=Candidate(np.eye(3),np.array([[1,0,0]]),native_scores={'score':2.})
        b=Candidate(np.eye(3),np.array([[1,0,0]]),native_scores={'score':1.})
        # Use the native score name that actually drives replacement.
        a.native_scores={'seed_linear_sumerr':2.};b.native_scores={'seed_linear_sumerr':1.}
        events=[];st.on_solution_event=events.append
        with patch('gidind_reference.state.time.perf_counter',side_effect=[1.,2.,3.]):
            st.add_pending(a);st.add_pending(b);st.record_solution(b,0,solution(),'accepted')
        timing=events[0]['candidate_timing']
        self.assertEqual(timing['assignment_first_seen_wall_seconds'],1.)
        self.assertEqual(timing['discovered_wall_seconds'],2.)
        self.assertEqual(events[0]['wall_seconds'],3.)
        self.assertEqual(st.partial_reasons,[])
        self.assertFalse(hasattr(st.settings,'reference'))

    def test_posthoc_budget_evaluation_and_alias_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);runs=[]
            for i,budget in enumerate((120.,600.)):
                folder=root/str(i);folder.mkdir()
                events=[] if i==0 else [dict(sequence=0,wall_seconds=5.,solution_slot=0,
                    operation='accepted',solution=solution(),candidate_timing=None)]
                r=result(folder,events,[] if i==0 else [solution()])
                (folder/'results.json').write_text(json.dumps(r))
                runs.append(dict(case='synthetic',variant='original',mode=r['mode'],budget_seconds=budget,
                    reused=False,results=str(folder/'results.json')))
            (root/'benchmark.json').write_text(json.dumps(dict(status='finished',jobs=runs)))
            reference=dict(REFERENCE,orientation_status='Tentative synthetic orientation')
            refs=root/'refs.json';refs.write_text(json.dumps(dict(criteria=CRITERIA,
                criteria_status='User-approved provisional',top_k=5,cases={'synthetic':reference})))
            evaluated=evaluate_run(root,refs,root/'evaluation')
            self.assertEqual(evaluated['minimum_tested_budgets'][0]['smallest_tested_successful_budget_seconds'],600.)
            self.assertFalse(evaluated['reference_used_by_search'])
            self.assertEqual(evaluated['jobs'][0]['reference_orientation_status'],'Tentative synthetic orientation')
            self.assertEqual(evaluated['jobs'][0]['reference_criteria_status'],'User-approved provisional')
            self.assertIn('Tentative synthetic orientation',(root/'evaluation/evaluation.csv').read_text())
            failed=json.loads((root/'0/results.json').read_text())
            failed.update(status='failure',solutions=[solution()])
            (root/'0/results.json').write_text(json.dumps(failed))
            again=evaluate_run(root,refs,root/'evaluation_with_failure')
            self.assertEqual(again['minimum_tested_budgets'][0]['smallest_tested_successful_budget_seconds'],600.)
            alias=root/'alias';alias.mkdir();copy_run_artifacts(root/'1',alias)
            self.assertEqual((alias/'solution_events.jsonl').read_bytes(),(root/'1/solution_events.jsonl').read_bytes())


if __name__=='__main__':unittest.main()
