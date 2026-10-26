"""Corrected tail-search counterexamples, mode isolation and process parity."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from gidind_reference.config import Settings
from gidind_reference.geometry import hkl_grid, metric_design, reciprocal_metric
from gidind_reference.inputs import read_peaks, seed_coordinates, Peaks
from gidind_reference.without_specular import seed_batch, seed_systems
from gidind_reference.finalize import finalize
from gidind_reference.parallel import OrderedExecutor
from gidind_reference.state import State, Candidate

ROOT=Path(__file__).resolve().parent/'fixtures'


def counterexample():
    settings=Settings(without_specular_seed_hkl=[2,2,2],without_specular_index_hkl=[4,4,4])
    peaks=read_peaks(ROOT/'mapbbr3_without_specular.csv','without-specular',settings)
    seed,_=seed_coordinates(peaks,'without-specular',settings)
    grid=hkl_grid((2,2,2),sorted_norm=True)
    lookup={tuple(h):i for i,h in enumerate(grid)}
    traces=json.loads((ROOT/'audit.json').read_text())['traces']
    digits=np.array([[lookup[tuple(h)] for h in t['enumerated_h4']] for t in traces])
    data=dict(settings=settings,seed=seed,seed_hkls=grid,coordinates=peaks.coordinates,
              indexing=hkl_grid((4,4,4)))
    return settings,peaks,data,digits


def signature(results):
    records=[]
    for counts,outcomes in results:
        entries=[]
        for key,candidate,rejection in outcomes:
            entries.append([key,None if candidate is None else dict(
                metric=candidate.metric.tolist(),hkl=candidate.hkl.tolist(),
                projection=candidate.projection.tolist(),scores=candidate.native_scores),rejection])
        records.append([dict(counts),entries])
    return json.dumps(records,sort_keys=True)


class CorrectedTests(unittest.TestCase):
    def test_discrete_window_anchors_after_first_accepted_orientation(self):
        from test_reference import synthetic
        cell,metric,_,h,coords=synthetic()
        bad_cell=cell.copy();bad_cell[0]=6.
        peaks=Peaks(coords,np.arange(1,len(coords)+1),{})
        def finish(candidate,*args):
            return candidate.metric,h,np.eye(3),{}
        for algorithm in ('upstream','corrected'):
            settings=Settings(algorithm=algorithm,planes=[[0,0,1]],rmse_cutoffs=[1e-8]*3)
            state=State(settings)
            state.pending=[Candidate(reciprocal_metric(bad_cell),h,
                native_scores={'seed_linear_sumerr':.001}),
                Candidate(metric,h,native_scores={'seed_linear_sumerr':.1}),
                Candidate(metric,h,native_scores={'seed_linear_sumerr':.2})]
            with patch('gidind_reference.finalize._native_finish',side_effect=finish):
                finalize(peaks,'without-specular',settings,state)
            if algorithm=='upstream':self.assertEqual(state.solutions,[])
            else:
                self.assertTrue(state.solutions)
                self.assertEqual(state.counts['corrected_discrete_window_anchored'],1)
                self.assertEqual(state.counts['corrected_discrete_window_skipped'],1)

    def test_default_validation_and_cli_json_override(self):
        from gidind_reference.cli import main
        self.assertEqual(Settings().algorithm,'corrected')
        for kw in ({'algorithm':'matlab-exact'},{'tail_hkl_candidates':0},
                   {'tail_hkl_candidates':True},{'tail_hkl_candidates':1.5}):
            with self.assertRaises(ValueError):Settings(**kw).validate()
        with tempfile.TemporaryDirectory() as tmp:
            config=Path(tmp)/'settings.json'
            config.write_text('{"algorithm":"corrected"}')
            for command in ('index','benchmark'):
                argv=[command,'--input' if command=='index' else '--input-dir',str(Path(tmp)/'in'),
                      '--output-dir',str(Path(tmp)/'out'),'--config',str(config),'--algorithm','upstream']
                if command=='index':argv+=['--mode','without-specular']
                target='gidind_reference.supervision.index_file' if command=='index' else 'gidind_reference.batch.benchmark'
                with patch(target,return_value={'status':'completed'}) as call:
                    self.assertEqual(main(argv),0)
                    self.assertEqual(call.call_args.args[-1].algorithm,'upstream')
            self.assertEqual(Settings.load(config).algorithm,'corrected')

    def test_zero_hkl_counterexample_repaired_without_changing_cutoffs(self):
        settings,peaks,data,digits=counterexample()
        old_counts,old=seed_batch(dict(data,settings=replace(settings,algorithm='upstream'),
                                      check_cancel=lambda:None),digits)
        self.assertEqual(old_counts['singular_six_hkl'],15)
        self.assertEqual(old,[])
        counts,outcomes=seed_batch(dict(data,check_cancel=lambda:None),digits)
        self.assertEqual(counts['without_specular_hkl_quadruples'],15)
        self.assertEqual(counts['corrected_tail_systems'],15*8**2)
        state=State(settings)
        state.pending=[c for _,c,_ in outcomes if c is not None]
        self.assertTrue(state.pending)
        finalize(peaks,'without-specular',settings,state)
        self.assertTrue(state.solutions)
        best=state.solutions[0]
        self.assertLess(best['rmse']['q_xyz'],.003)
        self.assertLess(max(best['rmse'].values()),.004)
        self.assertLess(max(abs(best['cell'][k]-5.94) for k in ('a','b','c')),.04)

    def test_no_zero_tail_indices_and_no_uninformative_qz_quartet(self):
        settings,_,data,digits=counterexample()
        h4=data['seed_hkls'][digits]
        _,h6,selected,_=seed_systems(h4,data['seed'],data['seed_hkls'],corrected=True)
        self.assertTrue(len(h6))
        self.assertFalse(np.any(np.all(h6==0,axis=2)))
        self.assertTrue(np.all(np.any(data['seed'][selected[:,:4],1]>0,axis=1)))
        self.assertTrue(all(np.linalg.matrix_rank(metric_design(h[:4]))==4 for h in h6))
        zero_seed=data['seed'].copy();zero_seed[:,1]=0
        _,none,_,_=seed_systems(h4,zero_seed,data['seed_hkls'],corrected=True)
        self.assertEqual(len(none),0)

    def test_corrected_kernel_exact_serial_spawn_parity(self):
        settings,_,data,digits=counterexample()
        tasks=[digits[:7],digits[7:]]
        expected=None
        for workers in (1,2,4):
            executor=OrderedExecutor(dict(data,settings=replace(settings,workers=workers)),workers)
            try:actual=signature(list(executor.ordered('without_specular',tasks,lambda:None)))
            finally:executor.close(abort=True)
            if expected is None:expected=actual
            self.assertEqual(actual,expected)


if __name__=='__main__':unittest.main()
