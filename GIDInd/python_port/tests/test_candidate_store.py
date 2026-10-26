"""Lossless spill storage, stable numerical ordering and crash checkpoints."""
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from gidind_reference.candidate_store import CandidateStore
from gidind_reference.config import Settings
from gidind_reference.state import Candidate, State
from gidind_reference.runner import index_file
from gidind_reference.finalize import finalize
from gidind_reference.inputs import Peaks
from test_reference import synthetic, write_csv


def candidate(i, score=1.):
    return Candidate(np.eye(3)*(i+1),np.array([[i,0,1]]),
                     projection=np.array([0.,0.,1.]),native_scores={'seed_linear_sumerr':score})


class CandidateStorageTests(unittest.TestCase):
    def test_no_default_cap_and_collection_beyond_previous_100000_limit(self):
        self.assertIsNone(Settings().max_intermediate_candidates)
        for value in (0,-1,True,1.5,'100'):
            with self.assertRaises(ValueError):Settings(max_intermediate_candidates=value).validate()
        with tempfile.TemporaryDirectory() as tmp:
            state=State(Settings(),candidate_path=Path(tmp)/'native.sqlite3')
            try:
                for i in range(100005):state.add_pending(candidate(i))
                state.pending.flush()
                self.assertEqual(len(state.pending),100005)
                self.assertEqual(state.partial_reasons,[])
                # No growing Python dictionary retains assignments or timings.
                self.assertEqual(state.pending_keys,{})
                self.assertEqual(state.candidate_timings,{})
                self.assertEqual(state.pending[-1].hkl[0,0],100004)
            finally:state.pending.close()

    def test_replacement_ties_planes_and_timings_match_memory_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            disk=State(Settings(),candidate_path=Path(tmp)/'native.sqlite3')
            memory=State(Settings())
            try:
                for state in (memory,disk):
                    clock=iter([10.,11.,12.,13.,14.,15.])
                    with patch('gidind_reference.state.time.perf_counter',side_effect=lambda:next(clock)):
                        for i,s in [(0,2.),(1,1.),(0,2.),(0,1.),(2,1.)]:
                            state.add_pending(candidate(i,s))
                        c=candidate(0,1.);c.plane=np.array([0,0,1]);state.add_pending(c)
                self.assertEqual(disk.counts,memory.counts)
                self.assertEqual(len(disk.pending),len(memory.pending))
                expected=sorted(memory.pending,key=lambda c:c.native_scores['seed_linear_sumerr'])
                for a,b in zip(disk.pending.view(),expected):
                    np.testing.assert_array_equal(a.metric,b.metric)
                    np.testing.assert_array_equal(a.hkl,b.hkl)
                    np.testing.assert_array_equal(a.projection,b.projection)
                    np.testing.assert_array_equal(a.plane,b.plane)
                    self.assertEqual(a.native_scores,b.native_scores)
                    # started is different per state; compare timestamp deltas.
                    self.assertEqual(a.timing['candidate_id'],b.timing['candidate_id'])
                    self.assertAlmostEqual(a.timing['discovered_wall_seconds']-
                        a.timing['assignment_first_seen_wall_seconds'],
                        b.timing['discovered_wall_seconds']-b.timing['assignment_first_seen_wall_seconds'])
            finally:disk.pending.close()

    def test_abg_disk_order_matches_python_sort_including_ties(self):
        with tempfile.TemporaryDirectory() as tmp:
            store=CandidateStore(Path(tmp)/'native.sqlite3')
            try:
                rows=[]
                for i in range(40):
                    row=(np.array([5.+i%3,6.,1.2]),(i%2,0),np.full((4,2,2),i),float(i%4))
                    rows.append(row);store.abgs.append(row)
                store.flush()
                for order in ('area','qxy_error'):
                    expected=sorted(rows,key=lambda r:(r[0][0]*r[0][1]*np.sin(r[0][2])
                        if order=='area' else r[3],r[3],tuple(r[0]),r[1]))
                    for a,b in zip(store.abgs.ordered(order),expected):
                        for aa,bb in zip(a,b):np.testing.assert_array_equal(aa,bb)
            finally:store.close()

    def test_large_checkpoint_uses_database_not_whole_pool_npz(self):
        _,_,_,_,coords=synthetic()
        def search(peaks,seed,settings,state):
            for i in range(25):state.add_pending(candidate(i))
            state.checkpoint(state)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p=root/'input.csv';write_csv(p,coords)
            with patch('gidind_reference.without_specular.search',side_effect=search),\
                 patch('gidind_reference.runner.finalize'):
                result=index_file(p,'without-specular',root/'run',Settings(candidate_snapshot_max_rows=2))
            self.assertEqual(result['pending_candidates'],25)
            self.assertEqual(result['partial_reasons'],[])
            self.assertEqual(result['status'],'no_solution')
            self.assertFalse((root/'run/intermediate_candidates.npz').exists())
            saved=json.loads((root/'run/checkpoint.json').read_text())
            self.assertEqual(saved['candidate_storage']['native_candidates'],25)
            from gidind_reference.batch import reused_result, copy_run_artifacts
            alias_dir=root/'alias';alias_dir.mkdir()
            peaks=Peaks(coords,np.arange(2,len(coords)+2),{'selected_specular_q':None})
            alias=reused_result(result,root/'run',peaks,'without-specular',Settings())
            copy_run_artifacts(root/'run',alias_dir)
            self.assertTrue((alias_dir/alias['candidate_storage']['path']).is_file())
            self.assertFalse((alias_dir/'native_candidates.sqlite3').exists())
            store=CandidateStore(root/'run/native_candidates.sqlite3',create=False)
            try:self.assertEqual(len(list(store)),25)
            finally:store.close()

    def test_unclean_exit_recovers_committed_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'native.sqlite3'
            code=('import os,numpy as np;from gidind_reference.state import State,Candidate;'
                  'from gidind_reference.config import Settings;'
                  f's=State(Settings(),candidate_path={str(path)!r});'
                  '\nfor i in range(2053):s.add_pending(Candidate(np.eye(3),np.array([[i,0,1]]),'
                  'native_scores={"seed_linear_sumerr":1.}))\nos._exit(19)')
            process=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True)
            self.assertEqual(process.returncode,19,process.stderr)
            store=CandidateStore(path,create=False)
            try:
                self.assertEqual(len(store),2048)
                self.assertEqual(store.db.execute('PRAGMA integrity_check').fetchone()[0],'ok')
            finally:store.close()

    def test_corrected_and_upstream_finalization_equal_memory_path(self):
        _,metric,plane,h,coords=synthetic()
        peaks=Peaks(coords,np.arange(2,len(coords)+2),{'selected_specular_q':None})
        with tempfile.TemporaryDirectory() as tmp:
            for algorithm in ('corrected','upstream'):
                settings=Settings(algorithm=algorithm,planes=[[0,0,1]])
                memory=State(settings)
                disk=State(settings,candidate_path=Path(tmp)/f'{algorithm}.sqlite3')
                try:
                    for state in (memory,disk):
                        for i,s in [(0,.1),(1,.05),(2,.06),(3,.2)]:
                            hh=h.copy();hh[0,0]+=i
                            state.add_pending(Candidate(metric,hh,projection=metric@plane/np.sqrt(plane@metric@plane),
                                                        native_scores={'seed_linear_sumerr':s}))
                        finalize(peaks,'without-specular',settings,state)
                    self.assertEqual(disk.solutions,memory.solutions)
                    self.assertEqual(disk.counts,memory.counts)
                finally:disk.pending.close()


if __name__=='__main__':unittest.main()
