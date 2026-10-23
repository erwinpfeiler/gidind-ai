"""Exact same-work parity and bounded-worker lifecycle tests."""
from dataclasses import replace
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch, Mock
import numpy as np

from gidind_reference.config import Settings
from gidind_reference.geometry import hkl_grid, predict, reciprocal_metric
from gidind_reference.parallel import OrderedExecutor
from gidind_reference.runner import index_file
from gidind_reference.state import Candidate, State, SearchLimit
from gidind_reference.without_specular import commit_seed_batch
from test_reference import synthetic, write_csv


def scheduling_kernel(kind, data, payload):
    delay, value = payload
    time.sleep(delay)
    if kind == 'error':
        raise ValueError('deliberate worker failure')
    if kind == 'death':
        os._exit(17)
    return (value, time.monotonic(), os.getpid(), os.environ.get('OPENBLAS_NUM_THREADS'),
            data.get('test_array').flags.writeable if 'test_array' in data else None)


def small_cases(root):
    cell, metric, plane, h, coords = synthetic()
    specpath = root/'specular.csv'
    write_csv(specpath, np.vstack([[0, np.sqrt(plane@metric@plane)], coords]))
    spec = Settings(algorithm='upstream',specular_seed_peaks=4, specular_seed_hk=1, specular_index_hk=2,
                    specular_l=3, specular_uv=1, specular_w=1, redundancy_hk=2,
                    planes=[[0,0,1]], batch_size=729, time_limit_seconds=300,
                    rmse_cutoffs=[1e-6]*3, max_intermediate_candidates=2000)
    cell[3:] = np.deg2rad([93,98,104])
    pp = predict(reciprocal_metric(cell), hkl_grid((1,1,1)), plane=plane)
    coords = pp[(pp[:,1]>.05)&(pp[:,0]>.05), :2]
    nopath = root/'without-specular.csv'
    write_csv(nopath, coords)
    nospec = Settings(algorithm='upstream',without_specular_seed_hkl=[1,1,1], planes=[[0,0,1]],
                      time_limit_seconds=300, rmse_cutoffs=[1e-6]*3)
    return [(specpath, 'specular', spec), (nopath, 'without-specular', nospec)]


def signature(result, folder):
    result = {key: result[key] for key in ('status','counts','solutions','pending_candidates','partial_reasons')}
    result['intermediate_sha256'] = {}
    if (folder/'intermediate_candidates.npz').exists():
        with np.load(folder/'intermediate_candidates.npz') as arrays:
            result['intermediate_sha256'] = {k: hashlib.sha256(arrays[k].tobytes()).hexdigest() for k in arrays.files}
    return result


class ParallelTests(unittest.TestCase):
    def assert_no_children(self):
        self.assertEqual(mp.active_children(), [])

    def test_ordered_bounded_pool_and_single_blas_thread(self):
        executor = OrderedExecutor({'test_array':np.arange(3)}, 2, kernel=scheduling_kernel)
        try:
            results = list(executor.ordered('echo', [(0.3,0),(0.,1),(0.,2),(0.,3),(0.,4)], lambda: None))
        finally:
            executor.close()
        self.assertEqual([r[0] for r in results], list(range(5)))
        self.assertLess(results[1][1], results[0][1])  # Actually completed out of order.
        self.assertEqual({r[3] for r in results}, {'1'})
        self.assertEqual({r[4] for r in results}, {False})
        self.assertEqual(executor.info['started_workers'], 2)
        self.assertLessEqual(executor.info['max_outstanding_tasks'], 4)
        self.assertEqual(executor.info['committed_tasks'], 5)
        self.assertEqual(executor.info['discarded_tasks'], 0)
        self.assert_no_children()

    def test_timeout_discards_uncommitted_prefix_and_pool_can_serve_next_stage(self):
        executor = OrderedExecutor({}, 2, kernel=scheduling_kernel)
        executor.start(lambda: None)
        deadline = time.monotonic()+.12
        def check():
            if time.monotonic() >= deadline:
                raise SearchLimit('time_limit')
        committed = []
        try:
            with self.assertRaises(SearchLimit):
                for result in executor.ordered('echo', [(0.,0),(0.5,1),(0.,2),(0.,3)], check):
                    committed.append(result[0])
            self.assertEqual(committed, [0])
            results = list(executor.ordered('echo', [(0.,9)], lambda: None))
            self.assertEqual(results[0][0], 9)
        finally:
            executor.close(abort=True)
        self.assertEqual(executor.info['submitted_tasks'], executor.info['committed_tasks']+executor.info['discarded_tasks'])
        self.assert_no_children()

    def test_worker_error_and_unexpected_death_are_failures_without_orphans(self):
        for kind, message in [('error','deliberate worker failure'), ('death','exited unexpectedly')]:
            executor = OrderedExecutor({}, 2, kernel=scheduling_kernel)
            try:
                with self.assertRaisesRegex((ValueError, RuntimeError), message):
                    list(executor.ordered(kind, [(0.,0)], lambda: None))
            finally:
                executor.close(abort=True)
            self.assert_no_children()

    def test_interruption_during_pool_shutdown_reaps_before_reraising(self):
        executor=OrderedExecutor({},2)
        executor.pool=Mock()
        executor.pool.join.side_effect=[KeyboardInterrupt(),None]
        executor.ready=Mock()
        with self.assertRaises(KeyboardInterrupt): executor.close()
        executor.pool.terminate.assert_called_once()
        self.assertEqual(executor.pool.join.call_count,2)
        executor.ready.close.assert_called_once()
        self.assertIn('child_process_cpu_seconds',executor.info)

    def test_worker_failure_writes_readable_index_checkpoint(self):
        def fail(peaks, seed, settings, state):
            list(state.executor.ordered('invalid-kind', [None], state.check))
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); path, mode, settings=small_cases(root)[1]
            with patch('gidind_reference.without_specular.search', side_effect=fail):
                result=index_file(path,mode,root/'failed',replace(settings,workers=2))
            saved=json.loads((root/'failed/checkpoint.json').read_text())
            self.assertEqual(result['status'],'failure')
            self.assertEqual(saved['status'],'failure')
            self.assertIn('Unknown numerical batch',saved['error'])
            self.assert_no_children()

    def test_global_cache_clear_duplicate_tie_and_better_replacement(self):
        class SmallCache(set):
            def __len__(self):
                return 200001 if super().__len__() > 2 else super().__len__()
        state=State(Settings())
        h=np.array([[1,0,0]])
        def candidate(score):
            return Candidate(np.eye(3)*score,h,native_scores={'seed_linear_sumerr':score})
        cache=SmallCache()
        commit_seed_batch({}, [('a',candidate(2.),None),('a',None,'must_be_ignored'),
                               ('b',candidate(2.),None),('c',candidate(1.),None),
                               ('a',candidate(1.),None)], cache, state)
        self.assertEqual(cache, {'a'})
        self.assertEqual(state.counts['duplicate_seed_geometry'],1)
        self.assertNotIn('must_be_ignored',state.counts)
        self.assertEqual(state.counts['duplicate_native_assignments'],3)
        self.assertEqual(state.pending[0].native_scores['seed_linear_sumerr'],1.)
        np.testing.assert_array_equal(state.pending[0].metric,np.eye(3))

    def test_flag_validation_and_cli_precedence(self):
        for value in (0,-1,True,1.5,'2'):
            with self.assertRaises(ValueError): Settings(workers=value).validate()
        self.assertEqual(Settings().workers,1)
        from gidind_reference.cli import main
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); config=root/'settings.json';config.write_text('{"workers": 4}')
            with patch('gidind_reference.supervision.index_file',return_value={'status':'completed'}) as run:
                self.assertEqual(main(['index','--input',str(root/'in.csv'),'--mode','specular',
                                      '--config',str(config),'--workers','2','--output-dir',str(root/'out')]),0)
                self.assertEqual(run.call_args.args[3].workers,2)
        self.assert_no_children()

    def test_overnight_launcher_preserves_output_argument_and_passes_workers(self):
        import shutil
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);port=root/'GIDInd/python_port';port.mkdir(parents=True)
            launcher=port/'overnight.sh'
            shutil.copy2(Path(__file__).resolve().parents[1]/'overnight.sh',launcher)
            python=port/'.venv/bin/python';python.parent.mkdir(parents=True)
            python.write_text('#!/bin/sh\nexit 99\n');python.chmod(0o755)
            run=port/'run.sh'
            run.write_text("#!/bin/sh\nprintf '%s\\n' \"$@\" > \"$GIDIND_LAUNCH_CAPTURE\"\n")
            run.chmod(0o755)
            inputs=root/'gidind_peak_lists_all';inputs.mkdir();(inputs/'manifest.json').touch()
            for sample in ('pq','mapbbr3','naib_s11','cu_bod_mof'):
                for suffix in ('','_without_specular'):
                    (inputs/f'{sample}{suffix}.csv').touch()
            capture=root/'argv.txt';output=root/'results with spaces'
            env=dict(os.environ,GIDIND_LAUNCH_CAPTURE=str(capture))
            for args in ([str(output),'--workers','4'], ['--workers','4',str(output)]):
                result=subprocess.run([str(launcher),*args],env=env,text=True,capture_output=True)
                self.assertEqual(result.returncode,0,result.stderr)
                argv=capture.read_text().splitlines()
                self.assertEqual(argv[-2:],['--workers','4'])
                self.assertEqual(argv[argv.index('--output-dir')+1],str(output))
            for args in (['--workers','0'],['--workers','00'],['--workers','-2'],['--workers']):
                result=subprocess.run([str(launcher),*args],env=env,text=True,capture_output=True)
                self.assertNotEqual(result.returncode,0)

    def test_eight_inputs_six_unique_computations_with_worker_settings(self):
        from gidind_reference.batch import benchmark
        from gidind_reference.runner import write_outputs
        from gidind_reference.inputs import read_peaks
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); inputs=root/'inputs';inputs.mkdir()
            cell,metric,plane,h,coords=synthetic()
            cases=[]
            for i,case in enumerate(('pq','mapbbr3','naib_s11','cu_bod_mof')):
                # Four distinct prepared coordinate lists; only middle two have a specular peak.
                points=coords*(1+.02*i)
                original=inputs/f'{case}.csv';filtered=inputs/f'{case}_without_specular.csv'
                write_csv(filtered,points)
                if i in (1,2):
                    points=np.vstack([[0,np.sqrt(plane@metric@plane)*(1+.02*i)],points])
                write_csv(original,np.vstack([[0,0],points]))
                digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
                cases.append({'id':case,'filename':original.name,'without_specular':{
                    'filename':filtered.name,'sha256':digest(filtered),'source_sha256':digest(original)}})
            (inputs/'manifest.json').write_text(json.dumps({'cases':cases}))
            def fake_index(path,mode,folder,settings):
                self.assertEqual(settings.workers,4)
                folder.mkdir()
                result={'status':'completed','solutions':[],
                        'input':read_peaks(path,mode,settings).provenance}
                write_outputs(folder,result)
                return result
            with patch('gidind_reference.batch.index_file',side_effect=fake_index) as index:
                result=benchmark(inputs,root/'out',Settings(workers=4))
            self.assertEqual(index.call_count,6)
            self.assertEqual(result['unique_computations'],6)
            self.assertEqual(len(result['jobs']),8)
            self.assertEqual(sum(j['reused'] for j in result['jobs']),2)
            self.assertEqual(sum(bool(j['specular_comparison_unavailable']) for j in result['jobs']),2)

    def test_exact_completed_search_parity_against_serial_and_pre_refactor(self):
        baseline=json.loads((Path(__file__).parent/'fixtures/pre_multicore.json').read_text())['cases']
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for path, mode, settings in small_cases(root):
                serial=None
                for workers in (1,2,4):
                    folder=root/f'{mode}_{workers}'
                    result=index_file(path,mode,folder,replace(settings,workers=workers))
                    actual=signature(result,folder)
                    self.assertEqual(result['status'],'completed')
                    self.assertTrue(result['search_work']['exhaustion_observed'])
                    self.assertEqual(result['search_work']['estimated_remaining_search_wall_seconds'],0.)
                    self.assertTrue(all(p['enumeration_fraction']==1. for p in result['search_work']['phases']))
                    self.assertIsNotNone(result['timing']['first_validated_solution_wall_seconds'])
                    self.assertLess(result['timing']['first_validated_solution_wall_seconds'],result['timing']['total_wall_seconds'])
                    if workers==1:
                        serial=actual
                        # Frozen bitwise numerical baseline is specific to its pinned environment.
                        if sys.version_info[:2]==(3,12) and np.__version__=='2.5.4' and sys.platform=='linux' and os.uname().machine=='x86_64':
                            self.assertEqual(actual['counts'],baseline[mode]['counts'])
                            self.assertEqual(actual['intermediate_sha256'],baseline[mode]['intermediate_sha256'])
                            digest=hashlib.sha256(json.dumps(actual['solutions'],sort_keys=True).encode()).hexdigest()
                            self.assertEqual(digest,baseline[mode]['solutions_sha256'])
                    self.assertEqual(actual,serial)
                    self.assertLessEqual(result['execution']['max_outstanding_tasks'],max(1,2*workers))
                    self.assert_no_children()

    def test_candidate_and_solution_limits_are_worker_invariant(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path,mode,settings=small_cases(root)[0]
            for label, changes in [('candidate',{'max_intermediate_candidates':2}), ('solution',{'max_solutions':1})]:
                signatures=[]
                for workers in (1,2):
                    folder=root/f'{label}_{workers}'
                    result=index_file(path,mode,folder,replace(settings,workers=workers,**changes))
                    signatures.append(signature(result,folder))
                    self.assertEqual(result['status'],'search_limited')
                self.assertEqual(*signatures)
                self.assert_no_children()

    def test_cli_sigint_and_deadline_save_results_and_reap_workers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path,mode,settings=small_cases(root)[1]
            for label in ('deadline','sigint'):
                config=root/f'{label}.json'
                config.write_text(json.dumps(replace(settings,workers=2,time_limit_seconds=2 if label=='deadline' else 300,checkpoint_seconds=.1,without_specular_seed_hkl=[1,1,2]).to_dict()))
                out=root/label
                proc=subprocess.Popen([sys.executable,'-m','gidind_reference','index','--input',str(path),
                                       '--mode',mode,'--config',str(config),'--output-dir',str(out)],
                                      stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
                try:
                    if label=='sigint':
                        until=time.monotonic()+15
                        while time.monotonic()<until:
                            checkpoint=out/'checkpoint.json'
                            if checkpoint.exists():
                                saved=json.loads(checkpoint.read_text())
                                if saved.get('execution',{}).get('started_workers')==2 and saved.get('counts'):
                                    break
                            time.sleep(.05)
                        else:
                            self.fail('Workers did not start in time')
                        proc.send_signal(signal.SIGINT)
                    stdout,stderr=proc.communicate(timeout=15)
                finally:
                    if proc.poll() is None:
                        proc.kill();proc.communicate()
                self.assertEqual(proc.returncode,130 if label=='sigint' else 0,(stdout,stderr))
                saved=json.loads((out/'results.json').read_text())
                self.assertEqual(saved['status'],'interrupted' if label=='sigint' else 'time_limited')
                self.assertEqual(json.loads((out/'checkpoint.json').read_text())['status'],saved['status'])
                for pid in saved['execution']['worker_pids']:
                    with self.assertRaises(ProcessLookupError): os.kill(pid,0)


if __name__=='__main__':
    unittest.main()
