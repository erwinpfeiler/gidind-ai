"""Regression coverage for slow checkpoints, partial journals and hard watchdogs."""
import copy
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
import numpy as np

from gidind_reference.config import Settings
from gidind_reference.state import Candidate,SearchLimit,State
from gidind_reference.records import encode_record,read_records
from gidind_reference.runner import index_file
from gidind_reference.supervision import index_file as supervised_index
from test_parallel import small_cases


class RuntimeTests(unittest.TestCase):
    def test_child_diagnostics_never_start_async_stack_walks(self):
        from gidind_reference import supervision
        with tempfile.TemporaryDirectory() as tmp:
            with patch('gidind_reference.runner.index_file',return_value={'status':'completed'}),\
                 patch('gidind_reference.config.Settings.load',return_value=Settings()),\
                 patch.object(supervision.faulthandler,'enable') as enable,\
                 patch.object(supervision.faulthandler,'register') as register,\
                 patch.object(supervision.faulthandler,'dump_traceback_later') as timer:
                try:
                    self.assertEqual(supervision._child(['input.csv','specular',tmp,'settings.json']),0)
                    enable.assert_called_once_with(file=supervision._DIAGNOSTIC_STREAM,all_threads=False)
                    register.assert_not_called();timer.assert_not_called()
                finally:
                    supervision._DIAGNOSTIC_STREAM.close()

    @unittest.skipUnless(os.name=='posix','Process-group crash regression is POSIX specific')
    def test_crashed_coordinator_stops_surviving_descendants(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path,mode,settings=small_cases(root)[1]
            descendant=root/'descendant.pid'
            # SIGSEGV reproduces the exit status, not the unconfirmed C bug.
            # Disable core files so this regression cannot fill the disk.
            code=('import os,resource,signal,subprocess,sys,time;'
                  'resource.setrlimit(resource.RLIMIT_CORE,(0,0));'
                  'p=subprocess.Popen([sys.executable,"-c",'
                  '"import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(60)"]);'
                  f'open({str(descendant)!r},"w").write(str(p.pid));'
                  'time.sleep(.1);os.kill(os.getpid(),signal.SIGSEGV)')
            result=supervised_index(path,mode,root/'run',settings,_command=[sys.executable,'-c',code])
            self.assertEqual(result['status'],'failure')
            self.assertEqual(result['supervision']['child_exit_code'],-signal.SIGSEGV)
            self.assertEqual(result['supervision']['child_signal'],'SIGSEGV')
            self.assertFalse(result['supervision']['timed_out'])
            self.assertIn('child_process_failure',result['partial_reasons'])
            pid=int(descendant.read_text())
            # A killed orphan may remain a zombie until the host init reaps it.
            for _ in range(50):
                stat=Path(f'/proc/{pid}/stat')
                if not stat.exists() or stat.read_text().split()[2]=='Z':break
                time.sleep(.02)
            if stat.exists():self.assertEqual(stat.read_text().split()[2],'Z')
            else:
                with self.assertRaises(ProcessLookupError):os.kill(pid,0)

    def test_slow_checkpoint_waits_a_full_interval_and_rechecks_deadline(self):
        now=[20.];calls=[]
        def checkpoint(state):calls.append(now[0]);now[0]+=20.
        st=State(Settings(time_limit_seconds=100,checkpoint_seconds=15),checkpoint,started=0.)
        with patch('gidind_reference.state.time.perf_counter',side_effect=lambda:now[0]):
            st.check();self.assertEqual(st.last_checkpoint,40.)
            now[0]=41.;st.check();self.assertEqual(calls,[20.])
            now[0]=90.
            with self.assertRaises(SearchLimit):st.check()
            self.assertEqual(st.last_checkpoint,110.)

    def test_journal_replacement_checksum_and_partial_tail(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'records'
            a={'cell':[1,2,3],'assignments':[{'h':1,'k':2,'l':3}]*1000}
            b=dict(a,cell=[4,5,6])
            raw=encode_record({'sequence':0,'solution_slot':0},a)
            raw+=encode_record({'sequence':1,'solution_slot':0},b)
            self.assertLess(len(raw),len(json.dumps(a))+len(json.dumps(b)))
            path.write_text(raw+'{"unfinished":')
            solutions,stats=read_records(path)
            self.assertEqual(solutions,{0:b});self.assertTrue(stats['truncated_final_record'])
            records=raw.splitlines();r=json.loads(records[1]);r['sha256']='bad'
            path.write_text(records[0]+'\n'+json.dumps(r)+'\n')
            with self.assertRaisesRegex(ValueError,'checksum'):read_records(path)

    def test_finalization_checkpoints_do_not_rewrite_raw_arrays_or_full_fits(self):
        fixture=Path(__file__).parent/'fixtures/finalization_solution.json'
        solution=json.loads(fixture.read_text())['solutions'][0]
        def search(peaks,seed,settings,state):
            state.set_stage('without_specular_triclinic_seeds')
            candidate=Candidate(np.eye(3),np.zeros((len(peaks.coordinates),3),dtype=int),
                                native_scores={'seed_linear_sumerr':1.})
            state.add_pending(candidate);state.checkpoint(state)
        def finish(peaks,mode,settings,state):
            state.set_stage('correction_reduction_discrete_projection')
            for i in range(20):
                state.solutions.append(copy.deepcopy(solution))
                state.record_solution(state.pending[0],i,state.solutions[-1],'accepted')
            state.checkpoint(state);state.checkpoint(state)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path,mode,settings=small_cases(root)[1];out=root/'run'
            with patch('gidind_reference.without_specular.search',side_effect=search),\
                 patch('gidind_reference.runner.finalize',side_effect=finish),\
                 patch('gidind_reference.runner.np.savez_compressed',wraps=np.savez_compressed) as save:
                result=index_file(path,mode,out,settings)
            self.assertEqual(result['status'],'completed')
            self.assertEqual(save.call_count,2) # Search snapshot and final immutable snapshot only.
            checkpoint=json.loads((out/'checkpoint.json').read_text())
            self.assertEqual(checkpoint['solutions'],[])
            self.assertEqual(checkpoint['validated_solution_count'],20)
            self.assertLess((out/'checkpoint.json').stat().st_size,20000)
            records,stats=read_records(out/'solution_records.jsonl')
            self.assertEqual(len(records),20)
            self.assertEqual(records[0]['assignments'],solution['assignments'])

    @unittest.skipUnless(os.name=='posix','Process-group watchdog regression is POSIX specific')
    def test_hung_child_and_descendant_are_stopped_with_explicit_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path,mode,settings=small_cases(root)[1]
            settings.time_limit_seconds=.4;settings.workers=2
            child_pid=root/'descendant.pid'
            code=('import subprocess,sys,time; p=subprocess.Popen([sys.executable,"-c",'
                  '"import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(60)"]);'
                  f'open({str(child_pid)!r},"w").write(str(p.pid));time.sleep(60)')
            started=time.monotonic()
            result=supervised_index(path,mode,root/'run',settings,
                _command=[sys.executable,'-c',code],_grace=.2)
            self.assertLess(time.monotonic()-started,6.)
            self.assertEqual(result['status'],'failure')
            self.assertIn('supervisor_timeout',result['partial_reasons'])
            self.assertTrue(result['supervision']['timed_out'])
            saved=json.loads((root/'run/checkpoint.json').read_text())
            self.assertEqual(saved['status'],'failure')
            pid=int(child_pid.read_text())
            stat=Path(f'/proc/{pid}/stat')
            if stat.exists():self.assertEqual(stat.read_text().split()[2],'Z')
            else:
                with self.assertRaises(ProcessLookupError):os.kill(pid,0)


if __name__=='__main__':unittest.main()
