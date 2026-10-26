"""SLURM array submission checks with fake sbatch/srun; never submits real jobs."""
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from gidind_reference.config import Settings
from test_array import inputs


class ArrayShellTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.port=self.root/'GIDInd/python_port';self.port.mkdir(parents=True)
        self.actual_port=Path(__file__).resolve().parents[1]
        for name in ('submit_esrf_array.sh','slurm_esrf_array.sh','slurm_esrf_collect.sh','submit_revision_2h.sh'):
            shutil.copy2(self.actual_port/name,self.port/name)
        preflight=self.port/'slurm_esrf.sh';preflight.write_text('#!/bin/sh\nexit 0\n')
        python=self.port/'.venv/bin/python';python.parent.mkdir(parents=True)
        python.write_text(f'#!/bin/sh\nexec {shlex.quote(sys.executable)} "$@"\n');python.chmod(0o755)
        (self.port/'configs').mkdir();(self.port/'configs/revision_2h_32cpu.json').write_text(json.dumps(Settings().to_dict()))
        inputs(self.root/'gidind_peak_lists_all')
        binpath=self.root/'bin';binpath.mkdir()
        self.log=self.root/'submissions.jsonl'
        fake=self.root/'fake_sbatch.py'
        fake.write_text('''import json,os,sys
with open(os.environ['ARRAY_TEST_LOG'],'a') as f: f.write(json.dumps(sys.argv[1:])+'\\n')
collector=any(a.startswith('--dependency=') for a in sys.argv)
if collector and os.environ.get('FAIL_COLLECTOR'): raise SystemExit(1)
print('502' if collector else '501')
''')
        sbatch=binpath/'sbatch';sbatch.write_text(f'#!/bin/sh\nexec {shlex.quote(sys.executable)} {shlex.quote(str(fake))} "$@"\n');sbatch.chmod(0o755)
        srun=binpath/'srun';srun.write_text("#!/bin/sh\nprintf '%s\\n' \"$@\" > \"$ARRAY_TEST_SRUN\"\n");srun.chmod(0o755)
        self.capture=self.root/'srun.txt'
        self.env={k:v for k,v in os.environ.items() if not k.startswith('SLURM_')}
        self.env.update(GIDIND_PROJECT_ROOT=str(self.root),ARRAY_TEST_LOG=str(self.log),
                        ARRAY_TEST_SRUN=str(self.capture),PATH=str(binpath)+os.pathsep+os.environ['PATH'],
                        PYTHONPATH=str(self.actual_port)+os.pathsep+os.environ.get('PYTHONPATH',''))

    def call(self,script,*args,**env):
        return subprocess.run(['bash',str(self.port/script),*map(str,args)],env=dict(self.env,**env),
                              text=True,capture_output=True)

    def test_all_six_tasks_eligible_and_thirty_two_cpus_per_case(self):
        script=(self.port/'slurm_esrf_array.sh').read_text()
        self.assertIn('#SBATCH --array=0-5\n',script)
        self.assertNotIn('%4',script)
        self.assertIn('#SBATCH --cpus-per-task=32',script)
        result=self.call('slurm_esrf_array.sh',self.root/'run',SLURM_ARRAY_TASK_ID='3',
                         SLURM_ARRAY_JOB_ID='501',SLURM_CPUS_PER_TASK='32')
        self.assertEqual(result.returncode,0,result.stderr)
        args=self.capture.read_text().splitlines()
        self.assertIn('--cpus-per-task=32',args)
        self.assertEqual(args[args.index('--task-id')+1],'3')
        self.assertIn('gidind_reference.slurm_array',args)

    def test_submission_freezes_budget_and_collects_after_any_outcome(self):
        for mode,budget,wall in [('smoke',120.,'00:10:00'),('full',7200.,'02:30:00')]:
            run=self.root/f'run_{mode}'
            result=self.call('submit_esrf_array.sh',mode,run)
            self.assertEqual(result.returncode,0,result.stderr)
            plan=json.loads((run/'array_plan.json').read_text())
            self.assertEqual(len(plan['tasks']),6)
            self.assertEqual(plan['effective_settings']['workers'],32)
            self.assertEqual(plan['effective_settings']['time_limit_seconds'],budget)
            commands=[json.loads(line) for line in self.log.read_text().splitlines()][-2:]
            self.assertIn('--time='+wall,commands[0])
            self.assertIn('--dependency=afterany:501',commands[1])
            self.assertEqual((run/'array_job_id.txt').read_text().strip(),'501')
            self.assertEqual((run/'collector_job_id.txt').read_text().strip(),'502')
            self.assertFalse((run/'benchmark.json').exists())

    def test_collector_failure_preserves_submitted_array_receipt(self):
        run=self.root/'run'
        result=self.call('submit_esrf_array.sh','full',run,FAIL_COLLECTOR='1')
        self.assertNotEqual(result.returncode,0)
        self.assertIn('array IS submitted',result.stderr)
        self.assertTrue((run/'array_job_id.txt').is_file())
        self.assertFalse((run/'collector_job_id.txt').exists())

    def test_revision_full_waits_for_successful_smoke_collection(self):
        run=self.root/'revision'
        result=self.call('submit_revision_2h.sh',run)
        self.assertEqual(result.returncode,0,result.stderr)
        commands=[json.loads(line) for line in self.log.read_text().splitlines()]
        self.assertEqual(len(commands),4)
        self.assertIn('--dependency=afterok:502',commands[2])
        self.assertIn('--kill-on-invalid-dep=yes',commands[2])
        self.assertIn('--time=02:30:00',commands[2])
        for name,budget in (('smoke',120.),('full',7200.)):
            plan=json.loads((run/name/'array_plan.json').read_text())
            self.assertEqual(plan['effective_settings']['workers'],32)
            self.assertEqual(plan['effective_settings']['time_limit_seconds'],budget)

    def test_algorithm_override_is_frozen_in_array_plan(self):
        run=self.root/'upstream_run'
        result=self.call('submit_esrf_array.sh','smoke','--algorithm','upstream',run)
        self.assertEqual(result.returncode,0,result.stderr)
        plan=json.loads((run/'array_plan.json').read_text())
        self.assertEqual(plan['effective_settings']['algorithm'],'upstream')
        self.assertNotEqual(self.call('submit_esrf_array.sh','smoke','--algorithm','unknown').returncode,0)

    def test_ladder_keeps_six_allocations_and_four_sequential_budgets(self):
        run=self.root/'ladder'
        result=self.call('submit_esrf_array.sh','ladder',run,'--algorithm','corrected')
        self.assertEqual(result.returncode,0,result.stderr)
        plan=json.loads((run/'array_plan.json').read_text())
        self.assertEqual(len(plan['tasks']),6)
        self.assertEqual([r['budget_seconds'] for r in plan['budget_runs']],[120.,600.,3600.,7200.])
        self.assertEqual(plan['effective_settings']['workers'],32)
        commands=[json.loads(line) for line in self.log.read_text().splitlines()]
        self.assertEqual(len(commands),2)
        self.assertIn('--time=04:00:00',commands[0])

    def test_collection_is_one_cpu_and_no_real_submission_inside_allocation(self):
        result=self.call('slurm_esrf_collect.sh',self.root/'run',SLURM_JOB_ID='502')
        self.assertEqual(result.returncode,0,result.stderr)
        args=self.capture.read_text().splitlines()
        self.assertIn('--cpus-per-task=1',args)
        self.assertIn('collect',args)
        self.assertNotEqual(self.call('submit_esrf_array.sh','full',SLURM_JOB_ID='99').returncode,0)
        self.assertFalse(self.log.exists())


if __name__=='__main__':
    unittest.main()
