"""Local shell-dispatch checks; no scheduler or cluster access is used."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class SlurmLauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.port=self.root/'GIDInd/python_port';self.port.mkdir(parents=True)
        self.script=self.port/'slurm_esrf.sh'
        shutil.copy2(Path(__file__).resolve().parents[1]/'slurm_esrf.sh',self.script)
        python=self.port/'.venv/bin/python';python.parent.mkdir(parents=True)
        python.write_text('#!/bin/sh\ncat >/dev/null\nexit "${PREFLIGHT_EXIT:-0}"\n');python.chmod(0o755)
        binpath=self.root/'bin';binpath.mkdir()
        srun=binpath/'srun';srun.write_text("#!/bin/sh\nprintf '%s\\n' \"$@\" > \"$SLURM_TEST_CAPTURE\"\n");srun.chmod(0o755)
        self.capture=self.root/'srun_args.txt'
        self.env={k:v for k,v in os.environ.items() if not k.startswith('SLURM_')}
        self.env.update(GIDIND_PROJECT_ROOT=str(self.root),SLURM_TEST_CAPTURE=str(self.capture),
                        PATH=str(binpath)+os.pathsep+os.environ['PATH'])

    def run_script(self,mode,**env):
        return subprocess.run(['bash',str(self.script),mode],env=dict(self.env,**env),
                              text=True,capture_output=True)

    def test_preflight_never_submits_or_indexes_and_smoke_requires_slurm(self):
        result=self.run_script('preflight')
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertFalse(self.capture.exists())
        self.assertNotEqual(self.run_script('smoke').returncode,0)
        self.assertFalse(self.capture.exists())
        self.assertNotEqual(self.run_script('invalid').returncode,0)

    def test_smoke_uses_all_requested_cpus_and_short_budget(self):
        result=self.run_script('smoke',SLURM_JOB_ID='123',SLURM_CPUS_PER_TASK='4')
        self.assertEqual(result.returncode,0,result.stderr)
        args=self.capture.read_text().splitlines()
        self.assertIn('--cpus-per-task=4',args)
        self.assertEqual(args[args.index('--workers')+1],'4')
        self.assertEqual(args[args.index('--time-limit-seconds')+1],'30')
        self.assertEqual(args[args.index('--output-dir')+1],str(self.port/'benchmark_runs/esrf_smoke_123'))

    def test_full_uses_configured_budget_and_refuses_existing_output(self):
        result=self.run_script('full',SLURM_JOB_ID='456',SLURM_CPUS_PER_TASK='8')
        self.assertEqual(result.returncode,0,result.stderr)
        args=self.capture.read_text().splitlines()
        self.assertNotIn('--time-limit-seconds',args)
        self.assertEqual(args[args.index('--workers')+1],'8')
        (self.port/'benchmark_runs/esrf_full_456').mkdir(parents=True)
        self.capture.unlink()
        self.assertNotEqual(self.run_script('full',SLURM_JOB_ID='456',SLURM_CPUS_PER_TASK='8').returncode,0)
        self.assertFalse(self.capture.exists())

    def test_failed_preflight_and_invalid_allocation_do_not_launch_srun(self):
        self.assertEqual(self.run_script('full',SLURM_JOB_ID='789',PREFLIGHT_EXIT='1').returncode,1)
        self.assertFalse(self.capture.exists())
        for cpus in ('0','00','bad','-1'):
            self.assertNotEqual(self.run_script('full',SLURM_JOB_ID='789',SLURM_CPUS_PER_TASK=cpus).returncode,0)
            self.assertFalse(self.capture.exists())


if __name__=='__main__':
    unittest.main()
