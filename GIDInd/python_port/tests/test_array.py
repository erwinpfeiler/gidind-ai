"""Array planning/deduplication, isolated task dispatch and post-array collection."""
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from gidind_reference.config import Settings
from gidind_reference.geometry import reciprocal_metric, predict, hkl_grid
from gidind_reference.inputs import read_peaks, seed_coordinates
from gidind_reference.runner import atomic_json, port_fingerprint, write_outputs
from gidind_reference.slurm_array import prepare, run_task, collect, load_plan
from test_reference import synthetic, write_csv


def inputs(root, six=True):
    root.mkdir()
    _,metric,plane,_,coords=synthetic()
    if not six:
        cell=np.r_[[5.,7.,9.],np.deg2rad([93,98,104])]
        pp=predict(reciprocal_metric(cell),hkl_grid((1,1,1)),plane=plane)
        coords=pp[(pp[:,1]>.05)&(pp[:,0]>.05),:2]
    cases=[]
    for i,case in enumerate(('pq','mapbbr3','naib_s11','cu_bod_mof')):
        points=coords*(1+.02*i) if six else coords
        original=root/f'{case}.csv';filtered=root/f'{case}_without_specular.csv'
        write_csv(filtered,points)
        if six and i in (1,2):
            points=np.vstack([[0,np.sqrt(plane@metric@plane)*(1+.02*i)],points])
        write_csv(original,np.vstack([[0,0],points]))
        digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
        cases.append({'id':case,'filename':original.name,'without_specular':{
            'filename':filtered.name,'sha256':digest(filtered),'source_sha256':digest(original)}})
    (root/'manifest.json').write_text(json.dumps({'cases':cases}))
    return root


def fake_index(path,mode,out,settings):
    peaks=read_peaks(path,mode,settings)
    seed,rows=seed_coordinates(peaks,mode,settings)
    peaks.provenance.update(seed_coordinates=seed.tolist(),seed_source_rows=rows.tolist())
    result={'mode':mode,'input':peaks.provenance,'effective_settings':settings.to_dict(),
            'port_source_sha256':port_fingerprint(),'status':'no_solution','solutions':[],
            'timing':{'total_wall_seconds':1.},'search_work':None}
    write_outputs(out,result)
    return result


class ArrayTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.source=inputs(self.root/'inputs')
        self.out=self.root/'run'
        self.settings=Settings(workers=16,time_limit_seconds=43200.)

    def make_plan(self):
        return prepare(self.source,self.out,self.settings)

    def test_plan_six_unique_jobs_and_frozen_twelve_hour_settings(self):
        plan=self.make_plan()
        self.assertEqual(len(plan['tasks']),6)
        self.assertEqual(len(plan['entries']),8)
        self.assertEqual(plan['effective_settings']['workers'],16)
        self.assertEqual(plan['effective_settings']['time_limit_seconds'],43200)
        self.assertEqual(sum(e['mode']=='specular' for e in plan['entries']),2)
        for case in ('pq','cu_bod_mof'):
            entries=[e for e in plan['entries'] if e['case']==case]
            self.assertEqual(entries[0]['task_id'],entries[1]['task_id'])
            self.assertIsNotNone(entries[0]['specular_comparison_unavailable'])
        with self.assertRaises(FileExistsError): self.make_plan()

    def test_collection_missing_results_is_explicit_and_can_be_repeated(self):
        plan=self.make_plan()
        report=collect(self.out)
        self.assertEqual(report['status'],'incomplete')
        self.assertEqual(len(report['jobs']),8)
        self.assertTrue(all(j['status']=='missing_result' for j in report['jobs']))
        with patch('gidind_reference.slurm_array.index_file',side_effect=fake_index) as index:
            with patch.dict(os.environ,{'SLURM_CPUS_PER_TASK':'16'}):
                for task in reversed(plan['tasks']):
                    run_task(self.out,task['task_id'])
        self.assertEqual(index.call_count,6)
        report=collect(self.out)
        self.assertEqual(report['status'],'finished')
        self.assertEqual(sum(j['reused'] for j in report['jobs']),2)
        alias=json.loads((self.out/'pq_without_specular/results.json').read_text())
        self.assertFalse(alias['deduplication']['independent_timing_measurement'])
        self.assertEqual(collect(self.out),report)

    def test_failed_task_remains_failure_not_successful_exhaustion(self):
        plan=self.make_plan()
        with patch('gidind_reference.slurm_array.index_file',side_effect=fake_index):
            with patch.dict(os.environ,{'SLURM_CPUS_PER_TASK':'16'}):
                for task in plan['tasks']: run_task(self.out,task['task_id'])
        path=self.out/plan['tasks'][0]['output_name']/'results.json'
        data=json.loads(path.read_text());data['status']='failure';data['error']='test failure'
        atomic_json(path,data)
        report=collect(self.out)
        self.assertEqual(report['status'],'finished_with_failures')
        self.assertEqual(sum(j['status']=='failure' for j in report['jobs']),2)

    def test_invalid_id_cpu_mismatch_changed_input_and_code_are_rejected(self):
        plan=self.make_plan()
        for task_id in (-1,6):
            with self.assertRaises(ValueError):run_task(self.out,task_id)
        with patch.dict(os.environ,{'SLURM_CPUS_PER_TASK':'8'}):
            with self.assertRaisesRegex(ValueError,'allocation'):run_task(self.out,0)
        with patch('gidind_reference.slurm_array.port_fingerprint',return_value='changed'):
            with self.assertRaisesRegex(ValueError,'code changed'):load_plan(self.out)
        path=Path(plan['tasks'][0]['input_path']);path.write_text(path.read_text()+'\n')
        with self.assertRaisesRegex(ValueError,'input changed'):run_task(self.out,0)

    def test_duplicate_task_never_overwrites_existing_work(self):
        self.make_plan()
        with patch('gidind_reference.slurm_array.index_file',side_effect=fake_index):
            with patch.dict(os.environ,{'SLURM_CPUS_PER_TASK':'16'}):run_task(self.out,0)
        with patch.dict(os.environ,{'SLURM_CPUS_PER_TASK':'16'}):
            with self.assertRaises(FileExistsError):run_task(self.out,0)

    def test_collector_rejects_mismatched_result_settings(self):
        plan=self.make_plan()
        with patch('gidind_reference.slurm_array.index_file',side_effect=fake_index):
            with patch.dict(os.environ,{'SLURM_CPUS_PER_TASK':'16'}):run_task(self.out,0)
        path=self.out/plan['tasks'][0]['output_name']/'results.json'
        data=json.loads(path.read_text());data['effective_settings']['workers']=1
        atomic_json(path,data)
        with self.assertRaisesRegex(ValueError,'does not match'):collect(self.out)

    def test_real_array_task_and_alias_assignment_remapping(self):
        import shutil
        shutil.rmtree(self.source);inputs(self.source,six=False)
        settings=Settings(algorithm='upstream',workers=2,time_limit_seconds=300,without_specular_seed_hkl=[1,1,1],
                          planes=[[0,0,1]],rmse_cutoffs=[1e-6]*3)
        plan=prepare(self.source,self.out,settings)
        self.assertEqual(len(plan['tasks']),1)
        with patch.dict(os.environ,{'SLURM_CPUS_PER_TASK':'2'}):result=run_task(self.out,0)
        self.assertEqual(result['status'],'completed')
        report=collect(self.out)
        self.assertEqual(report['status'],'finished')
        self.assertEqual(sum(j['reused'] for j in report['jobs']),7)
        alias=json.loads((self.out/'pq_without_specular/results.json').read_text())
        self.assertEqual(alias['solutions'][0]['assignments'][0]['source_row'],2)
        self.assertTrue((self.out/'pq_without_specular/port_source_snapshot.zip').is_file())
        self.assertEqual(alias['search_work'],result['search_work'])


if __name__=='__main__':
    unittest.main()
