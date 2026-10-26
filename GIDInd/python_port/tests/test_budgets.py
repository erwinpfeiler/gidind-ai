"""Fixed-budget ladder: six task allocations, fresh independent searches."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from gidind_reference.config import Settings
from gidind_reference.budgets import validate_budgets,DEFAULT_BUDGETS
from gidind_reference.slurm_array import prepare_ladder,run_task,collect
from test_array import inputs,fake_index


class BudgetTests(unittest.TestCase):
    def test_validation(self):
        self.assertEqual(DEFAULT_BUDGETS,(120.,600.,3600.,7200.))
        for values in ([],[0],[True],[float('nan')],[2,1],[1,1]):
            with self.assertRaises(ValueError):validate_budgets(values)

    def test_sequential_budget_tasks_and_eight_entry_deduplication_per_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=inputs(root/'inputs')
            plan=prepare_ladder(source,root/'run',Settings(workers=2),[1,2])
            self.assertEqual(len(plan['tasks']),6)
            self.assertEqual([b['budget_seconds'] for b in plan['budget_runs']],[1.,2.])
            order=[]
            def index(path,mode,out,settings):
                order.append(settings.time_limit_seconds)
                return fake_index(path,mode,out,settings)
            with patch('gidind_reference.slurm_array.index_file',side_effect=index),patch.dict('os.environ',{'SLURM_CPUS_PER_TASK':'2'}):
                for task in plan['tasks']:self.assertEqual(run_task(root/'run',task['task_id'])['status'],'completed')
            self.assertEqual(order,[1.,2.]*6)
            summary=collect(root/'run')
            self.assertEqual(summary['status'],'finished')
            self.assertEqual(summary['unique_computations'],12)
            self.assertEqual(len(summary['jobs']),16)
            self.assertEqual(sum(j['reused'] for j in summary['jobs']),4)


if __name__=='__main__':unittest.main()
