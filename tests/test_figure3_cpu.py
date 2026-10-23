"""CPU condition shards must preserve counts and reject misassigned fits."""

import argparse
import json

import pytest

from gidind_ai.figure3_benchmark import condition_grid, database
from gidind_ai.figure3_cpu import merge_condition


def test_sparse_and_full_grids_allocate_one_worker_per_distinct_condition():
    args = argparse.Namespace(
        baseline_only=False,
        max_peaks=20,
        noise_steps=21,
        max_noise=0.02,
        peak_step=1,
        condition_index=None,
    )
    full = condition_grid(args)
    assert len(full) == 61
    assert len(set(full)) == 61
    assert ("noise", 0.0) not in full
    args.peak_step, args.noise_steps = 5, 5
    assert len(condition_grid(args)) == 13
    args.condition_index = 12
    assert condition_grid(args) == [("noise", 0.02)]


def test_resuming_and_merging_a_shard_never_duplicates_evaluations(tmp_path):
    parent = database(tmp_path)
    shard = tmp_path / "condition_0"
    shard.mkdir()
    child = database(shard)
    with child:
        child.execute("INSERT INTO fits VALUES('peaks',0,0,?)", (json.dumps({"correct": True}),))
    assert merge_condition(parent, shard, "peaks", 0.0, 2) == 1
    # An interrupted condition finishes its second evaluation on resume.
    with child:
        child.execute("INSERT INTO fits VALUES('peaks',0,1,?)", (json.dumps({"correct": False}),))
    assert merge_condition(parent, shard, "peaks", 0.0, 2) == 2
    assert merge_condition(parent, shard, "peaks", 0.0, 2) == 2
    assert parent.execute("SELECT count(*) FROM fits").fetchone()[0] == 2
    child.close()
    parent.close()


@pytest.mark.parametrize(
    "arm,value,sample", [("noise", 0.01, 0), ("peaks", -20, 0), ("peaks", 0, 2)]
)
def test_wrong_conditions_and_sample_ids_are_rejected_without_polluting_parent(
    tmp_path, arm, value, sample
):
    parent = database(tmp_path)
    shard = tmp_path / "condition_0"
    shard.mkdir()
    child = database(shard)
    with child:
        child.execute("INSERT INTO fits VALUES(?,?,?,?)", (arm, value, sample, "{}"))
    child.close()
    with pytest.raises(ValueError, match="Unexpected evaluation"):
        merge_condition(parent, shard, "peaks", 0.0, 2)
    assert parent.execute("SELECT count(*) FROM fits").fetchone()[0] == 0
    parent.close()
