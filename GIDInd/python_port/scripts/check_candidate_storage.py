"""Synthetic storage/RAM check, not a physical indexing or scaling benchmark.

Run with the standalone Python: scripts/check_candidate_storage.py --output-dir NEW_DIR
"""
import argparse
import json
from pathlib import Path
import resource
import sys
import time
import numpy as np
from gidind_reference import __version__
from gidind_reference.config import Settings
from gidind_reference.runner import port_fingerprint
from gidind_reference.state import Candidate, State


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--rows',type=int,default=100005)
    args=parser.parse_args()
    if args.rows<1:parser.error('--rows must be positive')
    args.output_dir.mkdir(parents=True,exist_ok=False)
    state=State(Settings(),candidate_path=args.output_dir/'native_candidates.sqlite3')
    before=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    started=time.perf_counter()
    try:
        for i in range(args.rows):
            state.add_pending(Candidate(np.eye(3),np.array([[i,0,1]]),
                native_scores={'seed_linear_sumerr':0. if i==args.rows-1 else 1.+i%17}))
        state.pending.flush()
        best=next(iter(state.pending.view()))
        assert best.hkl[0,0]==args.rows-1
        result=dict(synthetic_storage_check=True,not_a_physical_indexing_benchmark=True,
            port_version=__version__,port_source_sha256=port_fingerprint(),
            rows=len(state.pending),wall_seconds=time.perf_counter()-started,
            maxrss_before=before,maxrss_after=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            maxrss_unit='bytes' if sys.platform=='darwin' else 'KiB',
            metadata=state.pending.metadata(),late_best_candidate_preserved=True,
            assignment_dictionary_entries=len(state.pending_keys),
            timing_dictionary_entries=len(state.candidate_timings),partial_reasons=state.partial_reasons)
        (args.output_dir/'report.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result,indent=2))
    finally:state.pending.close()


if __name__=='__main__':main()
