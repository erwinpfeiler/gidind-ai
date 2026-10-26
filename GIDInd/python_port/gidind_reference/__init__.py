"""GIDInd Python translation; derived parts are GPL-3.0-or-later.

Original GIDInd: M. Kainz and J. Simbrunner, Graz University of Technology (2021).
See ../PROVENANCE.md for source mappings and numerical adaptations.
"""
import os

# Set before importing NumPy, including when the Python API is used directly.
for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_key] = "1"

__version__ = "0.5.0"
UPSTREAM_COMMIT = "d15a4b8808d20869e7b92c87bff04a66d9c58a3b"
IMPLEMENTATION = "Python port of GIDInd with discrete-orientation restriction"

def index_file(*args, **kwargs):
    """Run indexing; see runner.index_file for arguments and result contract."""
    from .runner import index_file as run
    return run(*args, **kwargs)
