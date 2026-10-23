# Download experimental peak lists

From a clone of this repository, run:

```sh
python3 scripts/download_real_case_peak_lists.py
```

This downloads the four specular-free experimental peak lists for PQ, Cu-BOD,
MAPbBr3 and NAIB-S11 into `gidind_peak_lists/`, together with their manifest.
It needs Python 3.10 or newer and no additional packages, MATLAB, model weights,
or synthetic benchmark assets.

To download all eight CSVs, including the unchanged original lists:

```sh
python3 scripts/download_real_case_peak_lists.py \
  --include-originals --output-dir gidind_peak_lists_all
```

The script pins dataset `epfeiler/gidind-ai-paper-assets` to revision
`9c2aef88ad916c6d353e2664c4185fdebe159216`. It verifies the manifest against
a checksum embedded in the script, then verifies every CSV against that manifest
before saving any files. It preserves the CSV bytes and refuses to overwrite
different local files. Repeating the command with matching files is safe.
If the dataset requires authentication, the script uses `HF_TOKEN` or the locally
cached Hugging Face login (`hf auth login`). It also respects `HF_HOME`,
`HF_TOKEN_PATH`, and `XDG_CACHE_HOME`. No token is written to the downloaded files.

The files retain original columns, ordering and coordinate precision. Read the
named `q_xy` and `q_z` columns because their positions differ among source lists.
The specular-free versions exclude `|q_xy| <= 0.05 Å⁻¹` and zero placeholders.
PQ's original has no measured specular peaks; Cu-BOD's two versions are identical.
