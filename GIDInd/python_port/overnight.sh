#!/bin/sh
set -eu
port_root="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
repo_root="$(CDPATH= cd -- "$port_root/../.." && pwd)"
workers=1
algorithm=
requested_output=
while [ "$#" -gt 0 ]; do
    case "$1" in
        --workers)
            [ "$#" -ge 2 ] || { echo "--workers requires a positive integer" >&2; exit 1; }
            workers="$2"; shift 2 ;;
        --algorithm)
            [ "$#" -ge 2 ] || { echo '--algorithm requires corrected or upstream' >&2; exit 1; }
            algorithm=$2; shift 2 ;;
        --*) echo "Unknown option: $1" >&2; exit 1 ;;
        *)
            [ -z "$requested_output" ] || { echo "Usage: $0 [output-directory] [--workers N]" >&2; exit 1; }
            requested_output="$1"; shift ;;
    esac
done
case "$algorithm" in ''|corrected|upstream) ;; *) echo '--algorithm requires corrected or upstream' >&2; exit 1 ;; esac
case "$workers" in ''|*[!0-9]*) echo "--workers requires a positive integer" >&2; exit 1 ;; esac
case "$workers" in *[1-9]*) ;; *) echo "--workers requires a positive integer" >&2; exit 1 ;; esac
if [ ! -x "$port_root/.venv/bin/python" ]; then
    echo "First run: $port_root/setup.sh" >&2
    exit 1
fi
input_dir="$repo_root/gidind_peak_lists_all"
need_download=false
if [ ! -f "$input_dir/manifest.json" ]; then need_download=true; fi
for sample in pq mapbbr3 naib_s11 cu_bod_mof; do
    for suffix in '' _without_specular; do
        if [ ! -f "$input_dir/${sample}${suffix}.csv" ]; then need_download=true; fi
    done
done
if [ "$need_download" = true ]; then
    "$port_root/.venv/bin/python" "$repo_root/scripts/download_real_case_peak_lists.py" \
        --include-originals --output-dir "$input_dir"
fi
output_dir="${requested_output:-$port_root/benchmark_runs/overnight_$(date -u +%Y%m%dT%H%M%SZ)}"
echo "Benchmark output: $output_dir"
echo "CPU-only, $workers worker(s), one BLAS thread each; two-hour budget per unique input/mode."
set --
if [ -n "$algorithm" ]; then set -- --algorithm "$algorithm"; fi
exec "$port_root/run.sh" benchmark --input-dir "$input_dir" \
    --config "$port_root/configs/benchmark_settings.json" --output-dir "$output_dir" "$@" --workers "$workers"
