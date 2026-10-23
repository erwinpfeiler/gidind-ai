import argparse
import json
from pathlib import Path
from .config import Settings


def main(argv=None):
    parser = argparse.ArgumentParser(description='Headless GIDInd Python reference port; timings apply to this port.')
    sub = parser.add_subparsers(dest='command',required=True)
    for name in ('index','benchmark'):
        p = sub.add_parser(name)
        p.add_argument('--input' if name=='index' else '--input-dir', required=True, type=Path)
        if name=='index':
            p.add_argument('--mode',choices=('specular','without-specular'),required=True)
        else:
            p.add_argument('--budgets-seconds',type=float,nargs='+',
                           help='Fresh independent runs at each increasing wall budget')
        p.add_argument('--output-dir',required=True,type=Path)
        p.add_argument('--config',type=Path)
        p.add_argument('--algorithm',choices=('corrected','upstream'),
                       help='corrected (default) or previous upstream-compatibility search; MATLAB parity unverified')
        p.add_argument('--plane-limit',type=int)
        p.add_argument('--planes',help='Explicit integer plane list as JSON, e.g. [[0,0,1],[1,0,0]]')
        p.add_argument('--time-limit-seconds',type=float)
        p.add_argument('--workers',type=int,help='CPU worker processes (default: 1; one BLAS thread each)')
    sub.add_parser('defaults',help='Print the effective default settings as JSON')
    p=sub.add_parser('evaluate',help='Post-run reference evaluation; never influences the search')
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--references',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--top-k',type=int)
    p.add_argument('--case',help='Reference case ID when evaluating one results.json directory')
    args = parser.parse_args(argv)
    if args.command=='defaults':
        print(json.dumps(Settings().to_dict(),indent=2)); return 0
    try:
        if args.command=='evaluate':
            from .evaluation import evaluate_run
            result=evaluate_run(args.run_dir,args.references,args.output_dir,top_k=args.top_k,case=args.case)
            return 0 if result['status']=='finished' else 1
        settings = Settings.load(args.config,plane_limit=args.plane_limit,
                                 planes=json.loads(args.planes) if args.planes else None,
                                 time_limit_seconds=args.time_limit_seconds, workers=args.workers,
                                 algorithm=args.algorithm)
        if args.command=='index':
            from .supervision import index_file
            result = index_file(args.input,args.mode,args.output_dir,settings)
            return 1 if result['status']=='failure' else 130 if result['status']=='interrupted' else 0
        if args.budgets_seconds:
            from .budgets import benchmark_ladder
            result=benchmark_ladder(args.input_dir,args.output_dir,settings,args.budgets_seconds)
        else:
            from .batch import benchmark
            result = benchmark(args.input_dir,args.output_dir,settings)
        return 1 if result['status']=='finished_with_failures' else 130 if result['status']=='interrupted' else 0
    except (ValueError,TypeError,OSError,KeyError) as exc:
        parser.exit(1,f'{type(exc).__name__}: {exc}\n')


if __name__ == '__main__':
    raise SystemExit(main())
