"""Command-line interface for the reviewer-facing gidind-ai workflows."""

from __future__ import annotations

import argparse
import sys

from .cod_prepare import prepare_cod_snapshot
from .paper_assets import paper_asset_release, resolve_paper_assets
from .release_models import real_case_profile, release_model
from .workflows import (
    DEFAULT_MODEL_REPOSITORY,
    DEFAULT_MODEL_REVISION,
    WorkflowError,
    resolve_checkpoint,
    run_benchmark,
    run_inference,
)


def _model_options(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--checkpoint", help="Path to a local Lightning .ckpt file.")
    group.add_argument("--model-repo", help="Hugging Face model repository, e.g. OWNER/gidind-ai.")
    group.add_argument(
        "--release-model",
        help="Pinned release model declared in gidind-ai's packaged manifest.",
    )
    parser.add_argument(
        "--model-file",
        default="default_model.ckpt",
        help="Filename within --model-repo (default: default_model.ckpt).",
    )
    parser.add_argument("--revision", help="Immutable Hugging Face revision/tag for --model-repo.")


def _mcmc_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--refinement-algorithm", choices=("gradient_laplace", "mcmc"), default="gradient_laplace"
    )
    parser.add_argument(
        "--q-sigma",
        type=float,
        help="Required for MCMC: measured coordinate uncertainty in inverse angstroms.",
    )
    for flag, default in (
        ("ensembles", 4),
        ("walkers", 32),
        ("burn-in", 2000),
        ("production", 4000),
        ("thin", 2),
        ("seed", 42),
        ("initialization-attempts", 1000),
        ("orientation-chunk-size", 1),
        ("reflection-chunk-size", 256),
    ):
        parser.add_argument(f"--mcmc-{flag}", type=int, default=default)
    for flag, float_default in (
        ("gamma", 2.38 / (12**0.5)),
        ("jitter", 1e-6),
        ("initial-sigma", 0.005),
    ):
        parser.add_argument(f"--mcmc-{flag}", type=float, default=float_default)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gidind-ai", description="gidind-ai reviewer workflows")
    commands = parser.add_subparsers(dest="command", required=True)
    infer = commands.add_parser("infer", help="Infer and refine a real peak-list input.")
    infer.add_argument("--input", required=True, help="CSV, NPY/NPZ, or HDF5 peak-list input.")
    infer.add_argument(
        "--output-dir", required=True, help="Directory for JSON, report, and figures."
    )
    infer.add_argument(
        "--q-scale", type=float, help="Positive physical q scale; infer automatically when omitted."
    )
    infer.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    infer.add_argument("--workers", default=1, type=int)
    infer.add_argument(
        "--refinement-backend", default="auto", choices=("auto", "serial", "batched")
    )
    infer.add_argument("--no-figures", action="store_true", help="Skip reconstruction PNGs.")
    infer.add_argument(
        "--case-profile",
        help="Use the declared q scaling and refinement settings for an approved real case.",
    )
    _mcmc_options(infer)
    _model_options(infer)
    benchmark = commands.add_parser(
        "benchmark", help="Run the deterministic fast synthetic benchmark."
    )
    benchmark.add_argument("--output-dir", required=True)
    benchmark.add_argument(
        "--samples", default=4, type=int, help="Small CPU-friendly sample count."
    )
    benchmark.add_argument("--seed", default=42, type=int)
    benchmark.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    benchmark.add_argument("--workers", default=1, type=int)
    benchmark.add_argument(
        "--refinement-backend", default="auto", choices=("auto", "serial", "batched")
    )
    asset_group = benchmark.add_mutually_exclusive_group()
    asset_group.add_argument("--paper-assets", help="Local immutable paper-asset directory.")
    asset_group.add_argument("--paper-assets-repo", help="Hugging Face Dataset repository.")
    asset_group.add_argument(
        "--paper-assets-release",
        default="paper-v1",
        help="Pinned paper-asset release declared by gidind-ai (default: paper-v1).",
    )
    benchmark.add_argument(
        "--paper-assets-revision",
        help="Immutable Hugging Face Dataset revision required with --paper-assets-repo.",
    )
    _mcmc_options(benchmark)
    _model_options(benchmark)
    cod = commands.add_parser("cod", help="Prepare an auditable local COD snapshot.")
    cod_commands = cod.add_subparsers(dest="cod_command", required=True)
    prepare = cod_commands.add_parser(
        "prepare", help="Parse a pinned local CIF snapshot and make deterministic splits."
    )
    prepare.add_argument(
        "--cif-dir", required=True, help="Root directory of a downloaded CIF snapshot."
    )
    prepare.add_argument(
        "--output-dir", required=True, help="Empty directory for the derived release."
    )
    prepare.add_argument(
        "--source-url", required=True, help="Upstream URL used to obtain the snapshot."
    )
    prepare.add_argument(
        "--source-revision",
        required=True,
        help="Immutable snapshot label and/or archive SHA-256; never use an unpinned live URL alone.",
    )
    prepare.add_argument("--seed", default=42, type=int)
    prepare.add_argument("--test-fraction", default=0.2, type=float)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        mcmc_options = {}
        if getattr(args, "refinement_algorithm", None) == "mcmc":
            from .mcmc import MCMCSettings, validate_sigma

            validate_sigma(args.q_sigma)
            settings = MCMCSettings(
                **{
                    key.removeprefix("mcmc_"): value
                    for key, value in vars(args).items()
                    if key.startswith("mcmc_")
                }
            )
            mcmc_options = {
                "refinement_algorithm": "mcmc",
                "q_sigma": args.q_sigma,
                "mcmc_settings": settings,
            }
        if args.command == "cod":
            prepared = prepare_cod_snapshot(
                cif_dir=args.cif_dir,
                output_dir=args.output_dir,
                source_url=args.source_url,
                source_revision=args.source_revision,
                seed=args.seed,
                test_fraction=args.test_fraction,
            )
            print(
                f"Prepared {prepared.accepted_entries} COD entries "
                f"({prepared.rejected_entries} rejected): {prepared.manifest_path}"
            )
            return 0
        if args.checkpoint and (args.revision or args.model_file != "default_model.ckpt"):
            raise WorkflowError("--revision may only be used with --model-repo.")
        if args.release_model and (args.revision or args.model_file != "default_model.ckpt"):
            raise WorkflowError(
                "--revision and --model-file may not be combined with --release-model."
            )

        profile = (
            real_case_profile(args.case_profile) if getattr(args, "case_profile", None) else None
        )
        if args.release_model:
            declared_model = release_model(args.release_model)
        elif profile:
            declared_model = release_model(profile["model_id"])
        else:
            declared_model = None
        if profile and args.release_model and profile["model_id"] != args.release_model:
            raise WorkflowError(
                f"Case profile {profile['id']!r} requires release model {profile['model_id']!r}."
            )
        if profile and not args.checkpoint:
            assert declared_model is not None
            for option, supplied, expected in (
                ("--model-repo", args.model_repo, declared_model["repository"]),
                ("--revision", args.revision, declared_model["revision"]),
                ("--model-file", args.model_file, declared_model["filename"]),
            ):
                default_filename = option == "--model-file" and supplied == "default_model.ckpt"
                if supplied and not default_filename and supplied != expected:
                    raise WorkflowError(
                        f"Case profile {profile['id']!r} requires {option} {expected!r}."
                    )

        model_repo = args.model_repo
        revision = args.revision
        filename = args.model_file
        model_id = None
        sha256 = None
        if declared_model and not args.checkpoint:
            if declared_model["revision"].startswith("__"):
                raise WorkflowError(f"Release model {declared_model['id']!r} is not published yet.")
            model_repo = declared_model["repository"]
            revision = declared_model["revision"]
            filename = declared_model["filename"]
            model_id = declared_model["id"]
            sha256 = declared_model["sha256"]
        elif declared_model:
            # A profile may be exercised against an explicitly supplied local
            # checkpoint, but its manifest digest cannot attest to that file.
            model_id = declared_model["id"]
        elif not args.checkpoint and model_repo is None:
            model_repo = DEFAULT_MODEL_REPOSITORY
            revision = DEFAULT_MODEL_REVISION
        elif model_repo == DEFAULT_MODEL_REPOSITORY and revision is None:
            revision = DEFAULT_MODEL_REVISION
        checkpoint = resolve_checkpoint(
            checkpoint=args.checkpoint,
            model_repo=model_repo,
            revision=revision,
            filename=filename,
            model_id=model_id,
            sha256=sha256,
        )
        if args.command == "infer":
            refinement = (profile or {}).get("refinement") or (
                declared_model["refinement"] if declared_model else None
            )
            run_inference(
                input_path=args.input,
                output_dir=args.output_dir,
                checkpoint_info=checkpoint,
                device_name=args.device,
                workers=args.workers,
                refinement_backend=args.refinement_backend,
                **mcmc_options,
                q_scale=args.q_scale
                if args.q_scale is not None
                else (profile or {}).get("q_scale"),
                figures=not args.no_figures,
                refinement=refinement,
                case_profile=(profile or {}).get("id"),
            )
        else:
            assets_dir = args.paper_assets
            assets_repo = args.paper_assets_repo
            assets_revision = args.paper_assets_revision
            if not assets_dir and not assets_repo:
                declared_assets = paper_asset_release(args.paper_assets_release)
                assets_repo = declared_assets["repository"]
                assets_revision = declared_assets["revision"]
            elif assets_dir and assets_revision:
                raise WorkflowError(
                    "--paper-assets-revision may only be used with --paper-assets-repo."
                )
            paper_assets = resolve_paper_assets(
                assets_dir=assets_dir,
                assets_repo=assets_repo,
                revision=assets_revision,
            )
            run_benchmark(
                output_dir=args.output_dir,
                checkpoint_info=checkpoint,
                samples=args.samples,
                seed=args.seed,
                device_name=args.device,
                workers=args.workers,
                refinement_backend=args.refinement_backend,
                **mcmc_options,
                paper_assets=paper_assets,
            )
    except (WorkflowError, ImportError, OSError, ValueError) as exc:
        print(f"gidind-ai: error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
