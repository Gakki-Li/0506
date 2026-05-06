import argparse
import json
import subprocess
import sys
from pathlib import Path

from ablation_registry import list_experiment_names, resolve_experiments


def default_images_dir():
    return Path(__file__).resolve().parents[2] / "BrainMRI" / "kaggle_3m"


def parse_args():
    parser = argparse.ArgumentParser(description="Run baseline, ablations, and combined improvements.")
    parser.add_argument("--images", type=Path, default=default_images_dir())
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parent / "outputs")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--steps-per-epoch", type=int, default=0)
    parser.add_argument("--validate-every", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--validation-cases", type=int, default=10)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--cache-dir", type=Path, default=Path(__file__).resolve().parent / ".cache")
    parser.add_argument("--preprocess-workers", type=int, default=4)
    parser.add_argument("--vis-images", type=int, default=200)
    parser.add_argument("--vis-freq", type=int, default=10)
    parser.add_argument("--init-features", type=int, default=32)
    parser.add_argument("--flip-prob", type=float, default=0.5)
    parser.add_argument("--experiments", type=str, default="all", help="Comma-separated experiment names or 'all'.")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--list-experiments", action="store_true")
    return parser.parse_args()


def common_train_args(args, experiment_name):
    return [
        sys.executable,
        str(Path(__file__).resolve().parent / "train.py"),
        "--images",
        str(args.images),
        "--output-root",
        str(args.output_root),
        "--experiment-name",
        experiment_name,
        "--device",
        args.device,
        "--seed",
        str(args.seed),
        "--batch-size",
        str(args.batch_size),
        "--eval-batch-size",
        str(args.eval_batch_size),
        "--epochs",
        str(args.epochs),
        "--steps-per-epoch",
        str(args.steps_per_epoch),
        "--validate-every",
        str(args.validate_every),
        "--workers",
        str(args.workers),
        "--validation-cases",
        str(args.validation_cases),
        "--image-size",
        str(args.image_size),
        "--cache-dir",
        str(args.cache_dir),
        "--preprocess-workers",
        str(args.preprocess_workers),
        "--vis-images",
        str(args.vis_images),
        "--vis-freq",
        str(args.vis_freq),
        "--init-features",
        str(args.init_features),
        "--flip-prob",
        str(args.flip_prob),
    ]


def common_eval_args(args, experiment_dir):
    return [
        sys.executable,
        str(Path(__file__).resolve().parent / "evaluate.py"),
        "--images",
        str(args.images),
        "--experiment-dir",
        str(experiment_dir),
        "--device",
        args.device,
        "--batch-size",
        str(args.eval_batch_size),
        "--workers",
        str(args.workers),
        "--validation-cases",
        str(args.validation_cases),
        "--cache-dir",
        str(args.cache_dir),
        "--preprocess-workers",
        str(args.preprocess_workers),
    ]


def run_command(cmd):
    print("$ {}".format(" ".join(cmd)))
    subprocess.run(cmd, check=True)


def metrics_exist(experiment_dir):
    return (
        (experiment_dir / "metrics.json").exists()
        and (experiment_dir / "evaluation" / "metrics.json").exists()
    )


def write_manifest(args, experiments):
    manifest = {
        "images": str(args.images),
        "output_root": str(args.output_root),
        "device": args.device,
        "common_config": {
            "seed": args.seed,
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "epochs": args.epochs,
            "steps_per_epoch": args.steps_per_epoch,
            "validate_every": args.validate_every,
            "workers": args.workers,
            "validation_cases": args.validation_cases,
            "image_size": args.image_size,
            "cache_dir": str(args.cache_dir),
            "preprocess_workers": args.preprocess_workers,
            "vis_images": args.vis_images,
            "vis_freq": args.vis_freq,
            "init_features": args.init_features,
            "flip_prob": args.flip_prob,
        },
        "experiments": experiments,
    }
    (args.output_root / "ablation_manifest.json").write_text(json.dumps(manifest, indent=2))


def run_summary(args, experiment_names):
    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent / "summarize_experiments.py"),
        "--output-root",
        str(args.output_root),
        "--experiments",
        ",".join(experiment_names),
    ]
    run_command(cmd)


def main():
    args = parse_args()
    if args.list_experiments:
        for name in list_experiment_names():
            print(name)
        return

    args.images = args.images.resolve()
    args.output_root = args.output_root.resolve()
    args.cache_dir = args.cache_dir.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    selected_names = None if args.experiments == "all" else [name.strip() for name in args.experiments.split(",") if name.strip()]
    experiments = resolve_experiments(selected_names)
    write_manifest(args, experiments)

    if not args.summary_only:
        for experiment in experiments:
            experiment_dir = args.output_root / experiment["name"]
            if args.skip_existing and metrics_exist(experiment_dir):
                print("Skip existing experiment: {}".format(experiment["name"]))
                continue

            train_cmd = common_train_args(args, experiment["name"]) + experiment["extra_args"]
            run_command(train_cmd)

            if not args.skip_eval:
                eval_cmd = common_eval_args(args, experiment_dir)
                run_command(eval_cmd)

    run_summary(args, [experiment["name"] for experiment in experiments])


if __name__ == "__main__":
    main()
