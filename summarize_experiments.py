import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))

import matplotlib.pyplot as plt
import numpy as np

from ablation_registry import EXPERIMENT_INDEX, resolve_experiments


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize ablation experiments and plot training curves.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--experiments", type=str, default="all")
    return parser.parse_args()


def read_json(path):
    return json.loads(path.read_text())


def read_history(path):
    rows = []
    with path.open() as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            parsed = {"epoch": int(row["epoch"]), "train_loss": float(row["train_loss"]), "lr": float(row["lr"])}
            valid_loss = row.get("valid_loss")
            valid_dsc = row.get("valid_dsc")
            parsed["valid_loss"] = None if valid_loss in (None, "", "None") else float(valid_loss)
            parsed["valid_dsc"] = None if valid_dsc in (None, "", "None") else float(valid_dsc)
            rows.append(parsed)
    return rows


def first_existing(paths):
    for path in paths:
        if path.exists():
            return path
    return None


def load_experiment(output_root, experiment):
    experiment_dir = output_root / experiment["name"]
    config_path = experiment_dir / "config.json"
    train_metrics_path = experiment_dir / "metrics.json"
    eval_metrics_path = experiment_dir / "evaluation" / "metrics.json"
    history_path = experiment_dir / "history.csv"
    if not all(path.exists() for path in [config_path, train_metrics_path, eval_metrics_path, history_path]):
        return None
    return {
        "meta": experiment,
        "dir": experiment_dir,
        "config": read_json(config_path),
        "train_metrics": read_json(train_metrics_path),
        "eval_metrics": read_json(eval_metrics_path),
        "history": read_history(history_path),
    }


def write_summary_csv(path, records):
    fieldnames = [
        "name",
        "label",
        "description",
        "best_epoch",
        "best_validation_mean_dsc",
        "evaluation_median_dsc",
        "optimizer",
        "scheduler",
        "loss",
        "lr",
        "weight_decay",
        "batch_norm",
        "bilinear",
        "dropout",
        "aug_scale",
        "aug_angle",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def write_summary_md(path, records):
    lines = [
        "# Ablation Summary",
        "",
        "| Experiment | Description | Best Epoch | Mean DSC | Median DSC |",
        "| --- | --- | --- | --- | --- |",
    ]
    for record in records:
        lines.append(
            "| {label} | {description} | {best_epoch} | {best_validation_mean_dsc:.4f} | {evaluation_median_dsc:.4f} |".format(
                **record
            )
        )
    path.write_text("\n".join(lines) + "\n")


def plot_curves(records, output_dir):
    if not records:
        return

    plot_specs = [
        ("train_loss", "Train Loss", "train_loss_curves.png"),
        ("valid_loss", "Validation Loss", "valid_loss_curves.png"),
        ("valid_dsc", "Validation Mean DSC", "valid_dsc_curves.png"),
    ]

    for key, title, filename in plot_specs:
        plt.figure(figsize=(10, 6))
        plotted = False
        for record in records:
            epochs = []
            values = []
            for row in record["history_rows"]:
                value = row[key]
                if value is None:
                    continue
                epochs.append(row["epoch"])
                values.append(value)
            if not values:
                continue
            plotted = True
            plt.plot(epochs, values, marker="o", linewidth=2, label=record["label"])
        if not plotted:
            plt.close()
            continue
        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel(title)
        plt.grid(alpha=0.3, linestyle="--")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / filename, dpi=200)
        plt.close()

    plt.figure(figsize=(12, 6))
    labels = [record["label"] for record in records]
    scores = [record["best_validation_mean_dsc"] for record in records]
    x = np.arange(len(labels))
    plt.bar(x, scores, color="skyblue")
    plt.xticks(x, labels, rotation=30, ha="right")
    plt.ylabel("Best Validation Mean DSC")
    plt.title("Ablation Results")
    plt.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig(output_dir / "best_dsc_bar.png", dpi=200)
    plt.close()


def main():
    args = parse_args()
    args.output_root = args.output_root.resolve()
    selected_names = None if args.experiments == "all" else [name.strip() for name in args.experiments.split(",") if name.strip()]
    experiments = resolve_experiments(selected_names)

    loaded = []
    for experiment in experiments:
        record = load_experiment(args.output_root, experiment)
        if record is not None:
            loaded.append(record)

    summary_dir = args.output_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    plot_records = []
    for item in loaded:
        config = item["config"]
        train_metrics = item["train_metrics"]
        eval_metrics = item["eval_metrics"]
        row = {
            "name": item["meta"]["name"],
            "label": item["meta"]["label"],
            "description": item["meta"]["description"],
            "best_epoch": train_metrics["best_epoch"],
            "best_validation_mean_dsc": float(eval_metrics["evaluation_mean_dsc"]),
            "evaluation_median_dsc": float(eval_metrics["evaluation_median_dsc"]),
            "optimizer": config["optimizer"],
            "scheduler": config["scheduler"],
            "loss": config["loss"],
            "lr": float(config["lr"]),
            "weight_decay": float(config["weight_decay"]),
            "batch_norm": bool(config["batch_norm"]),
            "bilinear": bool(config["bilinear"]),
            "dropout": float(config["dropout"]),
            "aug_scale": float(config["aug_scale"]),
            "aug_angle": float(config["aug_angle"]),
        }
        summary_rows.append(row)
        plot_records.append(
            {
                "label": item["meta"]["label"],
                "best_validation_mean_dsc": row["best_validation_mean_dsc"],
                "history_rows": item["history"],
            }
        )

    write_summary_csv(summary_dir / "ablation_results.csv", summary_rows)
    write_summary_md(summary_dir / "ablation_results.md", summary_rows)
    (summary_dir / "ablation_results.json").write_text(json.dumps(summary_rows, indent=2))
    plot_curves(plot_records, summary_dir)

    print(json.dumps({"summary_dir": str(summary_dir), "num_experiments": len(summary_rows)}, indent=2))


if __name__ == "__main__":
    main()
