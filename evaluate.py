import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg
from medpy.filter.binary import largest_connected_component
from skimage.io import imsave
from torch.utils.data import DataLoader

from dataset import BrainSegmentationDataset as Dataset
from unet import UNet
from utils import dsc, gray2rgb, outline


def default_images_dir():
    return Path(__file__).resolve().parents[2] / "BrainMRI" / "kaggle_3m"


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained U-Net experiment.")
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--images", type=Path, default=default_images_dir())
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--validation-cases", type=int, default=10)
    parser.add_argument("--cache-dir", type=Path, default=Path(__file__).resolve().parent / ".cache")
    parser.add_argument("--preprocess-workers", type=int, default=4)
    parser.add_argument("--export-all-slices", action="store_true")
    return parser.parse_args()


def resolve_device(device_arg):
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def data_loader(args, image_size, seed):
    dataset = Dataset(
        images_dir=str(args.images),
        subset="validation",
        image_size=image_size,
        random_sampling=False,
        validation_cases=args.validation_cases,
        cache_dir=str(args.cache_dir),
        seed=seed,
        preprocess_workers=args.preprocess_workers,
    )
    return DataLoader(dataset, batch_size=args.batch_size, drop_last=False, num_workers=args.workers)


def postprocess_per_volume(input_list, pred_list, true_list, patient_slice_index, patients):
    volumes = {}
    num_slices = np.bincount([p[0] for p in patient_slice_index])
    index = 0
    for patient_idx, num_patient_slices in enumerate(num_slices):
        volume_in = np.array(input_list[index : index + num_patient_slices])
        volume_pred = np.round(np.array(pred_list[index : index + num_patient_slices])).astype(int)
        volume_true = np.array(true_list[index : index + num_patient_slices])
        if volume_pred.ndim == 4 and volume_pred.shape[1] == 1:
            volume_pred = volume_pred[:, 0]
        if volume_true.ndim == 4 and volume_true.shape[1] == 1:
            volume_true = volume_true[:, 0]
        if np.any(volume_pred):
            try:
                volume_pred = largest_connected_component(volume_pred)
            except ValueError:
                volume_pred = np.zeros_like(volume_pred)
        volumes[patients[patient_idx]] = (volume_in, volume_pred[:, None, ...], volume_true[:, None, ...])
        index += num_patient_slices
    return volumes


def dsc_distribution(volumes):
    distribution = {}
    for patient_id, (_, prediction, target) in volumes.items():
        distribution[patient_id] = dsc(prediction, target, lcc=False)
    return distribution


def plot_dsc(dsc_dist):
    y_positions = np.arange(len(dsc_dist))
    items = sorted(dsc_dist.items(), key=lambda item: item[1])
    values = [value for _, value in items]
    labels = ["_".join(key.split("_")[1:-1]) for key, _ in items]
    fig = plt.figure(figsize=(12, 8))
    canvas = FigureCanvasAgg(fig)
    plt.barh(y_positions, values, align="center", color="skyblue")
    plt.yticks(y_positions, labels, fontsize=8)
    plt.xticks(np.arange(0.0, 1.1, 0.1))
    plt.xlim([0.0, 1.0])
    plt.gca().axvline(np.mean(values), color="tomato", linewidth=2, label="mean")
    plt.gca().axvline(np.median(values), color="forestgreen", linewidth=2, label="median")
    plt.xlabel("Dice coefficient")
    plt.legend()
    plt.gca().xaxis.grid(color="silver", alpha=0.5, linestyle="--", linewidth=1)
    plt.tight_layout()
    canvas.draw()
    plt.close()
    buffer, (width, height) = canvas.print_to_buffer()
    return np.frombuffer(buffer, np.uint8).reshape((height, width, 4))


def evenly_spaced_indices(num_items, max_items=4):
    if num_items <= max_items:
        return list(range(num_items))
    return np.linspace(0, num_items - 1, max_items, dtype=int).tolist()


def export_overlay_summary(output_path, patient_id, volume):
    x, y_pred, y_true = volume
    tiles = []
    for slice_index in evenly_spaced_indices(x.shape[0], max_items=4):
        image = gray2rgb(x[slice_index, 1])
        image = outline(image, y_pred[slice_index, 0], color=[255, 0, 0])
        image = outline(image, y_true[slice_index, 0], color=[0, 255, 0])
        tiles.append(image)
    summary = np.concatenate(tiles, axis=1)
    imsave(output_path, summary)


def main():
    args = parse_args()
    args.images = args.images.resolve()
    args.experiment_dir = args.experiment_dir.resolve()
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    output_dir = args.experiment_dir / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(args.experiment_dir / "weights" / "best.pt", map_location="cpu")
    device = resolve_device(args.device)
    model = UNet(**checkpoint["model_kwargs"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    train_args = checkpoint.get("args", {})
    image_size = int(train_args.get("image_size", 128))
    seed = int(train_args.get("seed", 42))
    validation_cases = int(train_args.get("validation_cases", args.validation_cases))
    args.validation_cases = validation_cases

    loader = data_loader(args, image_size=image_size, seed=seed)

    input_list = []
    pred_list = []
    true_list = []

    with torch.no_grad():
        for x, y_true in loader:
            x = x.to(device)
            y_true = y_true.to(device)
            y_pred = model(x)
            pred_list.extend([sample for sample in y_pred.detach().cpu().numpy()])
            true_list.extend([sample for sample in y_true.detach().cpu().numpy()])
            input_list.extend([sample for sample in x.detach().cpu().numpy()])

    volumes = postprocess_per_volume(
        input_list,
        pred_list,
        true_list,
        loader.dataset.patient_slice_index,
        loader.dataset.patients,
    )
    dsc_dist = dsc_distribution(volumes)
    mean_dsc = float(np.mean(list(dsc_dist.values())))
    median_dsc = float(np.median(list(dsc_dist.values())))

    distribution_plot = plot_dsc(dsc_dist)
    imsave(output_dir / "dsc_distribution.png", distribution_plot)

    ranking = sorted(dsc_dist.items(), key=lambda item: item[1])
    representative = [
        ("worst", ranking[0][0]),
        ("median", ranking[len(ranking) // 2][0]),
        ("best", ranking[-1][0]),
    ]
    for label, patient_id in representative:
        export_overlay_summary(output_dir / f"{label}_{patient_id}.png", patient_id, volumes[patient_id])

    if args.export_all_slices:
        all_slices_dir = output_dir / "all_slices"
        all_slices_dir.mkdir(exist_ok=True)
        for patient_id, (x, y_pred, y_true) in volumes.items():
            for slice_index in range(x.shape[0]):
                image = gray2rgb(x[slice_index, 1])
                image = outline(image, y_pred[slice_index, 0], color=[255, 0, 0])
                image = outline(image, y_true[slice_index, 0], color=[0, 255, 0])
                imsave(all_slices_dir / f"{patient_id}-{slice_index:02d}.png", image)

    with (output_dir / "per_volume_dsc.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["patient_id", "dsc"])
        for patient_id, score in ranking:
            writer.writerow([patient_id, f"{score:.6f}"])

    metrics = {
        "best_checkpoint_epoch": checkpoint.get("epoch"),
        "best_validation_mean_dsc": checkpoint.get("best_val_dsc"),
        "evaluation_mean_dsc": mean_dsc,
        "evaluation_median_dsc": median_dsc,
        "num_validation_patients": len(dsc_dist),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
