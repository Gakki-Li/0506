import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import BrainSegmentationDataset as Dataset
from logger import Logger
from loss import BCEDiceLoss, DiceLoss
from transform import transforms
from unet import UNet
from utils import dsc, log_images, seed_everything


def default_images_dir():
    return Path(__file__).resolve().parents[2] / "BrainMRI" / "kaggle_3m"


def parse_args():
    parser = argparse.ArgumentParser(description="Train U-Net for brain MRI segmentation.")
    parser.add_argument("--images", type=Path, default=default_images_dir())
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parent / "outputs")
    parser.add_argument("--experiment-name", type=str, required=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--steps-per-epoch", type=int, default=0)
    parser.add_argument("--validate-every", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--validation-cases", type=int, default=10)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--cache-dir", type=Path, default=Path(__file__).resolve().parent / ".cache")
    parser.add_argument("--preprocess-workers", type=int, default=4)
    parser.add_argument("--aug-scale", type=float, default=0.05)
    parser.add_argument("--aug-angle", type=float, default=15.0)
    parser.add_argument("--flip-prob", type=float, default=0.5)
    parser.add_argument("--vis-images", type=int, default=200)
    parser.add_argument("--vis-freq", type=int, default=10)
    parser.add_argument("--optimizer", choices=["adam", "adamw"], default="adam")
    parser.add_argument("--scheduler", choices=["none", "plateau", "cosine"], default="none")
    parser.add_argument("--loss", choices=["dice", "bce_dice"], default="dice")
    parser.add_argument("--bce-weight", type=float, default=0.5)
    parser.add_argument("--init-features", type=int, default=32)
    parser.add_argument("--bilinear", action="store_true")
    parser.add_argument("--batch-norm", action="store_true")
    parser.add_argument("--dropout", type=float, default=0.0)
    return parser.parse_args()


def resolve_device(device_arg):
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def worker_init(worker_id):
    np.random.seed(42 + worker_id)


def build_dataloaders(args):
    train_dataset = Dataset(
        images_dir=str(args.images),
        subset="train",
        image_size=args.image_size,
        transform=transforms(scale=args.aug_scale, angle=args.aug_angle, flip_prob=args.flip_prob),
        validation_cases=args.validation_cases,
        cache_dir=str(args.cache_dir),
        seed=args.seed,
        preprocess_workers=args.preprocess_workers,
    )
    valid_dataset = Dataset(
        images_dir=str(args.images),
        subset="validation",
        image_size=args.image_size,
        random_sampling=False,
        validation_cases=args.validation_cases,
        cache_dir=str(args.cache_dir),
        seed=args.seed,
        preprocess_workers=args.preprocess_workers,
    )
    loader_train = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
        worker_init_fn=worker_init,
    )
    loader_valid = DataLoader(
        valid_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.workers,
        worker_init_fn=worker_init,
    )
    return loader_train, loader_valid


def build_model(args):
    return UNet(
        in_channels=Dataset.in_channels,
        out_channels=Dataset.out_channels,
        init_features=args.init_features,
        bilinear=args.bilinear,
        use_batchnorm=args.batch_norm,
        dropout=args.dropout,
    )


def build_optimizer(args, model):
    if args.optimizer == "adam":
        return optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    return optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


def build_scheduler(args, optimizer):
    if args.scheduler == "plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2, verbose=True)
    if args.scheduler == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    return None


def build_loss(args):
    if args.loss == "dice":
        return DiceLoss()
    return BCEDiceLoss(bce_weight=args.bce_weight)


def dsc_per_volume(validation_pred, validation_true, patient_slice_index):
    dsc_list = []
    num_slices = np.bincount([p[0] for p in patient_slice_index])
    index = 0
    for patient_index, num_patient_slices in enumerate(num_slices):
        y_pred = np.array(validation_pred[index : index + num_patient_slices])
        y_true = np.array(validation_true[index : index + num_patient_slices])
        dsc_list.append(dsc(y_pred, y_true))
        index += num_patient_slices
    return dsc_list


def save_history_row(history_path, row):
    new_file = not history_path.exists()
    with history_path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def save_checkpoint(path, model, args, epoch, best_val_dsc):
    checkpoint = {
        "epoch": epoch,
        "best_val_dsc": best_val_dsc,
        "model_state_dict": model.state_dict(),
        "model_kwargs": {
            "in_channels": Dataset.in_channels,
            "out_channels": Dataset.out_channels,
            "init_features": args.init_features,
            "bilinear": args.bilinear,
            "use_batchnorm": args.batch_norm,
            "dropout": args.dropout,
        },
        "args": vars(args),
    }
    torch.save(checkpoint, path)


def save_visualizations(logger, x, y_true, y_pred, step, limit, tag):
    images = log_images(x, y_true, y_pred)[:limit]
    logger.image_list_summary(tag, images, step)


def run_epoch(model, loader, criterion, optimizer, device, is_train, steps_per_epoch=0):
    losses = []
    predictions = []
    targets = []
    batches = 0
    model.train(is_train)
    progress = tqdm(loader, leave=False)
    for batch_index, (x, y_true) in enumerate(progress):
        if steps_per_epoch and batch_index >= steps_per_epoch:
            break

        x = x.to(device)
        y_true = y_true.to(device)
        optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            y_pred = model(x)
            loss = criterion(y_pred, y_true)
            if is_train:
                loss.backward()
                optimizer.step()

        losses.append(loss.item())
        batches += 1
        progress.set_description("train" if is_train else "valid")

        if not is_train:
            predictions.extend([sample for sample in y_pred.detach().cpu().numpy()])
            targets.extend([sample for sample in y_true.detach().cpu().numpy()])

    mean_loss = float(np.mean(losses)) if losses else 0.0
    return {
        "loss": mean_loss,
        "predictions": predictions,
        "targets": targets,
        "batches": batches,
    }


def main():
    args = parse_args()
    args.images = args.images.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    experiment_dir = (args.output_root / args.experiment_name).resolve()
    weights_dir = experiment_dir / "weights"
    logs_dir = experiment_dir / "logs"
    weights_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(args.seed)
    device = resolve_device(args.device)
    logger = Logger(str(logs_dir))

    config_path = experiment_dir / "config.json"
    config_path.write_text(json.dumps(vars(args), indent=2, default=str))

    loader_train, loader_valid = build_dataloaders(args)
    model = build_model(args).to(device)
    optimizer = build_optimizer(args, model)
    scheduler = build_scheduler(args, optimizer)
    criterion = build_loss(args)

    history_path = experiment_dir / "history.csv"
    metrics_path = experiment_dir / "metrics.json"
    best_val_dsc = -1.0
    best_epoch = -1
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        train_result = run_epoch(
            model=model,
            loader=loader_train,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            is_train=True,
            steps_per_epoch=args.steps_per_epoch,
        )
        global_step += train_result["batches"]

        should_validate = (epoch % args.validate_every == 0) or (epoch == args.epochs)
        valid_loss = None
        val_dsc = None

        if should_validate:
            with torch.no_grad():
                valid_result = run_epoch(
                    model=model,
                    loader=loader_valid,
                    criterion=criterion,
                    optimizer=optimizer,
                    device=device,
                    is_train=False,
                )
            valid_loss = valid_result["loss"]
            val_dsc = float(
                np.mean(
                    dsc_per_volume(
                        valid_result["predictions"],
                        valid_result["targets"],
                        loader_valid.dataset.patient_slice_index,
                    )
                )
            )

        current_lr = optimizer.param_groups[0]["lr"]
        logger.scalar_summary("loss/train", train_result["loss"], global_step)
        if valid_loss is not None:
            logger.scalar_summary("loss/valid", valid_loss, global_step)
            logger.scalar_summary("dsc/valid", val_dsc, global_step)

        if should_validate and epoch % args.vis_freq == 0:
            batch = next(iter(loader_valid))
            x_vis, y_vis = [tensor.to(device) for tensor in batch]
            with torch.no_grad():
                y_pred_vis = model(x_vis)
            save_visualizations(logger, x_vis, y_vis, y_pred_vis, global_step, args.vis_images, f"valid_epoch_{epoch}")

        row = {
            "epoch": epoch,
            "train_loss": round(train_result["loss"], 6),
            "valid_loss": None if valid_loss is None else round(valid_loss, 6),
            "valid_dsc": None if val_dsc is None else round(val_dsc, 6),
            "lr": current_lr,
        }
        print(json.dumps(row))
        save_history_row(history_path, row)

        save_checkpoint(weights_dir / "last.pt", model, args, epoch, best_val_dsc)
        if val_dsc is not None and val_dsc > best_val_dsc:
            best_val_dsc = val_dsc
            best_epoch = epoch
            save_checkpoint(weights_dir / "best.pt", model, args, epoch, best_val_dsc)

        if scheduler is not None:
            if args.scheduler == "plateau" and val_dsc is not None:
                scheduler.step(val_dsc)
            elif args.scheduler != "plateau":
                scheduler.step()

    if best_epoch == -1:
        best_epoch = args.epochs
        best_val_dsc = float("nan")
        save_checkpoint(weights_dir / "best.pt", model, args, args.epochs, best_val_dsc)

    summary = {
        "experiment_name": args.experiment_name,
        "best_epoch": best_epoch,
        "best_validation_mean_dsc": best_val_dsc,
        "device": str(device),
        "images": str(args.images),
        "weights": str(weights_dir / "best.pt"),
    }
    metrics_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
