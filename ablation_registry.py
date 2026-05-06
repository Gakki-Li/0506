EXPERIMENTS = [
    {
        "name": "baseline",
        "label": "Baseline",
        "description": "Standard U-Net with Dice loss and Adam.",
        "extra_args": [],
    },
    {
        "name": "bn",
        "label": "Baseline + BatchNorm",
        "description": "Enable batch normalization in convolution blocks.",
        "extra_args": ["--batch-norm"],
    },
    {
        "name": "bilinear",
        "label": "Baseline + Bilinear",
        "description": "Use bilinear upsampling in the decoder.",
        "extra_args": ["--bilinear"],
    },
    {
        "name": "dropout",
        "label": "Baseline + Dropout",
        "description": "Add dropout to deeper encoder/decoder blocks.",
        "extra_args": ["--dropout", "0.1"],
    },
    {
        "name": "bce_dice",
        "label": "Baseline + BCE+Dice",
        "description": "Replace Dice loss with combined BCE and Dice loss.",
        "extra_args": ["--loss", "bce_dice", "--bce-weight", "0.4"],
    },
    {
        "name": "adamw",
        "label": "Baseline + AdamW",
        "description": "Replace Adam with AdamW and add mild weight decay.",
        "extra_args": ["--optimizer", "adamw", "--weight-decay", "1e-4"],
    },
    {
        "name": "cosine",
        "label": "Baseline + Cosine Scheduler",
        "description": "Add cosine annealing learning-rate scheduling.",
        "extra_args": ["--scheduler", "cosine"],
    },
    {
        "name": "strong_aug",
        "label": "Baseline + Stronger Aug",
        "description": "Increase scale and rotation augmentation strength.",
        "extra_args": ["--aug-scale", "0.10", "--aug-angle", "20.0"],
    },
    {
        "name": "improved",
        "label": "All Improvements",
        "description": "Combine batch norm, bilinear upsampling, dropout, BCE+Dice, AdamW, cosine scheduling, and stronger augmentation.",
        "extra_args": [
            "--optimizer",
            "adamw",
            "--loss",
            "bce_dice",
            "--bce-weight",
            "0.4",
            "--lr",
            "1e-3",
            "--weight-decay",
            "1e-4",
            "--scheduler",
            "cosine",
            "--batch-norm",
            "--bilinear",
            "--dropout",
            "0.1",
            "--aug-scale",
            "0.10",
            "--aug-angle",
            "20.0",
        ],
    },
]


EXPERIMENT_INDEX = {item["name"]: item for item in EXPERIMENTS}


def list_experiment_names():
    return [item["name"] for item in EXPERIMENTS]


def resolve_experiments(selected_names=None):
    if not selected_names:
        return EXPERIMENTS

    resolved = []
    for name in selected_names:
        if name not in EXPERIMENT_INDEX:
            raise KeyError("Unknown experiment: {}".format(name))
        resolved.append(EXPERIMENT_INDEX[name])
    return resolved
