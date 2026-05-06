import os
import os.path as osp 
import random
import pickle as pkl
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from tqdm import tqdm

import numpy as np
import torch
from skimage.io import imread
from torch.utils.data import Dataset

from utils import crop_sample, pad_sample, resize_sample, normalize_volume


class BrainSegmentationDataset(Dataset):
    """Brain MRI dataset for FLAIR abnormality segmentation"""

    in_channels = 3
    out_channels = 1

    def __init__(
        self,
        images_dir,
        transform=None,
        image_size=256,
        subset="train",
        random_sampling=True,
        validation_cases=10,
        seed=42,
        cache_dir=".cache",
        preprocess_workers=0,
    ):
        assert subset in ["all", "train", "validation"]

        images_dir = Path(images_dir)
        all_patients = sorted([path.name for path in images_dir.iterdir() if path.is_dir()])

        # select cases to subset
        if subset != "all":
            random.seed(seed)
            validation_patients = random.sample(all_patients, k=validation_cases)
            if subset == "validation":
                self.patients = validation_patients
            else:
                self.patients = sorted(list(set(all_patients).difference(validation_patients)))
        else:
            self.patients = all_patients

        # preprocess or load from cache (to save time)
        cache_key = "{}_{}_size{}_seed{}_val{}.pkl".format(
            osp.basename(osp.normpath(images_dir)),
            subset,
            image_size,
            seed,
            validation_cases,
        )
        cache_file = osp.join(cache_dir, cache_key)
        os.makedirs(cache_dir, exist_ok=True)

        if osp.exists(cache_file):
            self.volumes = self._load(cache_file)
        else:
            volumes = {}
            masks = {}
            print("reading {} images...".format(subset))
            for patient_id in self.patients:
                patient_dir = images_dir / patient_id
                filenames = sorted(
                    [path.name for path in patient_dir.glob("*.tif")],
                    key=lambda x: int(x.split(".")[-2].split("_")[4]),
                )
                image_slices = []
                mask_slices = []
                for filename in filenames:
                    filepath = patient_dir / filename
                    if "mask" in filename:
                        mask_slices.append(imread(filepath, as_gray=True))
                    else:
                        image_slices.append(imread(filepath))
                volumes[patient_id] = np.array(image_slices[1:-1])
                masks[patient_id] = np.array(mask_slices[1:-1])
            self.volumes = self._preprocess(
                self.patients,
                volumes,
                masks,
                subset,
                image_size,
                preprocess_workers=preprocess_workers,
            )
            self._save(self.volumes, cache_file)

        # probabilities for sampling slices based on masks
        self.slice_weights = [m.sum(axis=-1).sum(axis=-1) for v, m in self.volumes]
        self.slice_weights = [
            (s + (s.sum() * 0.1 / len(s))) / (s.sum() * 1.1) for s in self.slice_weights
        ]

        # add channel dimension to masks
        self.volumes = [(v, m[..., np.newaxis]) for (v, m) in self.volumes]

        print("done creating {} dataset".format(subset))

        # create global index for patient and slice (idx -> (p_idx, s_idx))
        num_slices = [v.shape[0] for v, m in self.volumes]
        self.patient_slice_index = list(
            zip(
                sum([[i] * num_slices[i] for i in range(len(num_slices))], []),
                sum([list(range(x)) for x in num_slices], []),
            )
        )

        self.random_sampling = random_sampling
        self.transform = transform


    def _load(self, cache_file):
        try:
            print("Load dataset from cache: {}".format(cache_file))
            with open(cache_file, "rb") as f:
                volumes = pkl.load(f)
            return volumes
        except Exception as e:
            raise Exception("Failed to load cached dataset, please delete {} \n{}".format(cache_file, str(e)))


    def _save(self, volumes, cache_file):
        with open(cache_file, "wb") as f:
             pkl.dump(volumes, f)


    def _preprocess(self, patients, volumes, masks, subset, image_size, preprocess_workers=0):
        print("preprocessing {} volumes...".format(subset))
        # create list of tuples (volume, mask)
        self.volumes = [(volumes[k], masks[k]) for k in patients]

        print("cropping {} volumes...".format(subset))
        # crop to smallest enclosing volume
        self.volumes = [crop_sample(v) for v in self.volumes]

        print("padding {} volumes...".format(subset))
        # pad to square
        self.volumes = [pad_sample(v) for v in self.volumes]

        print("resizing {} volumes...".format(subset))
        # resize
        if preprocess_workers and preprocess_workers > 1:
            resize_fn = partial(resize_sample, size=image_size)
            with ThreadPoolExecutor(max_workers=preprocess_workers) as executor:
                self.volumes = list(tqdm(executor.map(resize_fn, self.volumes), total=len(self.volumes)))
        else:
            self.volumes = [resize_sample(v, size=image_size) for v in tqdm(self.volumes)]

        print("normalizing {} volumes...".format(subset))
        # normalize channel-wise
        self.volumes = [(normalize_volume(v), m) for v, m in self.volumes]

        return self.volumes


    def __len__(self):
        return len(self.patient_slice_index)

    def __getitem__(self, idx):
        patient = self.patient_slice_index[idx][0]
        slice_n = self.patient_slice_index[idx][1]

        if self.random_sampling:
            patient = np.random.randint(len(self.volumes))
            slice_n = np.random.choice(
                range(self.volumes[patient][0].shape[0]), p=self.slice_weights[patient]
            )

        v, m = self.volumes[patient]
        image = v[slice_n]
        mask = m[slice_n]

        if self.transform is not None:
            image, mask = self.transform((image, mask))

        # fix dimensions (C, H, W)
        image = image.transpose(2, 0, 1)
        mask = mask.transpose(2, 0, 1)

        image_tensor = torch.from_numpy(image.astype(np.float32))
        mask_tensor = torch.from_numpy(mask.astype(np.float32))

        # normalize label to {0, 1}
        mask_tensor = mask_tensor / 255.

        # return tensors
        return image_tensor, mask_tensor
