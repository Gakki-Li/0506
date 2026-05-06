import numpy as np
import torch
from skimage.io import imsave

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None


def make_image_grid(images, columns=4):
    if len(images) == 0:
        return None
    images = np.array(images)
    if images.ndim != 4:
        raise ValueError("Expected images with shape (N, H, W, C)")
    rows = (len(images) + columns - 1) // columns
    h, w, c = images.shape[1:]
    grid = np.zeros((rows * h, columns * w, c), dtype=images.dtype)
    for idx, image in enumerate(images):
        row = idx // columns
        col = idx % columns
        grid[row * h : (row + 1) * h, col * w : (col + 1) * w] = image
    return grid


class Logger(object):

    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir) if SummaryWriter is not None else None

    def scalar_summary(self, tag, value, step):
        if self.writer is not None:
            self.writer.add_scalar(tag, value, step)
            self.writer.flush()

    def image_summary(self, tag, image, step):
        if self.writer is not None:
            self.writer.add_image(tag, image, step)
            self.writer.flush()
            return
        safe_tag = tag.replace("/", "_")
        imsave(f"{self.log_dir}/{safe_tag}_{step:06d}.png", image)

    def image_list_summary(self, tag, images, step):
        if len(images) == 0:
            return
        if self.writer is not None:
            images = np.array(images)
            images = torch.tensor(images).permute(0, 3, 1, 2)
            nrow = min(4, len(images))
            rows = []
            for start in range(0, len(images), nrow):
                chunk = images[start : start + nrow]
                row = torch.cat(list(chunk), dim=2)
                rows.append(row)
            grid = torch.cat(rows, dim=1)
            self.writer.add_image(tag, grid, step)
            self.writer.flush()
            return
        self.image_summary(tag, make_image_grid(images), step)
