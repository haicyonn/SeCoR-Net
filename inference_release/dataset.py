from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


METHOD_A_MEAN = np.array([0.406, 0.456, 0.485, 0.406, 0.456, 0.485], dtype=np.float32)
METHOD_A_STD = np.array([0.225, 0.224, 0.229, 0.225, 0.224, 0.229], dtype=np.float32)
METHOD_B_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
METHOD_B_STD = np.array([0.5, 0.5, 0.5], dtype=np.float32)


def _first_existing(root, names):
    for name in names:
        path = root / name
        if path.is_dir():
            return path
    return None


def _normalize_pipeline_name(pipeline):
    name = str(pipeline).strip().lower()
    if name not in {"methoda", "methodb"}:
        raise ValueError("Unsupported data pipeline: {}. Use methodA or methodB.".format(pipeline))
    return name


def _apply_method_a(pre, post, size, preserve_ab_order):
    image = np.concatenate([pre, post], axis=2).astype(np.float32)
    image = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
    image = image / 255.0
    image = (image - METHOD_A_MEAN) / METHOD_A_STD
    if preserve_ab_order:
        return np.concatenate([image[:, :, 2::-1], image[:, :, 5:2:-1]], axis=2)
    return image[:, :, ::-1].copy()


def _apply_method_b(pre, post, size):
    pre = cv2.resize(pre, (size, size), interpolation=cv2.INTER_LINEAR)
    post = cv2.resize(post, (size, size), interpolation=cv2.INTER_LINEAR)
    pre = pre[:, :, ::-1].astype(np.float32) / 255.0
    post = post[:, :, ::-1].astype(np.float32) / 255.0
    pre = (pre - METHOD_B_MEAN) / METHOD_B_STD
    post = (post - METHOD_B_MEAN) / METHOD_B_STD
    return np.concatenate([pre, post], axis=2).copy()


def apply_preprocess(method, pre, post, size, preserve_ab_order):
    method = _normalize_pipeline_name(method)
    if method == "methodb":
        return _apply_method_b(pre, post, size)
    return _apply_method_a(pre, post, size, preserve_ab_order)


class ChangeDataset(Dataset):
    def __init__(self, root, split="test", size=256, preserve_ab_order=False, pipeline="methodA"):
        root = Path(root)
        split_root = root / split
        if split_root.is_dir():
            root = split_root

        self.pre_dir = _first_existing(root, ["A", "time1", "Time1", "T1", "t1"])
        self.post_dir = _first_existing(root, ["B", "time2", "Time2", "T2", "t2"])
        self.label_dir = _first_existing(root, ["label", "labels", "gt", "GT", "mask", "masks"])
        if self.pre_dir is None or self.post_dir is None or self.label_dir is None:
            raise FileNotFoundError("Missing time1/time2 or A/B and label folders under: {}".format(root))

        image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        self.files = sorted(
            [
                p.name
                for p in self.pre_dir.iterdir()
                if p.is_file() and p.suffix.lower() in image_exts and ":zone.identifier" not in p.name.lower()
            ]
        )
        if not self.files:
            raise FileNotFoundError("No test images found under: {}".format(self.pre_dir))

        self.size = int(size)
        self.preserve_ab_order = bool(preserve_ab_order)
        self.pipeline = _normalize_pipeline_name(pipeline)

    def __len__(self):
        return len(self.files)

    def _read_label(self, name):
        path = self.label_dir / name
        label = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if label is None:
            raise FileNotFoundError("Could not read label: {}".format(path))
        label = cv2.resize(label, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        label = np.ceil(label.astype(np.float32) / 255.0).astype(np.int64)
        return torch.from_numpy(label)

    def __getitem__(self, index):
        name = self.files[index]
        pre_path = self.pre_dir / name
        post_path = self.post_dir / name
        pre = cv2.imread(str(pre_path), cv2.IMREAD_COLOR)
        post = cv2.imread(str(post_path), cv2.IMREAD_COLOR)
        if pre is None:
            raise FileNotFoundError("Could not read image: {}".format(pre_path))
        if post is None:
            raise FileNotFoundError("Could not read image: {}".format(post_path))

        image = apply_preprocess(self.pipeline, pre, post, self.size, self.preserve_ab_order)
        image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        label = self._read_label(name)
        return image, label, name
