"""Dữ liệu huấn luyện: đọc gt.csv, tìm ảnh, suy ra nguồn dataset (cho GRL), augmentation, Dataset."""
import glob
import os
import random
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

from ..attributes import NUM_ATTR
from ..vision import CLIP_MEAN, CLIP_STD

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DOMAIN_NAMES = ["Market1501", "PA100k", "PETA"]      # nguồn dataset dùng cho nhánh GRL
N_DOM = len(DOMAIN_NAMES)


def find_data_root(user_value=""):
    cands = [Path(v) for v in (user_value, os.environ.get("UPAR_DATA_ROOT", "")) if v]
    cands += [Path("data"), Path("../data"), Path("/content/data"), Path("/kaggle/working/data")]
    for pat in ("/kaggle/input/*", "/kaggle/input/*/data", "/kaggle/input/*/*", "/content/*", "/content/drive/MyDrive/*"):
        cands += [Path(p) for p in glob.glob(pat)]
    for c in cands:
        if (c / "annotations" / "task1" / "train" / "gt.csv").is_file():
            return c.resolve()
    raise FileNotFoundError("Không thấy data/annotations/task1/train/gt.csv. Hãy đặt cfg.data_root = '<đường dẫn thư mục data>'.")


def read_gt_csv(path, n_attr=NUM_ATTR):
    """Đọc gt.csv: 'image,<40 nhãn>'. Bỏ dòng comment '#' và dòng header chữ.
    Nhãn 0/1 giữ nguyên; mọi giá trị khác (-1, rỗng, NaN...) = chưa gán nhãn -> -1 (bị mask khỏi loss/metric)."""
    names, rows = [], []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = [p.strip().strip('"').strip("'") for p in s.split(",")]
            try:
                vals = [float(v) if v != "" else float("nan") for v in parts[1:]]
            except ValueError:
                continue                                  # header dạng chữ: image,Age-Young,...
            if len(vals) < n_attr:
                raise ValueError("%s: dòng có %d nhãn (<%d): %s" % (path, len(vals), n_attr, s[:120]))
            names.append(parts[0])
            rows.append(vals[:n_attr])
    y = np.asarray(rows, dtype=np.float32)
    y[~np.isin(y, (0.0, 1.0))] = -1.0
    return names, y.astype(np.int8)


class PathResolver:
    """Tìm file ảnh: ghép với từng root; nếu không thấy thì dò theo tên file (basename)."""

    def __init__(self, roots):
        self.roots = []
        for r in roots:
            if r and Path(r).is_dir() and Path(r).resolve() not in self.roots:
                self.roots.append(Path(r).resolve())
        self._index = None

    def _build_index(self):
        print("[data] Đang quét thư mục để lập chỉ mục ảnh theo tên file (chỉ chạy 1 lần)...")
        idx = {}
        for r in self.roots:
            for dp, _, fns in os.walk(r):
                for fn in fns:
                    if os.path.splitext(fn)[1].lower() in IMG_EXT:
                        idx.setdefault(fn, os.path.join(dp, fn))
        self._index = idx

    def __call__(self, rel):
        rel = rel.replace("\\", "/")
        for r in self.roots:
            p = r / rel
            if p.is_file():
                return str(p)
        if self._index is None:
            self._build_index()
        return self._index.get(os.path.basename(rel))


def infer_domain(*strings):
    """0=Market1501, 1=PA100k, 2=PETA, -1=không xác định (dựa vào tên thư mục trong đường dẫn)."""
    for s in strings:
        if not s:
            continue
        for p in s.replace("\\", "/").lower().split("/"):
            if p.startswith("market"):
                return 0
            if p in ("pa100k", "pa-100k", "pa_100k"):
                return 1
            if p.startswith("peta"):
                return 2
    return -1


class RandomDownUp:
    """Giả lập ảnh độ phân giải thấp (camera xa / drone): thu nhỏ rồi phóng lại."""

    def __init__(self, p=0.4, smin=0.35):
        self.p, self.smin = p, smin

    def __call__(self, img):
        if random.random() > self.p:
            return img
        w, h = img.size
        s = random.uniform(self.smin, 1.0)
        img = img.resize((max(8, int(w * s)), max(8, int(h * s))), Image.BILINEAR)
        return img.resize((w, h), Image.BICUBIC)


def make_transforms(img_h, img_w):
    """Augmentation KHÔNG đổi hue (giữ đúng nhãn màu). eval_tf khớp từng bước với upar.inference.preprocess."""
    size, bic = (img_h, img_w), T.InterpolationMode.BICUBIC
    train_tf = T.Compose([
        RandomDownUp(0.4, 0.35),
        T.Resize(size, interpolation=bic),
        T.RandomHorizontalFlip(0.5),
        T.RandomApply([T.ColorJitter(0.3, 0.3, 0.1, 0.0)], p=0.5),
        T.RandomApply([T.RandomAffine(0, translate=(0.04, 0.04), scale=(0.92, 1.08))], p=0.5),
        T.ToTensor(), T.Normalize(CLIP_MEAN, CLIP_STD),
        T.RandomErasing(p=0.25, scale=(0.02, 0.12), ratio=(0.3, 3.3), value=0),      # mô phỏng che khuất
    ])
    eval_tf = T.Compose([T.Resize(size, interpolation=bic), T.ToTensor(), T.Normalize(CLIP_MEAN, CLIP_STD)])
    return train_tf, eval_tf


class PARDataset(Dataset):
    def __init__(self, paths, labels=None, domains=None, transform=None):
        self.paths, self.labels, self.domains, self.tf = paths, labels, domains, transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        try:
            img, ok = Image.open(self.paths[i]).convert("RGB"), True
        except Exception:
            img, ok = Image.new("RGB", (64, 128)), False           # ảnh hỏng: nhãn = -1 (bị mask)
        x = self.tf(img)
        y = torch.full((NUM_ATTR,), -1.0) if (self.labels is None or not ok) else torch.from_numpy(self.labels[i].astype(np.float32))
        d = int(self.domains[i]) if self.domains is not None else -1
        return x, y, d
