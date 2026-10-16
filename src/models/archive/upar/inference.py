"""Predictor dùng trong môi trường chấm bài: không mạng, không gói `clip`, chỉ torch + numpy + Pillow.
Nạp: weights/clip_visual_fp16.pt (ViT-B/16 vision tower) + weights/head.pt (adapter, τ, text vectors đã tính sẵn, hiệu chuẩn)."""
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .attributes import ATTR_NAMES, NUM_ATTR
from .head import PartHead, apply_calibration, sigmoid_np
from .vision import CLIP_MEAN, CLIP_STD, PatchEncoder

_MEAN = np.asarray(CLIP_MEAN, dtype=np.float32).reshape(3, 1, 1)
_STD = np.asarray(CLIP_STD, dtype=np.float32).reshape(3, 1, 1)


def preprocess(img, img_h, img_w):
    """Giống hệt eval transform lúc huấn luyện: Resize(bicubic, PIL) -> /255 -> Normalize(CLIP). Trả về CHW float32."""
    img = img.convert("RGB").resize((img_w, img_h), Image.BICUBIC)
    arr = np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
    return (arr - _MEAN) / _STD


class Predictor:
    def __init__(self, weights_dir, device=None, tta=None, fp16=None, num_threads=None):
        wd = Path(weights_dir)
        head_ck = torch.load(str(wd / "head.pt"), map_location="cpu", weights_only=True)
        vis_sd = torch.load(str(wd / "clip_visual_fp16.pt"), map_location="cpu", weights_only=True)
        meta = head_ck["meta"]
        self.meta = meta
        self.img_h, self.img_w = int(meta["img_h"]), int(meta["img_w"])
        grid = (self.img_h // 16, self.img_w // 16)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        env_tta, env_fp16 = os.environ.get("UPAR_TTA"), os.environ.get("UPAR_FP16")
        self.tta = bool(int(env_tta)) if env_tta is not None else (bool(meta.get("tta", True)) if tta is None else bool(tta))
        want_fp16 = bool(int(env_fp16)) if env_fp16 is not None else (True if fp16 is None else bool(fp16))
        self.use_amp = bool(self.device.type == "cuda" and want_fp16)
        self.encoder = PatchEncoder.from_clip_visual_state(vis_sd, grid).to(self.device).eval()
        dim = int(head_ck["text_feats"].shape[-1])
        self.head = PartHead(dim, grid, tuple(meta["rows"]), meta["part_of_attr"], int(meta["adapter_reduction"]))
        self.head.load_state_dict(head_ck["head"], strict=True)
        self.head = self.head.to(self.device).eval()
        self.tfeat = head_ck["text_feats"].float().to(self.device)
        cal = head_ck["calib"]
        self.cal_a, self.cal_b, self.cal_off = (cal[k].double().numpy() for k in ("a", "b", "off"))
        self.pool = ThreadPoolExecutor(max_workers=int(num_threads or min(8, os.cpu_count() or 2)))
        self.n_load_errors = 0
        print("[upar] sẵn sàng | device=%s | fp16=%s | TTA=%s | ảnh %dx%d (lưới %dx%d) | val_hm=%s"
              % (self.device, self.use_amp, self.tta, self.img_h, self.img_w, grid[0], grid[1], meta.get("val_hm")),
              file=sys.stderr, flush=True)

    def _load(self, path):
        try:
            with Image.open(path) as im:
                return preprocess(im, self.img_h, self.img_w)
        except Exception as e:                         # ảnh hỏng: dùng ảnh xám để không làm sập cả lượt chạy
            self.n_load_errors += 1
            print("[upar] CẢNH BÁO không đọc được %s: %r" % (path, e), file=sys.stderr, flush=True)
            return np.zeros((3, self.img_h, self.img_w), dtype=np.float32)

    @torch.inference_mode()
    def predict_logits(self, paths):
        x = torch.from_numpy(np.stack(list(self.pool.map(self._load, paths)))).to(self.device)
        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp):
            parts, glob = self.head.parts_from_tokens(self.encoder(x))
            z = self.head.logits(parts, glob, self.tfeat)
            if self.tta:
                p2, g2 = self.head.parts_from_tokens(self.encoder(torch.flip(x, dims=[3])))
                z = 0.5 * (z + self.head.logits(p2, g2, self.tfeat))
        return z.float().cpu().numpy()

    def predict_paths(self, paths):
        """Xác suất đã hiệu chuẩn [N, 40], thứ tự = ATTR_NAMES; luôn hữu hạn và nằm trong [0, 1]."""
        z = self.predict_logits(paths)
        p = sigmoid_np(apply_calibration(z, self.cal_a, self.cal_b, self.cal_off))
        return np.clip(np.nan_to_num(p, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
