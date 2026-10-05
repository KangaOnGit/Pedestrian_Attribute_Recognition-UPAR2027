"""Đóng gói bài nộp theo đúng quy định của nền tảng:

    my_submission/            <- nén NỘI DUNG thư mục này (không nén chính thư mục)
    ├── run.py                <- entry point (ở GỐC file zip)
    ├── metadata.yaml         <- sao chép NGUYÊN VẸN từ example submission của ban tổ chức
    ├── upar/                 <- code suy luận (chỉ torch + numpy + Pillow)
    └── weights/
        ├── clip_visual_fp16.pt   ViT-B/16 vision tower (fp16)
        ├── head.pt               adapter, τ, 80 text vector tính sẵn (chỉ tensor)
        └── runtime.json          metadata, hiệu chuẩn, prior (tỉ lệ dương trên tập train; dùng khi phải fallback)
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np

EXCLUDE_DIRS = {"__pycache__", ".git"}
EXCLUDE_FILES = {".DS_Store"}


def _clean_meta(d):
    """Chỉ giữ kiểu Python thuần để torch.load(weights_only=True) đọc được trên torch 2.4.1."""
    out = {}
    for k, v in d.items():
        if isinstance(v, (tuple, list)):
            out[k] = [x.item() if hasattr(x, "item") else x for x in v]
        elif hasattr(v, "item"):
            out[k] = v.item()
        else:
            out[k] = v
    return out


def export_submission(model, tfeat, calib, cfg, sub_dir, project_root, metadata_yaml="", val_hm=None, prior=None):
    """Ghi sub_dir/{run.py, upar/, weights/, metadata.yaml?}. `tfeat`: [40,2,512] = text vector đã tính sẵn từ soft prompt.
    `prior`: 40 tỉ lệ dương trên tập train, dùng làm dự đoán dự phòng khi run.py không kịp hạn / gặp lỗi."""
    import torch
    sub_dir, project_root = Path(sub_dir), Path(project_root)
    if sub_dir.exists() and any(sub_dir.iterdir()) and not (sub_dir / "run.py").exists():
        child = sub_dir / "my_submission"                      # trỏ nhầm vào thư mục đang chứa file khác (vd. /kaggle/working)
        print("[export] %s đang chứa file khác -> dùng thư mục con %s" % (sub_dir, child))
        sub_dir = child
        if sub_dir.exists() and any(sub_dir.iterdir()) and not (sub_dir / "run.py").exists():
            raise RuntimeError("%s đã tồn tại và không phải thư mục bài nộp cũ; chọn cfg.submission_dir khác." % sub_dir)
    if sub_dir.exists():
        shutil.rmtree(sub_dir)
    (sub_dir / "weights").mkdir(parents=True)

    shutil.copy(project_root / "run.py", sub_dir / "run.py")
    shutil.copytree(project_root / "upar", sub_dir / "upar", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

    vis = {k: v.detach().cpu().half() for k, v in model.visual.state_dict().items()}          # CLIP gốc vốn là fp16 -> không mất gì
    torch.save(vis, sub_dir / "weights" / "clip_visual_fp16.pt")
    from upar.attributes import ATTR_NAMES, PART_OF_ATTR
    meta = _clean_meta(dict(img_h=cfg.img_h, img_w=cfg.img_w, rows=list(model.rows), part_of_attr=list(PART_OF_ATTR),
                            adapter_reduction=cfg.adapter_reduction, tta=bool(cfg.tta_flip), attr_names=list(ATTR_NAMES),
                            val_hm=None if val_hm is None else float(val_hm), version="1.0"))
    head = {"head": {k: v.detach().cpu().float() for k, v in model.head.state_dict().items()},
            "text_feats": tfeat.detach().cpu().float()}                                # chỉ tensor: torch.load(weights_only=True) an toàn
    torch.save(head, sub_dir / "weights" / "head.pt")
    prior_list = [0.0] * 40 if prior is None else [float(v) for v in np.asarray(prior).reshape(-1)]
    assert len(prior_list) == 40
    runtime = {"version": "2.0", "meta": meta, "prior": prior_list,
               "calib": {k: [float(v) for v in np.asarray(calib[k]).reshape(-1)] for k in ("a", "b", "off")}}
    (sub_dir / "weights" / "runtime.json").write_text(json.dumps(runtime))

    md = Path(metadata_yaml) if metadata_yaml else None
    for cand in ([md] if md else []) + [project_root / "metadata.yaml", Path("metadata.yaml")]:
        if cand and cand.is_file():
            shutil.copy(cand, sub_dir / "metadata.yaml")                                   # copy nguyên vẹn, không sửa
            break
    size = sum(f.stat().st_size for f in sub_dir.rglob("*") if f.is_file()) / 2 ** 20
    print("[export] đã ghi %s (%.0f MiB) | metadata.yaml: %s" % (sub_dir, size, "có" if (sub_dir / "metadata.yaml").exists()
                                                                  else "CHƯA CÓ — hãy copy từ example submission"))
    return sub_dir


def zip_submission(sub_dir, zip_path):
    """Nén NỘI DUNG sub_dir (run.py ở gốc zip), bỏ __pycache__ / .DS_Store / .git — tương đương
    `cd my_submission && zip -r ../my_submission.zip . -x "*/__pycache__/*" "*.DS_Store" ".git/*"`."""
    sub_dir, zip_path = Path(sub_dir), Path(zip_path)
    problems = [n for n in ("run.py", "metadata.yaml") if not (sub_dir / n).is_file()]
    if "run.py" in problems:
        raise FileNotFoundError("Thiếu run.py ở gốc %s" % sub_dir)
    if problems:
        print("[zip] CẢNH BÁO: thiếu %s — nền tảng yêu cầu file này ở gốc zip." % ", ".join(problems))
    if zip_path.exists():
        zip_path.unlink()
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(sub_dir):
            dirs[:] = sorted(d for d in dirs if d not in EXCLUDE_DIRS)
            for f in sorted(files):
                if f in EXCLUDE_FILES or f.endswith(".pyc"):
                    continue
                full = Path(root) / f
                zf.write(full, arcname=full.relative_to(sub_dir).as_posix())
                n += 1
    print("[zip] %s: %d file, %.0f MiB | run.py ở gốc: %s" % (zip_path, n, zip_path.stat().st_size / 2 ** 20,
                                                              "run.py" in zipfile.ZipFile(zip_path).namelist()))
    return zip_path


# ------------------------------------------------------------------------------------------------
# Mô phỏng ingestion program: chạy run.py trong TIẾN TRÌNH MỚI, cwd = thư mục bài nộp, chặn import
# các gói không có trong môi trường chấm (clip, ftfy, regex, torchvision, ...) để chắc chắn bài nộp tự chứa.
# ------------------------------------------------------------------------------------------------
_INGEST_SIM = r'''
import sys, os, json, time
import numpy as np
for m in ("clip", "ftfy", "regex", "torchvision", "open_clip", "timm", "huggingface_hub", "upar_train"):
    sys.modules[m] = None                        # import sẽ báo ImportError nếu bài nộp lỡ phụ thuộc vào chúng
paths = json.load(open(sys.argv[1])); out = sys.argv[2]
sys.path.insert(0, os.getcwd())
import run
from upar.attributes import ATTR_NAMES
t0 = time.time()
if hasattr(run, "load_model"):
    run.load_model()
t_load = time.time() - t0
bs = int(getattr(run, "BATCH_SIZE", 32))
rows, t1, checked = [], time.time(), False
for i in range(0, len(paths), bs):
    chunk = [{"index": i + j, "image": os.path.basename(p), "image_path": p, "attribute_names": list(ATTR_NAMES)}
             for j, p in enumerate(paths[i:i + bs])]
    res = run.predict_batch(chunk) if hasattr(run, "predict_batch") else [run.predict_image(s) for s in chunk]
    assert len(res) == len(chunk), "predict_batch phải trả về đúng %d dòng" % len(chunk)
    for r in res:
        if isinstance(r, dict):
            r = [r[n] for n in ATTR_NAMES]
        r = np.asarray(r, dtype=np.float64).reshape(-1)
        assert r.shape == (40,), "cần 40 xác suất, nhận %s" % (r.shape,)
        assert np.isfinite(r).all() and r.min() >= 0 and r.max() <= 1, "xác suất NaN/ngoài [0,1]"
        rows.append(r)
    if not checked and hasattr(run, "predict_image"):      # predict_image phải khớp predict_batch (chỉ so khi mức tính toán chưa đổi)
        checked = True
        lvl0 = getattr(getattr(run, "_governor", None), "level", None)
        s0 = {"index": 0, "image": os.path.basename(paths[0]), "image_path": paths[0], "attribute_names": list(ATTR_NAMES)}
        r0 = np.asarray(run.predict_image(s0), dtype=np.float64)
        if lvl0 == getattr(getattr(run, "_governor", None), "level", None):
            assert np.abs(r0 - rows[0]).max() < 5e-3, "predict_image lệch predict_batch: %.4g" % np.abs(r0 - rows[0]).max()
dt = time.time() - t1
np.save(out, np.asarray(rows, dtype=np.float32))
ms = 1000 * dt / max(1, len(paths))
print("[ingest-sim] %d ảnh | nạp mô hình %.1fs | suy luận %.1fs (%.1f ms/ảnh, batch=%d) | ước lượng 29248 ảnh: %.0f phút" % (len(paths), t_load, dt, ms, bs, ms * 29248 / 60000.0))
'''


def run_ingestion_sim(sub_dir, image_paths, timeout=None, extra_env=None, python=None):
    """Trả về mảng xác suất [N, 40] do run.py (trong thư mục bài nộp) sinh ra; ném lỗi nếu vi phạm quy định đầu ra."""
    sub_dir = Path(sub_dir).resolve()
    with tempfile.TemporaryDirectory() as td:
        lst, out, script = Path(td) / "paths.json", Path(td) / "probs.npy", Path(td) / "sim.py"
        lst.write_text(json.dumps([str(Path(p).resolve()) for p in image_paths]))
        script.write_text(_INGEST_SIM)
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)                                   # không để lọt code của project vào tiến trình mô phỏng
        env.update(extra_env or {})
        r = subprocess.run([python or sys.executable, str(script), str(lst), str(out)], cwd=str(sub_dir), env=env,
                           capture_output=True, text=True, timeout=timeout)
        print((r.stdout or "").strip())
        err_lines = [l for l in (r.stderr or "").strip().splitlines() if l.startswith(("[upar]", "[run.py]"))]
        if err_lines:
            print("--- log của run.py (stderr, %d dòng cuối) ---\n%s" % (min(len(err_lines), 12), "\n".join(err_lines[-12:])))
        if r.returncode != 0:
            raise RuntimeError("Mô phỏng ingestion LỖI:\n" + (r.stderr or "")[-2500:])
        return np.load(out)
