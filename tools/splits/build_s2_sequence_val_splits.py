#!/usr/bin/env python3
"""Create audited, sequence-disjoint inner-validation manifests for SASR S2."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image


SPLIT_ROOT = Path(
    "/root/autodl-tmp/data/autonomous_surgery/splits/sasr_s2_seqval_v1_20260810"
)

SPECS = {
    "endoscapes_seg50": {
        "data_root": "/root/autodl-tmp/data/autonomous_surgery/public_data/endoscapes_seg50_processed",
        "train": "/root/autodl-tmp/data/autonomous_surgery/public_data/endoscapes_seg50_processed/splits/050/labeled.txt",
        "test": "/root/autodl-tmp/data/autonomous_surgery/public_data/endoscapes_seg50_processed/splits/test.txt",
        "output": "endoscapes_seg50/s1_050_seqval20",
        "nclass": 7,
        "selected_val_sequences": ["10", "105", "117", "43", "82"],
        "sequence": lambda image: Path(image).name.split("_")[0],
    },
    "endovis2018": {
        "data_root": "/root/autodl-tmp/data/autonomous_surgery/public_data/endovis2018_processed",
        "train": "/root/autodl-tmp/data/autonomous_surgery/splits/official_no_val_v4_20260723_seed8116/endovis2018/official_train15_test4/ratio_0.30/train.txt",
        "test": "/root/autodl-tmp/data/autonomous_surgery/splits/official_no_val_v4_20260723_seed8116/endovis2018/official_train15_test4/ratio_0.30/test.txt",
        "output": "endovis2018/official_train15_test4_seqval20",
        "nclass": 12,
        "selected_val_sequences": ["seq04", "seq12", "seq14"],
        "sequence": lambda image: re.match(r"(seq\d+)_frame", Path(image).name).group(1),
    },
    "cataract1k_fold0": {
        "data_root": "/root/autodl-tmp/data/autonomous_surgery/public_data/cataract1k_official13",
        "train": "/root/autodl-tmp/data/autonomous_surgery/splits/official_no_val_v4_seedbalanced13_20260731/cataract1k/fold_0/ratio_0.30/train.txt",
        "test": "/root/autodl-tmp/data/autonomous_surgery/splits/official_no_val_v4_seedbalanced13_20260731/cataract1k/fold_0/ratio_0.30/test.txt",
        "output": "cataract1k/fold_0_seqval20",
        "nclass": 13,
        "selected_val_sequences": ["case5016", "case5032", "case5300", "case5309", "case5335"],
        "sequence": lambda image: re.match(r"(case_?\d+)_", Path(image).name).group(1),
    },
}


def read_pairs(path: Path) -> list[tuple[str, str]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            raise RuntimeError(f"malformed pair in {path}: {line!r}")
        rows.append((parts[0], parts[1]))
    return rows


def write_pairs(path: Path, rows: list[tuple[str, str]]) -> None:
    path.write_text("\n".join(f"{image}\t{mask}" for image, mask in rows) + "\n", encoding="utf-8")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ids(rows: list[tuple[str, str]]) -> set[str]:
    return {image for image, _ in rows}


def class_presence(root: Path, rows: list[tuple[str, str]], nclass: int) -> list[int]:
    hist = np.zeros(nclass, dtype=np.int64)
    for _, mask in rows:
        mask_path = root / mask
        if not mask_path.is_file():
            raise FileNotFoundError(mask_path)
        values = np.asarray(Image.open(mask_path))
        hist += np.bincount(values.reshape(-1), minlength=nclass)[:nclass]
    return [int(v) for v in hist]


def build(name: str, spec: dict) -> None:
    root = Path(spec["data_root"])
    source_train = Path(spec["train"])
    source_test = Path(spec["test"])
    train_all = read_pairs(source_train)
    test = read_pairs(source_test)
    selected = set(spec["selected_val_sequences"])
    grouped = {spec["sequence"](image) for image, _ in train_all}
    if not selected <= grouped:
        raise RuntimeError(f"{name}: selected sequences absent: {sorted(selected - grouped)}")
    val = [row for row in train_all if spec["sequence"](row[0]) in selected]
    train = [row for row in train_all if spec["sequence"](row[0]) not in selected]
    if not train or not val or not test:
        raise RuntimeError(f"{name}: empty train, val, or test partition")
    if ids(train) & ids(val) or ids(train) & ids(test) or ids(val) & ids(test):
        raise RuntimeError(f"{name}: image overlap across partitions")
    for image, mask in train + val + test:
        if not (root / image).is_file() or not (root / mask).is_file():
            raise FileNotFoundError(f"{name}: missing pair {(root / image, root / mask)}")
    train_hist = class_presence(root, train, int(spec["nclass"]))
    val_hist = class_presence(root, val, int(spec["nclass"]))
    if any(v == 0 for v in train_hist[1:]) or any(v == 0 for v in val_hist[1:]):
        raise RuntimeError(f"{name}: foreground-class coverage missing in train or val")
    out = SPLIT_ROOT / spec["output"]
    if out.exists():
        raise FileExistsError(f"refusing to overwrite existing split: {out}")
    out.mkdir(parents=True)
    write_pairs(out / "train.txt", train)
    write_pairs(out / "val.txt", val)
    write_pairs(out / "test.txt", test)
    audit = {
        "schema": "sasr-s2-sequence-val-v1",
        "dataset": name,
        "source_train": str(source_train),
        "source_test": str(source_test),
        "selected_val_sequences": sorted(selected),
        "all_training_sequences": sorted(grouped),
        "counts": {"train": len(train), "val": len(val), "test": len(test)},
        "val_ratio_of_outer_train": len(val) / len(train_all),
        "outer_test_untouched": True,
        "pairwise_image_disjoint": True,
        "train_pixel_histogram": train_hist,
        "val_pixel_histogram": val_hist,
        "foreground_covered_train": True,
        "foreground_covered_val": True,
        "sha256": {"train.txt": digest(out / "train.txt"), "val.txt": digest(out / "val.txt"), "test.txt": digest(out / "test.txt")},
    }
    (out / "audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"dataset": name, **audit["counts"], "val_sequences": audit["selected_val_sequences"]}))


if __name__ == "__main__":
    for dataset_name, dataset_spec in SPECS.items():
        build(dataset_name, dataset_spec)
