#!/usr/bin/env python3
"""Smoke-test official-no-val manifests with the repository dataset loader."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset.semi import SemiDataset, _read_manifest, _split_line, _video_frame_key


def image_ids(lines):
    return {_split_line(line)[0] for line in lines}


def video_count(lines):
    return len({_video_frame_key(line)[0] for line in lines})


def verify_ratio(dataset, data_root, ratio_dir, crop_size, max_frame_gap):
    paths = {
        name: ratio_dir / f"{name}.txt"
        for name in ("labeled", "unlabeled", "train", "test", "train_monitor")
    }
    if (ratio_dir / "val.txt").exists():
        raise RuntimeError(f"unexpected val.txt: {ratio_dir}")
    for name, path in paths.items():
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"missing/empty {name}: {path}")

    rows = {name: _read_manifest(str(path)) for name, path in paths.items()}
    labeled_ids = image_ids(rows["labeled"])
    unlabeled_ids = image_ids(rows["unlabeled"])
    train_ids = image_ids(rows["train"])
    test_ids = image_ids(rows["test"])
    if labeled_ids & unlabeled_ids:
        raise RuntimeError(f"labeled/unlabeled overlap: {ratio_dir}")
    if labeled_ids | unlabeled_ids != train_ids:
        raise RuntimeError(f"labeled+unlabeled != train: {ratio_dir}")
    if train_ids & test_ids:
        raise RuntimeError(f"train/test frame overlap: {ratio_dir}")

    labeled = SemiDataset(
        dataset, str(data_root), "train_l", crop_size,
        str(paths["labeled"]), temporal_pair=True,
    )
    unlabeled = SemiDataset(
        dataset, str(data_root), "train_u", crop_size,
        str(paths["unlabeled"]), temporal_pair=True,
    )
    test = SemiDataset(dataset, str(data_root), "val", id_path=str(paths["test"]))
    monitor = SemiDataset(
        dataset, str(data_root), "val", id_path=str(paths["train_monitor"])
    )

    labeled_sample = labeled[0]
    unlabeled_sample = unlabeled[0]
    test_sample = test[0]
    monitor_sample = monitor[0]
    return {
        "ratio_dir": str(ratio_dir),
        "labeled_frames": len(rows["labeled"]),
        "unlabeled_frames": len(rows["unlabeled"]),
        "train_frames": len(rows["train"]),
        "test_frames": len(rows["test"]),
        "labeled_videos": video_count(rows["labeled"]),
        "unlabeled_videos": video_count(rows["unlabeled"]),
        "train_videos": video_count(rows["train"]),
        "test_videos": video_count(rows["test"]),
        "labeled_temporal_pairs": int(sum(labeled.ref_valid or [])),
        "unlabeled_temporal_pairs": int(sum(unlabeled.ref_valid or [])),
        "labeled_image_shape": list(labeled_sample[0].shape),
        "unlabeled_image_shape": list(unlabeled_sample[0].shape),
        "test_image_shape": list(test_sample[0].shape),
        "monitor_image_shape": list(monitor_sample[0].shape),
        "no_val_manifest": True,
        "loader_smoke": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--crop-size", type=int, default=490)
    parser.add_argument("--temporal-max-frame-gap", type=int, default=3)
    args = parser.parse_args()

    ratio_dirs = sorted(path for path in args.split_root.glob("**/ratio_*") if path.is_dir())
    if not ratio_dirs:
        raise RuntimeError(f"no ratio directories under {args.split_root}")
    results = [
        verify_ratio(
            args.dataset,
            args.data_root.resolve(),
            ratio_dir.resolve(),
            args.crop_size,
            args.temporal_max_frame_gap,
        )
        for ratio_dir in ratio_dirs
    ]
    print(json.dumps({"verified": len(results), "results": results}, indent=2))


if __name__ == "__main__":
    main()
