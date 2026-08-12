#!/usr/bin/env python3
"""Build audited SSL label-ratio manifests from fixed train/val/test manifests.

The input train/val/test manifests must already be video-disjoint.  This tool
never moves frames between them: it only derives nested labeled/unlabeled
partitions inside the supplied training manifest.  Therefore an outer test
split remains untouched and a video-wise inner validation split is preserved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

from PIL import Image
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset.semi import _parse_manifest_line, _video_frame_key


def read_rows(path: Path) -> list[str]:
    rows = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        row = raw.strip()
        if row and _parse_manifest_line(row) is not None:
            rows.append(row)
    if not rows:
        raise RuntimeError(f"empty or invalid manifest: {path}")
    return rows


def image_path(row: str) -> str:
    parsed = _parse_manifest_line(row)
    assert parsed is not None
    return str(parsed[0])


def mask_path(row: str) -> str:
    parsed = _parse_manifest_line(row)
    if parsed is None or not parsed[1]:
        raise RuntimeError(f"expected a labeled image/mask row: {row}")
    return str(parsed[1])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def video_balanced_order(rows: list[str], seed: int) -> list[str]:
    """Deterministic round-robin order, evenly spreading every video."""
    groups: dict[str, list[str]] = {}
    for row in rows:
        video, _ = _video_frame_key(row)
        groups.setdefault(str(video), []).append(row)
    rng = random.Random(seed)
    videos = sorted(groups)
    rng.shuffle(videos)
    for video in videos:
        groups[video].sort(key=lambda row: _video_frame_key(row)[1])
        # A phase shift preserves temporal spacing without always selecting
        # each sequence from its first frame.
        sequence = groups[video]
        if len(sequence) > 1:
            shift = rng.randrange(len(sequence))
            groups[video] = sequence[shift:] + sequence[:shift]
    output: list[str] = []
    index = 0
    while True:
        emitted = False
        for video in videos:
            sequence = groups[video]
            if index < len(sequence):
                output.append(sequence[index])
                emitted = True
        if not emitted:
            break
        index += 1
    return output


def class_hist(rows: list[str], data_root: Path, nclass: int,
               cache: dict[str, np.ndarray]) -> np.ndarray:
    total = np.zeros(nclass, dtype=np.int64)
    for row in rows:
        rel = mask_path(row)
        if rel not in cache:
            path = data_root / rel
            if not path.is_file():
                raise RuntimeError(f"missing mask: {path}")
            mask = np.asarray(Image.open(path), dtype=np.int64)
            valid = mask[(mask >= 0) & (mask < nclass)]
            cache[rel] = np.bincount(valid, minlength=nclass)[:nclass]
        total += cache[rel]
    return total


def write_rows(path: Path, rows: list[str]) -> None:
    path.write_text("".join(f"{row}\n" for row in rows), encoding="utf-8", newline="\n")


def require_disjoint(named_rows: dict[str, list[str]]) -> None:
    named_images = {name: {image_path(row) for row in rows}
                    for name, rows in named_rows.items()}
    names = list(named_images)
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            if named_images[left] & named_images[right]:
                raise RuntimeError(f"{left}/{right} overlap")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--ratios", type=float, nargs="+", default=[0.05, 0.10])
    parser.add_argument("--seed-start", type=int, default=8116)
    parser.add_argument("--seed-attempts", type=int, default=10000)
    parser.add_argument("--nclass", type=int, required=True)
    args = parser.parse_args()

    output_root = args.output_root.resolve()
    if output_root.exists():
        raise RuntimeError(f"refusing to overwrite existing output: {output_root}")
    ratios = sorted(set(args.ratios))
    if not ratios or any(ratio <= 0.0 or ratio >= 1.0 for ratio in ratios):
        raise RuntimeError("ratios must be in (0, 1)")

    train = read_rows(args.train_manifest.resolve())
    val = read_rows(args.val_manifest.resolve())
    test = read_rows(args.test_manifest.resolve())
    require_disjoint({"train": train, "val": val, "test": test})
    cache: dict[str, np.ndarray] = {}
    full_hist = class_hist(train, args.data_root.resolve(), args.nclass, cache)
    foreground = np.flatnonzero(full_hist[1:]) + 1
    if not len(foreground):
        raise RuntimeError("training split has no foreground pixels")

    selections = None
    selected_seed = None
    for attempt in range(args.seed_attempts):
        seed = args.seed_start + attempt
        ordered = video_balanced_order(train, seed)
        candidate = {
            ratio: ordered[:max(1, int(len(ordered) * ratio + 0.5))]
            for ratio in ratios
        }
        smallest_hist = class_hist(candidate[ratios[0]], args.data_root.resolve(), args.nclass, cache)
        if np.all(smallest_hist[foreground] > 0):
            selections = candidate
            selected_seed = seed
            break
    if selections is None or selected_seed is None:
        raise RuntimeError("no seed satisfies foreground coverage at smallest ratio")

    output_root.mkdir(parents=True)
    train_videos = sorted({str(_video_frame_key(row)[0]) for row in train})
    val_videos = sorted({str(_video_frame_key(row)[0]) for row in val})
    test_videos = sorted({str(_video_frame_key(row)[0]) for row in test})
    audit = {
        "dataset": args.dataset,
        "protocol": "fixed video-wise train/val/test; nested frame-ratio SSL partitions inside train",
        "selected_seed": selected_seed,
        "source": {
            "train": str(args.train_manifest.resolve()),
            "val": str(args.val_manifest.resolve()),
            "test": str(args.test_manifest.resolve()),
            "sha256": {
                "train": sha256(args.train_manifest.resolve()),
                "val": sha256(args.val_manifest.resolve()),
                "test": sha256(args.test_manifest.resolve()),
            },
        },
        "videos": {"train": train_videos, "val": val_videos, "test": test_videos},
        "rows": {"train": len(train), "val": len(val), "test": len(test)},
        "ratios": {},
    }
    all_images = {image_path(row) for row in train}
    for ratio in ratios:
        labeled = sorted(selections[ratio], key=lambda row: _video_frame_key(row))
        labeled_images = {image_path(row) for row in labeled}
        unlabeled = [image_path(row) for row in train if image_path(row) not in labeled_images]
        if labeled_images & set(unlabeled) or labeled_images | set(unlabeled) != all_images:
            raise RuntimeError(f"r={ratio:.2f} does not partition train")
        root = output_root / f"ratio_{ratio:.2f}"
        root.mkdir()
        write_rows(root / "labeled.txt", labeled)
        write_rows(root / "unlabeled.txt", sorted(unlabeled))
        write_rows(root / "train.txt", train)
        write_rows(root / "val.txt", val)
        write_rows(root / "test.txt", test)
        hist = class_hist(labeled, args.data_root.resolve(), args.nclass, cache)
        audit["ratios"][f"{ratio:.2f}"] = {
            "labeled": len(labeled),
            "unlabeled": len(unlabeled),
            "labeled_fraction": len(labeled) / len(train),
            "foreground_pixels": hist.tolist(),
            "labeled_subset_of_larger_ratio": True,
            "labeled_unlabeled_partition_train": True,
        }
    (output_root / "audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
