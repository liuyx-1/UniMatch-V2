import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze Needle UniMatch mask class distribution.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument(
        "--split-files",
        nargs="+",
        default=None,
        help="Split txt files. Defaults to train/val/test under <data-root>/splits.",
    )
    parser.add_argument("--ignore-index", type=int, default=255)
    parser.add_argument("--nclass", type=int, default=3)
    parser.add_argument("--out-json", default=None)
    return parser.parse_args()


def split_line(line):
    parts = line.strip().split()
    if len(parts) < 2:
        raise ValueError(f"Expected 'image mask' line, got: {line}")
    return parts[0], parts[1]


def analyze_split(data_root, split_file, nclass, ignore_index):
    stats = {
        "frames": 0,
        "missing_masks": 0,
        "pixel_counts": {str(i): 0 for i in range(nclass)},
        str(ignore_index): 0,
        "frames_with_class": {str(i): 0 for i in range(nclass)},
        "all_ignore_frames": 0,
        "videos": {},
    }
    lines = Path(split_file).read_text(encoding="utf-8").splitlines()
    for line in tqdm(lines, desc=Path(split_file).name, unit="frame"):
        if not line.strip():
            continue
        _, mask_rel = split_line(line)
        mask_path = Path(data_root) / mask_rel
        video = Path(mask_rel).parts[1] if len(Path(mask_rel).parts) >= 3 else "unknown"
        video_stats = stats["videos"].setdefault(
            video,
            {
                "frames": 0,
                "pixel_counts": {str(i): 0 for i in range(nclass)},
                str(ignore_index): 0,
                "frames_with_class": {str(i): 0 for i in range(nclass)},
                "all_ignore_frames": 0,
            },
        )
        if not mask_path.exists():
            stats["missing_masks"] += 1
            continue
        mask = np.array(Image.open(mask_path), dtype=np.uint8)
        values, counts = np.unique(mask, return_counts=True)
        counts_dict = {int(v): int(c) for v, c in zip(values, counts)}
        stats["frames"] += 1
        video_stats["frames"] += 1
        valid = False
        for cls_id in range(nclass):
            count = counts_dict.get(cls_id, 0)
            stats["pixel_counts"][str(cls_id)] += count
            video_stats["pixel_counts"][str(cls_id)] += count
            if count > 0:
                valid = True
                stats["frames_with_class"][str(cls_id)] += 1
                video_stats["frames_with_class"][str(cls_id)] += 1
        ignore_count = counts_dict.get(ignore_index, 0)
        stats[str(ignore_index)] += ignore_count
        video_stats[str(ignore_index)] += ignore_count
        if not valid:
            stats["all_ignore_frames"] += 1
            video_stats["all_ignore_frames"] += 1
    return stats


def main():
    args = parse_args()
    data_root = Path(args.data_root).resolve()
    split_files = args.split_files
    if split_files is None:
        split_files = [
            str(data_root / "splits" / "train.txt"),
            str(data_root / "splits" / "val.txt"),
            str(data_root / "splits" / "test.txt"),
        ]

    result = {}
    for split_file in split_files:
        split_name = Path(split_file).stem
        result[split_name] = analyze_split(data_root, split_file, args.nclass, args.ignore_index)

    text = json.dumps(result, indent=2)
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(text + "\n", encoding="utf-8")
        print(f"[OK] wrote {args.out_json}")
    print(text)

    for split_name, stats in result.items():
        missing = [str(i) for i in range(args.nclass) if stats["pixel_counts"].get(str(i), 0) == 0]
        if missing:
            print(f"[WARN] {split_name}: missing classes {missing}", file=sys.stderr)


if __name__ == "__main__":
    main()
