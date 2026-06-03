import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert SAM2-Plus needle video annotations to UniMatch-V2 format."
    )
    parser.add_argument("--src-root", required=True, help="Frame root, e.g. /data/data/needle/frames_5fps")
    parser.add_argument(
        "--ann-root",
        default=None,
        help="SAM2-Plus annotation root. Defaults to <src-root>/sam2p_annotations/all",
    )
    parser.add_argument("--dst-root", required=True, help="Output UniMatch-V2 dataset root.")
    parser.add_argument("--nclass", type=int, default=3)
    parser.add_argument("--ignore-value", type=int, default=255)
    parser.add_argument(
        "--semantic-background",
        action="store_true",
        help="Remap SAM2-Plus masks from old 0/1/255 to semantic 0/1/2: 255->0 background, 0->1, 1->2.",
    )
    parser.add_argument(
        "--policy",
        choices=["video_holdout", "temporal_blocks"],
        default="temporal_blocks",
        help="Split by whole videos or by contiguous temporal blocks inside each video.",
    )
    parser.add_argument(
        "--val-videos",
        default="",
        help="Comma-separated video names for validation when --policy video_holdout.",
    )
    parser.add_argument(
        "--test-videos",
        default="",
        help="Comma-separated video names for testing when --policy video_holdout.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument(
        "--gap",
        type=int,
        default=5,
        help="Unused frame gap between train/val/test temporal blocks.",
    )
    parser.add_argument(
        "--copy-mode",
        choices=["copy", "hardlink", "symlink"],
        default="copy",
        help="How to place images/masks in dst-root.",
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip frames without semantic mask instead of failing.",
    )
    parser.add_argument(
        "--skip-empty",
        action="store_true",
        help="Skip frames whose mask contains only ignore-value.",
    )
    return parser.parse_args()


def natural_key(path):
    text = Path(path).stem
    parts = []
    buf = ""
    is_digit = False
    for ch in text:
        if ch.isdigit() != is_digit:
            if buf:
                parts.append(int(buf) if is_digit else buf)
            buf = ch
            is_digit = ch.isdigit()
        else:
            buf += ch
    if buf:
        parts.append(int(buf) if is_digit else buf)
    return parts


def list_videos(src_root):
    ignored = {"sam2p_annotations", "images", "masks", "splits"}
    videos = []
    for p in sorted(Path(src_root).iterdir(), key=lambda x: x.name):
        if p.is_dir() and p.name not in ignored and not p.name.startswith("."):
            if any(f.suffix in IMAGE_EXTS for f in p.iterdir() if f.is_file()):
                videos.append(p.name)
    if not videos:
        raise RuntimeError(f"No video frame folders found under {src_root}")
    return videos


def list_frames(src_root, video):
    video_dir = Path(src_root) / video
    frames = [p for p in video_dir.iterdir() if p.is_file() and p.suffix in IMAGE_EXTS]
    frames = sorted(frames, key=natural_key)
    if not frames:
        raise RuntimeError(f"No frames found in {video_dir}")
    return frames


def parse_csv_names(value):
    return {x.strip() for x in value.split(",") if x.strip()}


def split_temporal(frames, train_ratio, val_ratio, test_ratio, gap):
    n = len(frames)
    if n == 0:
        return [], [], []
    total = train_ratio + val_ratio + test_ratio
    train_ratio, val_ratio, test_ratio = train_ratio / total, val_ratio / total, test_ratio / total

    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    n_test = n - n_train - n_val

    train_end = max(0, min(n, n_train))
    val_start = min(n, train_end + gap)
    val_end = max(val_start, min(n, val_start + n_val))
    test_start = min(n, val_end + gap)

    train = frames[:train_end]
    val = frames[val_start:val_end]
    test = frames[test_start:]
    if not test and n_test > 0:
        test = frames[-n_test:]
    return train, val, test


def place_file(src, dst, mode):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "hardlink":
        os.link(src, dst)
    else:
        os.symlink(src, dst)


def load_mask(mask_path):
    mask = Image.open(mask_path)
    return np.array(mask, dtype=np.uint8)


def save_mask(mask, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8), mode="L").save(dst)


def validate_mask(mask, nclass, ignore_value, mask_path):
    allowed = set(range(nclass)) | {ignore_value}
    values = set(np.unique(mask).astype(int).tolist())
    bad = sorted(values - allowed)
    if bad:
        raise ValueError(
            f"Mask {mask_path} has invalid values {bad}; expected classes 0..{nclass - 1} and ignore {ignore_value}"
        )


def remap_with_background(mask, ignore_value):
    remapped = np.zeros_like(mask, dtype=np.uint8)
    valid = mask != ignore_value
    remapped[valid] = mask[valid] + 1
    return remapped


def add_record(records, split_name, video, frame_path, src_root, ann_root, dst_root, args):
    stem = frame_path.stem
    mask_path = Path(ann_root) / "semantic_masks" / video / f"{stem}.png"
    if not mask_path.exists():
        if args.skip_missing:
            return False
        raise FileNotFoundError(f"Missing semantic mask: {mask_path}")

    mask = load_mask(mask_path)
    validate_mask(mask, 2 if args.semantic_background else args.nclass, args.ignore_value, mask_path)
    if args.skip_empty and np.all(mask == args.ignore_value):
        return False
    if args.semantic_background:
        mask = remap_with_background(mask, args.ignore_value)
        validate_mask(mask, args.nclass, args.ignore_value, mask_path)

    img_rel = Path("images") / video / frame_path.name
    mask_rel = Path("masks") / video / f"{stem}.png"
    place_file(frame_path, Path(dst_root) / img_rel, args.copy_mode)
    save_mask(mask, Path(dst_root) / mask_rel)

    line = f"{img_rel.as_posix()} {mask_rel.as_posix()}"
    records[split_name].append(line)
    return True


def write_lines(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        if lines:
            f.write("\n")


def main():
    args = parse_args()
    src_root = Path(args.src_root).resolve()
    ann_root = Path(args.ann_root).resolve() if args.ann_root else src_root / "sam2p_annotations" / "all"
    dst_root = Path(args.dst_root).resolve()
    dst_root.mkdir(parents=True, exist_ok=True)

    videos = list_videos(src_root)
    records = {"train": [], "val": [], "test": []}
    split_manifest = {}

    if args.policy == "video_holdout":
        val_videos = parse_csv_names(args.val_videos)
        test_videos = parse_csv_names(args.test_videos)
        unknown = (val_videos | test_videos) - set(videos)
        if unknown:
            raise ValueError(f"Unknown videos in val/test lists: {sorted(unknown)}; available={videos}")
        if not val_videos or not test_videos:
            raise ValueError("--policy video_holdout requires --val-videos and --test-videos")

    for video in tqdm(videos, desc="Videos", unit="video"):
        frames = list_frames(src_root, video)
        if args.policy == "video_holdout":
            if video in parse_csv_names(args.val_videos):
                split_groups = {"val": frames}
            elif video in parse_csv_names(args.test_videos):
                split_groups = {"test": frames}
            else:
                split_groups = {"train": frames}
        else:
            train, val, test = split_temporal(
                frames, args.train_ratio, args.val_ratio, args.test_ratio, args.gap
            )
            split_groups = {"train": train, "val": val, "test": test}

        split_manifest[video] = {k: [p.name for p in v] for k, v in split_groups.items()}
        for split_name, split_frames in split_groups.items():
            iterator = tqdm(
                split_frames,
                desc=f"{video}:{split_name}",
                unit="frame",
                leave=False,
            )
            for frame_path in iterator:
                add_record(records, split_name, video, frame_path, src_root, ann_root, dst_root, args)

    splits_dir = dst_root / "splits"
    write_lines(splits_dir / "train.txt", records["train"])
    write_lines(splits_dir / "val.txt", records["val"])
    write_lines(splits_dir / "test.txt", records["test"])
    write_lines(splits_dir / "labeled.txt", records["train"])
    # UniMatch train_u ignores masks; image-only lines avoid depending on labels for unlabeled branch.
    write_lines(splits_dir / "unlabeled.txt", [line.split(" ")[0] for line in records["train"]])

    manifest = {
        "src_root": str(src_root),
        "ann_root": str(ann_root),
        "dst_root": str(dst_root),
        "nclass": args.nclass,
        "ignore_value": args.ignore_value,
        "semantic_background": args.semantic_background,
        "policy": args.policy,
        "videos": videos,
        "counts": {k: len(v) for k, v in records.items()},
        "split_manifest": split_manifest,
    }
    with open(dst_root / "split_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"[OK] Wrote UniMatch-V2 dataset to {dst_root}")
    print(f"[OK] train={len(records['train'])} val={len(records['val'])} test={len(records['test'])}")
    print(f"[OK] splits: {splits_dir}")


if __name__ == "__main__":
    main()
