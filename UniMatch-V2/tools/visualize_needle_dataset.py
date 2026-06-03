import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
COLORS = {
    0: (70, 70, 70),
    1: (0, 0, 255),
    2: (0, 255, 0),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create per-video overlay videos to inspect needle semantic masks."
    )
    parser.add_argument(
        "--data-root",
        required=True,
        help="Either UniMatch dataset root or original frame root.",
    )
    parser.add_argument(
        "--ann-root",
        default=None,
        help="Original SAM2-Plus annotation root. If omitted, read <data-root>/masks for UniMatch format.",
    )
    parser.add_argument("--out-root", required=True, help="Output directory for mp4 videos and summary json.")
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument("--max-width", type=int, default=1280)
    parser.add_argument("--ignore-value", type=int, default=255)
    return parser.parse_args()


def natural_key(path):
    text = Path(path).stem
    parts, buf, is_digit = [], "", False
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


def list_videos(data_root, ann_root):
    data_root = Path(data_root)
    if ann_root is None:
        image_root = data_root / "images"
        mask_root = data_root / "masks"
    else:
        image_root = data_root
        mask_root = Path(ann_root) / "semantic_masks"

    videos = []
    for p in sorted(image_root.iterdir(), key=lambda x: x.name):
        if p.is_dir() and any(f.suffix in IMAGE_EXTS for f in p.iterdir() if f.is_file()):
            if (mask_root / p.name).is_dir():
                videos.append(p.name)
    if not videos:
        raise RuntimeError(f"No videos found. image_root={image_root}, mask_root={mask_root}")
    return image_root, mask_root, videos


def read_mask(mask_path):
    return np.array(Image.open(mask_path), dtype=np.uint8)


def overlay_mask(image_bgr, mask, alpha, ignore_value):
    overlay = image_bgr.copy()
    valid = mask != ignore_value
    for cls_id, color in COLORS.items():
        overlay[mask == cls_id] = color
    out = image_bgr.copy()
    out[valid] = cv2.addWeighted(image_bgr, 1.0 - alpha, overlay, alpha, 0)[valid]
    return out


def draw_info(image, video, frame_name, counts, valid_ratio):
    text = (
        f"{video}/{frame_name}  "
        f"c0={counts.get(0, 0)} c1={counts.get(1, 0)} "
        f"ignore={counts.get(255, 0)} valid={valid_ratio:.4f}"
    )
    cv2.rectangle(image, (0, 0), (image.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(image, text, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)


def resize_if_needed(image, max_width):
    h, w = image.shape[:2]
    if max_width <= 0 or w <= max_width:
        return image
    scale = max_width / float(w)
    return cv2.resize(image, (max_width, int(round(h * scale))), interpolation=cv2.INTER_AREA)


def main():
    args = parse_args()
    data_root = Path(args.data_root).resolve()
    ann_root = Path(args.ann_root).resolve() if args.ann_root else None
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    image_root, mask_root, videos = list_videos(data_root, ann_root)
    summary = {
        "data_root": str(data_root),
        "ann_root": str(ann_root) if ann_root else None,
        "image_root": str(image_root),
        "mask_root": str(mask_root),
        "videos": {},
    }

    for video in tqdm(videos, desc="Videos", unit="video"):
        frames = sorted(
            [p for p in (image_root / video).iterdir() if p.is_file() and p.suffix in IMAGE_EXTS],
            key=natural_key,
        )
        if not frames:
            continue

        writer = None
        video_summary = {
            "frames": 0,
            "all_ignore_frames": 0,
            "missing_masks": 0,
            "pixel_counts": {},
        }
        out_path = out_root / f"{video}.mp4"

        for frame_path in tqdm(frames, desc=video, unit="frame", leave=False):
            mask_path = mask_root / video / f"{frame_path.stem}.png"
            image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if image is None:
                continue

            if not mask_path.exists():
                video_summary["missing_masks"] += 1
                canvas = resize_if_needed(image, args.max_width)
                cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 34), (0, 0, 0), -1)
                cv2.putText(
                    canvas,
                    f"{video}/{frame_path.name}  MISSING MASK",
                    (8, 23),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA,
                )
            else:
                mask = read_mask(mask_path)
                if mask.shape[:2] != image.shape[:2]:
                    mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
                values, counts = np.unique(mask, return_counts=True)
                counts_dict = {int(v): int(c) for v, c in zip(values, counts)}
                for value, count in counts_dict.items():
                    key = str(value)
                    video_summary["pixel_counts"][key] = video_summary["pixel_counts"].get(key, 0) + count
                valid_ratio = float(np.mean(mask != args.ignore_value))
                if valid_ratio == 0.0:
                    video_summary["all_ignore_frames"] += 1
                canvas = overlay_mask(image, mask, args.alpha, args.ignore_value)
                canvas = resize_if_needed(canvas, args.max_width)
                draw_info(canvas, video, frame_path.name, counts_dict, valid_ratio)

            if writer is None:
                h, w = canvas.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(out_path), fourcc, args.fps, (w, h))
                if not writer.isOpened():
                    raise RuntimeError(f"Cannot open video writer: {out_path}")
            writer.write(canvas)
            video_summary["frames"] += 1

        if writer is not None:
            writer.release()
        summary["videos"][video] = video_summary

    with open(out_root / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[OK] wrote videos to {out_root}")
    print(f"[OK] wrote summary to {out_root / 'summary.json'}")


if __name__ == "__main__":
    main()
