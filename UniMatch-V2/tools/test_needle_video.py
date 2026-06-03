import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.semseg.dpt import DPT


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
COLORS = {
    0: (70, 70, 70),
    1: (0, 0, 255),
    2: (0, 255, 0),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run Needle UniMatch-V2 inference and export videos.")
    parser.add_argument("--config", default="configs/needle.yaml")
    parser.add_argument("--checkpoint", required=True, help="Path to best.pth or latest.pth.")
    parser.add_argument("--data-root", default=None, help="UniMatch dataset root. Defaults to cfg['data_root'].")
    parser.add_argument("--split-file", default=None, help="Optional split txt, e.g. splits/test.txt.")
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--weights-key", choices=["model", "model_ema"], default="model_ema")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument("--max-width", type=int, default=1280)
    parser.add_argument("--ignore-index", type=int, default=255)
    parser.add_argument("--save-masks", action="store_true")
    parser.add_argument(
        "--merged-video",
        default=None,
        help="Optional filename for one long video containing all processed videos, e.g. all_test_videos.mp4.",
    )
    parser.add_argument(
        "--valid-gt-only",
        action="store_true",
        help="When GT masks are available, draw predictions only on pixels where GT is not ignore-index.",
    )
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


def build_model(cfg):
    model_configs = {
        "small": {"encoder_size": "small", "features": 64, "out_channels": [48, 96, 192, 384]},
        "base": {"encoder_size": "base", "features": 128, "out_channels": [96, 192, 384, 768]},
        "large": {"encoder_size": "large", "features": 256, "out_channels": [256, 512, 1024, 1024]},
        "giant": {"encoder_size": "giant", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
    }
    backbone_size = cfg["backbone"].split("_")[-1]
    return DPT(**{**model_configs[backbone_size], "nclass": cfg["nclass"]})


def clean_state_dict(state_dict):
    clean = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module."):]
        clean[key] = value
    return clean


def load_checkpoint(model, checkpoint_path, weights_key, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if weights_key not in checkpoint:
        available = [k for k in ["model", "model_ema"] if k in checkpoint]
        raise KeyError(f"{weights_key} not found in checkpoint. Available model keys: {available}")
    model.load_state_dict(clean_state_dict(checkpoint[weights_key]), strict=True)
    return checkpoint


def read_split(data_root, split_file):
    records = {}
    with open(split_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            img_rel = Path(parts[0])
            if len(img_rel.parts) < 3:
                continue
            video = img_rel.parts[1]
            frame_path = Path(data_root) / img_rel
            mask_path = Path(data_root) / parts[1] if len(parts) > 1 else None
            records.setdefault(video, []).append((frame_path, mask_path))
    for video in records:
        records[video] = sorted(records[video], key=lambda x: natural_key(x[0]))
    return records


def list_all_images(data_root):
    image_root = Path(data_root) / "images"
    records = {}
    for video_dir in sorted([p for p in image_root.iterdir() if p.is_dir()], key=lambda x: x.name):
        frames = sorted(
            [p for p in video_dir.iterdir() if p.is_file() and p.suffix in IMAGE_EXTS],
            key=natural_key,
        )
        records[video_dir.name] = [(p, Path(data_root) / "masks" / video_dir.name / f"{p.stem}.png") for p in frames]
    return records


def pad_to_multiple(image_rgb, multiple=14):
    h, w = image_rgb.shape[:2]
    new_h = int(np.ceil(h / multiple) * multiple)
    new_w = int(np.ceil(w / multiple) * multiple)
    pad = np.zeros((new_h, new_w, 3), dtype=image_rgb.dtype)
    pad[:h, :w] = image_rgb
    return pad, h, w


def preprocess(image_rgb, device):
    padded, ori_h, ori_w = pad_to_multiple(image_rgb, multiple=14)
    tensor = torch.from_numpy(padded).permute(2, 0, 1).float() / 255.0
    tensor = (tensor - MEAN) / STD
    return tensor.unsqueeze(0).to(device), ori_h, ori_w


@torch.inference_mode()
def predict(model, image_bgr, device):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    inp, ori_h, ori_w = preprocess(image_rgb, device)
    logits = model(inp)
    logits = F.interpolate(logits, size=inp.shape[-2:], mode="bilinear", align_corners=True)
    pred = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
    return pred[:ori_h, :ori_w]


def overlay_prediction(image_bgr, pred, alpha, valid_mask=None):
    color = image_bgr.copy()
    for cls_id, bgr in COLORS.items():
        color[pred == cls_id] = bgr
    blended = cv2.addWeighted(image_bgr, 1.0 - alpha, color, alpha, 0)
    if valid_mask is None:
        return blended
    out = image_bgr.copy()
    out[valid_mask] = blended[valid_mask]
    return out


def resize_if_needed(image, max_width):
    h, w = image.shape[:2]
    if max_width <= 0 or w <= max_width:
        return image
    scale = max_width / float(w)
    return cv2.resize(image, (max_width, int(round(h * scale))), interpolation=cv2.INTER_AREA)


def update_metrics(metrics, pred, gt, nclass, ignore_index):
    valid = gt != ignore_index
    for cls_id in range(nclass):
        p = (pred == cls_id) & valid
        g = (gt == cls_id) & valid
        metrics["intersection"][cls_id] += int(np.logical_and(p, g).sum())
        metrics["union"][cls_id] += int(np.logical_or(p, g).sum())


def draw_info(canvas, video, frame_name, pred):
    c0 = int((pred == 0).sum())
    c1 = int((pred == 1).sum())
    text = f"{video}/{frame_name}  pred_c0={c0} pred_c1={c1}"
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(canvas, text, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)


def main():
    args = parse_args()
    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
    data_root = Path(args.data_root or cfg["data_root"]).resolve()
    out_root = Path(args.out_root).resolve()
    video_root = out_root / "videos"
    mask_root = out_root / "pred_masks"
    video_root.mkdir(parents=True, exist_ok=True)
    if args.save_masks:
        mask_root.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    model = build_model(cfg).to(device)
    checkpoint = load_checkpoint(model, args.checkpoint, args.weights_key, device)
    model.eval()

    records = read_split(data_root, args.split_file) if args.split_file else list_all_images(data_root)
    metrics = {
        "intersection": np.zeros(cfg["nclass"], dtype=np.int64),
        "union": np.zeros(cfg["nclass"], dtype=np.int64),
    }
    summary = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "weights_key": args.weights_key,
        "data_root": str(data_root),
        "epoch": checkpoint.get("epoch"),
        "videos": {},
    }
    merged_writer = None
    merged_path = out_root / args.merged_video if args.merged_video else None

    for video, frames in tqdm(records.items(), desc="Videos", unit="video"):
        writer = None
        out_video = video_root / f"{video}.mp4"
        summary["videos"][video] = {"frames": 0}
        if args.save_masks:
            (mask_root / video).mkdir(parents=True, exist_ok=True)

        for image_path, gt_path in tqdm(frames, desc=video, unit="frame", leave=False):
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                continue
            pred = predict(model, image, device)
            if args.save_masks:
                Image.fromarray(pred, mode="L").save(mask_root / video / f"{image_path.stem}.png")

            valid_mask = None
            if gt_path is not None and gt_path.exists():
                gt = np.array(Image.open(gt_path), dtype=np.uint8)
                if gt.shape != pred.shape:
                    gt = cv2.resize(gt, (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_NEAREST)
                update_metrics(metrics, pred, gt, cfg["nclass"], args.ignore_index)
                if args.valid_gt_only:
                    valid_mask = gt != args.ignore_index

            canvas = overlay_prediction(image, pred, args.alpha, valid_mask=valid_mask)
            canvas = resize_if_needed(canvas, args.max_width)
            draw_info(canvas, video, image_path.name, pred)
            if writer is None:
                h, w = canvas.shape[:2]
                writer = cv2.VideoWriter(
                    str(out_video),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    args.fps,
                    (w, h),
                )
                if not writer.isOpened():
                    raise RuntimeError(f"Cannot create video: {out_video}")
            writer.write(canvas)
            if merged_path is not None:
                if merged_writer is None:
                    h, w = canvas.shape[:2]
                    merged_writer = cv2.VideoWriter(
                        str(merged_path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        args.fps,
                        (w, h),
                    )
                    if not merged_writer.isOpened():
                        raise RuntimeError(f"Cannot create merged video: {merged_path}")
                merged_writer.write(canvas)
            summary["videos"][video]["frames"] += 1

        if writer is not None:
            writer.release()

    if merged_writer is not None:
        merged_writer.release()

    iou = metrics["intersection"] / np.maximum(metrics["union"], 1) * 100.0
    summary["metrics"] = {
        "iou": {str(i): float(v) for i, v in enumerate(iou)},
        "miou": float(iou.mean()),
        "intersection": metrics["intersection"].tolist(),
        "union": metrics["union"].tolist(),
    }
    with open(out_root / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[OK] videos: {video_root}")
    if merged_path is not None:
        print(f"[OK] merged video: {merged_path}")
    if args.save_masks:
        print(f"[OK] masks: {mask_root}")
    print(f"[OK] summary: {out_root / 'summary.json'}")
    print(f"[OK] mIoU: {summary['metrics']['miou']:.2f}")


if __name__ == "__main__":
    main()
