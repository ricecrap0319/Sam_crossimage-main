"""
extract_and_label.py
--------------------
Creates more training data from your existing videos in two passes:

  PASS 1 – Frame extraction
    Samples every Nth frame from all videos in a folder, discards near-
    duplicate frames (perceptual hash), and saves the keepers as images.

  PASS 2 – Auto-labelling with your existing YOLO-OBB model
    Runs inference on every extracted image and writes a .txt label file
    (YOLO OBB format) alongside each image.  Low-confidence and very small
    detections are skipped so your label set stays clean.

After running this script:
  1. Open the output folder in a labelling tool (LabelImg / Roboflow /
     Label Studio) and *correct* any wrong boxes.
  2. Add the corrected images+labels to your training set and re-train.

Usage
─────
  python extract_and_label.py                    # uses defaults below
  python extract_and_label.py --video-dir vids   # custom video folder
"""

import os
import argparse
import struct, hashlib
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# ──────────────────────────────────────────────
#  CONFIGURATION  ← edit before running
# ──────────────────────────────────────────────
VIDEO_DIR           = "."           # folder that contains your .mp4 files
OUTPUT_DIR          = "labeling_dataset"  # extracted images + labels go here
FRAME_EVERY_N       = 15            # extract 1 frame every N frames
                                    # (15 @ 30fps ≈ 2 frames/sec)

# Auto-labelling settings
YOLO_PATH           = "YOLO_best.pt"
LABEL_CONF          = 0.15          # min confidence to include a detection
LABEL_IOU           = 0.2
MIN_BOX_PIXELS      = 8             # skip boxes smaller than this in any dim

# Deduplication – frames with perceptual-hash distance ≤ this are skipped
HASH_THRESHOLD      = 8             # 0 = identical; higher = more tolerant

# Augmentation for each extracted frame (creates extra label-ready images)
# Set to 0 to disable augmentation
AUGMENT_COPIES      = 2             # how many augmented copies per frame
# ──────────────────────────────────────────────


# ── Perceptual hash (dHash) ──────────────────────────────────────────────────
def dhash(image, hash_size=8):
    """Difference hash – fast near-duplicate detector."""
    gray   = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (hash_size + 1, hash_size))
    diff   = resized[:, 1:] > resized[:, :-1]
    return sum(b << i for i, b in enumerate(diff.flatten()))


def hash_distance(h1, h2):
    return bin(h1 ^ h2).count("1")


# ── Augmentation helpers ─────────────────────────────────────────────────────
def augment_frame(frame):
    """Return a list of augmented versions of the frame."""
    augmented = []
    h, w = frame.shape[:2]

    # 1. Random brightness / contrast
    alpha = np.random.uniform(0.7, 1.3)   # contrast
    beta  = np.random.randint(-30, 30)     # brightness
    aug1  = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
    augmented.append(("bright", aug1))

    # 2. Gaussian blur (simulates slight defocus on microscope)
    ksize = np.random.choice([3, 5])
    aug2  = cv2.GaussianBlur(frame, (ksize, ksize), 0)
    augmented.append(("blur", aug2))

    # 3. Horizontal flip  (crystals have no inherent chirality)
    aug3 = cv2.flip(frame, 1)
    augmented.append(("hflip", aug3))

    # 4. Small rotation (±15°)
    angle  = np.random.uniform(-15, 15)
    M      = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    aug4   = cv2.warpAffine(frame, M, (w, h),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT)
    augmented.append(("rot", aug4))

    return augmented[:AUGMENT_COPIES]


# ── Label writing ─────────────────────────────────────────────────────────────
def xyxy_to_yolo_obb(xyxy_tensor, img_w, img_h):
    """
    Convert an OBB xyxyxyxy tensor (8 values) to YOLO-OBB normalised format:
      x1 y1 x2 y2 x3 y3 x4 y4  (all 0-1)
    Falls back to axis-aligned box corners if obb.xyxyxyxy unavailable.
    """
    pts = xyxy_tensor.cpu().numpy().reshape(4, 2)  # (4,2)
    pts[:, 0] /= img_w
    pts[:, 1] /= img_h
    pts = pts.clip(0, 1)
    return pts.flatten().tolist()


def write_label_file(label_path, results, img_w, img_h):
    """Write a YOLO OBB .txt label file."""
    lines = []
    obb = results.obb

    has_xyxyxyxy = hasattr(obb, "xyxyxyxy") and obb.xyxyxyxy is not None

    for i in range(len(obb.conf)):
        conf  = float(obb.conf[i].cpu())
        cls   = int(obb.cls[i].cpu())
        if conf < LABEL_CONF:
            continue

        # Skip tiny boxes
        x_min, y_min, x_max, y_max = obb.xyxy[i].cpu()
        if (x_max - x_min) < MIN_BOX_PIXELS or (y_max - y_min) < MIN_BOX_PIXELS:
            continue

        if has_xyxyxyxy:
            coords = xyxy_to_yolo_obb(obb.xyxyxyxy[i], img_w, img_h)
        else:
            # Fallback: treat as axis-aligned rectangle
            x1, y1, x2, y2 = (
                float(x_min) / img_w, float(y_min) / img_h,
                float(x_max) / img_w, float(y_max) / img_h,
            )
            coords = [x1, y1, x2, y1, x2, y2, x1, y2]

        coord_str = " ".join(f"{v:.6f}" for v in coords)
        lines.append(f"{cls} {coord_str}")

    with open(label_path, "w") as f:
        f.write("\n".join(lines))

    return len(lines)   # number of labels written


# ── Pass 1: frame extraction ──────────────────────────────────────────────────
def extract_frames(video_dir, output_dir):
    img_dir = os.path.join(output_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    video_exts = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}
    videos     = [p for p in Path(video_dir).iterdir()
                  if p.suffix.lower() in video_exts]

    if not videos:
        raise FileNotFoundError(
            f"No video files found in '{video_dir}'. "
            "Check VIDEO_DIR or pass --video-dir."
        )

    print(f"\n{'═'*50}")
    print(f"  PASS 1 — Frame extraction")
    print(f"  Found {len(videos)} video(s) in '{video_dir}'")
    print(f"  Sampling every {FRAME_EVERY_N} frames")
    print(f"{'═'*50}\n")

    saved_total   = 0
    skipped_dup   = 0
    recent_hashes = []   # keep a rolling window to catch near-duplicates

    for vid_path in videos:
        cap       = cv2.VideoCapture(str(vid_path))
        total_f   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        vid_name  = vid_path.stem
        frame_idx = 0
        saved_vid = 0

        print(f"  ▶  {vid_path.name}  ({total_f} frames)")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % FRAME_EVERY_N == 0:
                h = dhash(frame)

                # Deduplication check
                is_dup = any(
                    hash_distance(h, prev) <= HASH_THRESHOLD
                    for prev in recent_hashes[-30:]   # compare against last 30 saved
                )

                if not is_dup:
                    fname = f"{vid_name}_f{frame_idx:06d}.jpg"
                    cv2.imwrite(os.path.join(img_dir, fname), frame)
                    saved_vid   += 1
                    saved_total += 1
                    recent_hashes.append(h)
                else:
                    skipped_dup += 1

            frame_idx += 1

        cap.release()
        print(f"     Saved {saved_vid} frames")

    print(f"\n  ✔  Total extracted : {saved_total} frames")
    print(f"     Skipped (duplicates): {skipped_dup}\n")
    return img_dir, saved_total


# ── Pass 2: auto-labelling ────────────────────────────────────────────────────
def auto_label(img_dir, output_dir):
    lbl_dir = os.path.join(output_dir, "labels")
    os.makedirs(lbl_dir, exist_ok=True)

    aug_img_dir = os.path.join(output_dir, "images")   # same folder; augmented images added here
    aug_lbl_dir = lbl_dir

    print(f"{'═'*50}")
    print(f"  PASS 2 — Auto-labelling  (YOLO: {YOLO_PATH})")
    print(f"  Confidence threshold : {LABEL_CONF}")
    print(f"{'═'*50}\n")

    model = YOLO(YOLO_PATH)

    images      = sorted(Path(img_dir).glob("*.jpg"))
    labelled    = 0
    total_boxes = 0
    empty       = 0

    for img_path in images:
        frame  = cv2.imread(str(img_path))
        h, w   = frame.shape[:2]
        results = model(frame, conf=LABEL_CONF, iou=LABEL_IOU, verbose=False)[0]

        lbl_path = os.path.join(lbl_dir, img_path.stem + ".txt")
        n_boxes  = write_label_file(lbl_path, results, w, h)

        if n_boxes == 0:
            empty += 1
        else:
            total_boxes += n_boxes
            labelled    += 1

        # ── Augmentation (only if the base frame has detections) ────────
        if n_boxes > 0 and AUGMENT_COPIES > 0:
            for suffix, aug_frame in augment_frame(frame):
                aug_h, aug_w = aug_frame.shape[:2]
                aug_img_name = img_path.stem + f"_aug_{suffix}.jpg"
                aug_img_path = os.path.join(aug_img_dir, aug_img_name)
                cv2.imwrite(aug_img_path, aug_frame)

                aug_results  = model(aug_frame, conf=LABEL_CONF,
                                     iou=LABEL_IOU, verbose=False)[0]
                aug_lbl_path = os.path.join(aug_lbl_dir,
                                            img_path.stem + f"_aug_{suffix}.txt")
                write_label_file(aug_lbl_path, aug_results, aug_w, aug_h)

    # ── Write dataset YAML ───────────────────────────────────────────────
    yaml_path = os.path.join(output_dir, "dataset.yaml")
    with open(yaml_path, "w") as f:
        f.write(
            f"# Auto-generated YOLO OBB dataset config\n"
            f"path: {os.path.abspath(output_dir)}\n"
            f"train: images\n"
            f"val: images    # split manually before training!\n\n"
            f"nc: 1\n"
            f"names: ['crystal']\n\n"
            f"# Task: obb\n"
        )

    print(f"\n  ✔  Labelled frames    : {labelled}")
    print(f"     Total boxes        : {total_boxes}")
    print(f"     Empty frames       : {empty}")
    print(f"     Dataset YAML       : {yaml_path}\n")
    print("  ⚠  IMPORTANT: Review labels in LabelImg / Roboflow before training!")
    print("     Remove or correct any wrong boxes.\n")


# ── Train-val split helper ────────────────────────────────────────────────────
def split_dataset(output_dir, val_fraction=0.15):
    """
    Moves a random fraction of images+labels into val/ subfolders.
    Run this AFTER reviewing labels.
    """
    import random, shutil
    img_dir = os.path.join(output_dir, "images")
    lbl_dir = os.path.join(output_dir, "labels")
    val_img = os.path.join(output_dir, "val_images")
    val_lbl = os.path.join(output_dir, "val_labels")
    os.makedirs(val_img, exist_ok=True)
    os.makedirs(val_lbl, exist_ok=True)

    images = list(Path(img_dir).glob("*.jpg"))
    random.shuffle(images)
    n_val  = max(1, int(len(images) * val_fraction))
    for img_p in images[:n_val]:
        lbl_p = Path(lbl_dir) / (img_p.stem + ".txt")
        shutil.move(str(img_p), os.path.join(val_img, img_p.name))
        if lbl_p.exists():
            shutil.move(str(lbl_p), os.path.join(val_lbl, lbl_p.name))

    print(f"  Split: {len(images)-n_val} train  /  {n_val} val")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract frames from videos and auto-label for YOLO OBB training"
    )
    parser.add_argument("--video-dir",   default=VIDEO_DIR)
    parser.add_argument("--output-dir",  default=OUTPUT_DIR)
    parser.add_argument("--frame-every", type=int, default=FRAME_EVERY_N,
                        help="Extract 1 frame every N frames")
    parser.add_argument("--split", action="store_true",
                        help="Also create train/val split after labelling")
    args = parser.parse_args()

    img_dir, n_frames = extract_frames(args.video_dir, args.output_dir)
    if n_frames > 0:
        auto_label(img_dir, args.output_dir)
        if args.split:
            split_dataset(args.output_dir)
    else:
        print("No frames extracted – nothing to label.")
