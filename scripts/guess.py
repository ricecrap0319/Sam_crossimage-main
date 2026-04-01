#!/usr/bin/env python3
import sys
import os
from contextlib import nullcontext
from importlib.resources import files

import matplotlib.pyplot as plt
import torch
from PIL import Image

# Make sure the project root is on sys.path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import sam3
from sam3 import build_sam3_image_model
from sam3.model.box_ops import box_xywh_to_cxcywh
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualization_utils import normalize_bbox, plot_results, plot_bbox


def clean_path(p: str) -> str:
    return p.strip().strip('"').strip("'")


def is_image_file(filename):
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    return os.path.splitext(filename.lower())[1] in valid_exts


def get_all_detected_boxes(state, image_width, image_height, score_threshold=0.5):
    """
    Read all detected boxes from SAM3 output state.

    Returns:
        A list like:
        [
            {
                "box_xyxy": [x1, y1, x2, y2],
                "box_xywh": [x, y, w, h],
                "box_yolo": [x_center, y_center, width, height],
                "score": 0.91
            },
            ...
        ]
    """
    def to_list(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().tolist()
        return x

    boxes = None
    for key in ["boxes", "pred_boxes", "bboxes"]:
        if key in state and state[key] is not None:
            boxes = to_list(state[key])
            break

    if boxes is None:
        return []

    if len(boxes) == 4 and not isinstance(boxes[0], (list, tuple)):
        boxes = [boxes]

    scores = None
    for key in ["scores", "pred_scores", "confidence", "confidences"]:
        if key in state and state[key] is not None:
            scores = to_list(state[key])
            break

    if scores is None:
        scores = [None] * len(boxes)
    elif not isinstance(scores, (list, tuple)):
        scores = [scores]

    if len(scores) < len(boxes):
        scores = list(scores) + [None] * (len(boxes) - len(scores))

    results = []

    for box, score in zip(boxes, scores):
        if box is None or len(box) != 4:
            continue

        score_val = float(score) if score is not None else 1.0
        if score_val < score_threshold:
            continue

        x1, y1, x2, y2 = [float(v) for v in box]

        # Heuristic: if box looks like cx, cy, w, h
        if x2 < x1 or y2 < y1:
            cx, cy, w, h = x1, y1, x2, y2
            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2

        # If coordinates look normalized, convert to pixels
        if 0.0 <= x1 <= 1.0 and 0.0 <= x2 <= 1.0:
            x1 *= image_width
            x2 *= image_width
        if 0.0 <= y1 <= 1.0 and 0.0 <= y2 <= 1.0:
            y1 *= image_height
            y2 *= image_height

        # Clip to image bounds
        x1 = max(0.0, min(float(image_width), x1))
        x2 = max(0.0, min(float(image_width), x2))
        y1 = max(0.0, min(float(image_height), y1))
        y2 = max(0.0, min(float(image_height), y2))

        if x2 <= x1 or y2 <= y1:
            continue

        w = x2 - x1
        h = y2 - y1

        # YOLO normalized format
        x_center = x1 + w / 2.0
        y_center = y1 + h / 2.0

        x_center_norm = x_center / image_width
        y_center_norm = y_center / image_height
        w_norm = w / image_width
        h_norm = h / image_height

        results.append(
            {
                "box_xyxy": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                "box_xywh": [round(x1, 2), round(y1, 2), round(w, 2), round(h, 2)],
                "box_yolo": [
                    round(x_center_norm, 6),
                    round(y_center_norm, 6),
                    round(w_norm, 6),
                    round(h_norm, 6),
                ],
                "score": round(score_val, 4),
            }
        )

    return results


def save_coordinate_txt(txt_path, image_index, image_name, detections, class_id=0):
    """
    Save coordinate info to a text file.
    Includes image number so it corresponds to the image processed.
    """
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"image_index: {image_index}\n")
        f.write(f"image_name: {image_name}\n")
        f.write(f"detection_count: {len(detections)}\n\n")

        if not detections:
            f.write("No detections above threshold.\n")
            return

        for i, det in enumerate(detections, start=1):
            x1, y1, x2, y2 = det["box_xyxy"]
            x, y, w, h = det["box_xywh"]
            xc, yc, wn, hn = det["box_yolo"]
            score = det["score"]

            f.write(f"detection_{i}\n")
            f.write(f"class_id: {class_id}\n")
            f.write(f"score: {score}\n")
            f.write(f"xyxy: {x1}, {y1}, {x2}, {y2}\n")
            f.write(f"xywh: {x}, {y}, {w}, {h}\n")
            f.write(f"yolo: {class_id} {xc} {yc} {wn} {hn}\n")
            f.write("\n")


def main():
    print("\n=== SAM3 Cross Image Prompt Tool (Folder -> 2 files per image) ===\n")

    img1 = clean_path(input("Path to reference image: "))
    target_folder = clean_path(input("Path to folder of target images: "))

    print("\nEnter bounding box for reference image")
    x = float(input("x (top-left): "))
    y = float(input("y (top-left): "))
    w = float(input("width: "))
    h = float(input("height: "))

    threshold_in = input("Confidence threshold (default 0.5): ").strip()
    threshold = float(threshold_in) if threshold_in else 0.5

    class_id_in = input("Class id for output (default 0): ").strip()
    class_id = int(class_id_in) if class_id_in else 0

    default_output_dir = r"C:\Users\mingx\Downloads\Sam_crossimage-main\output"
    output_dir = clean_path(
        input(f"Output folder (default {default_output_dir}): ").strip()
    )
    if not output_dir:
        output_dir = default_output_dir
    os.makedirs(output_dir, exist_ok=True)

    reference_vis_path = os.path.join(output_dir, "reference_prompt_result.png")

    if not os.path.isdir(target_folder):
        raise NotADirectoryError(f"Target folder not found:\n{target_folder}")

    target_files = sorted(
        f for f in os.listdir(target_folder)
        if os.path.isfile(os.path.join(target_folder, f)) and is_image_file(f)
    )

    if not target_files:
        raise FileNotFoundError(f"No image files found in folder:\n{target_folder}")

    print(f"\nFound {len(target_files)} target image(s).\n")

    box = [x, y, w, h]

    # Device setup
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}\n")

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    else:
        autocast_ctx = nullcontext()

    with autocast_ctx:
        bpe_path = str(files("sam3").joinpath("assets/bpe_simple_vocab_16e6.txt.gz"))

        # Assumes sam3.pt is in project root
        project_root = os.path.dirname(os.path.dirname(__file__))
        checkpoint_path = os.path.join(project_root, "sam3.pt")

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Could not find local checkpoint at:\n{checkpoint_path}\n"
                "Put sam3.pt in the project root folder or edit checkpoint_path."
            )

        print("Building model from local checkpoint...")
        model = build_sam3_image_model(
            bpe_path=bpe_path,
            checkpoint_path=checkpoint_path,
            load_from_HF=False,
            device=str(device),
            eval_mode=True,
        )

        processor = Sam3Processor(model, confidence_threshold=threshold)

        # ---- reference image ----
        image1 = Image.open(img1).convert("RGB")
        w1, h1 = image1.size

        print("Extracting prompt from reference image...")

        state1 = processor.set_image(image1)

        box_xywh = torch.tensor(box, dtype=torch.float32).view(-1, 4)
        box_cxcywh = box_xywh_to_cxcywh(box_xywh)
        norm_box = normalize_bbox(box_cxcywh, w1, h1).flatten().tolist()

        state1 = processor._add_box_prompt(box=norm_box, label=True, state=state1)
        state1_inference = processor._forward_grounding(state1.copy())

        plt.figure(figsize=(10, 10))
        plot_results(image1, state1_inference)
        plot_bbox(
            h1,
            w1,
            box,
            box_format="XYWH",
            color="yellow",
            linestyle="dashed",
            text="PROMPT",
            relative_coords=False,
        )
        plt.title(f"Prompt on Source Image: {os.path.basename(img1)}")
        plt.axis("off")
        plt.savefig(reference_vis_path, bbox_inches="tight", dpi=150)
        plt.close()

        saved_prompt = state1["prompt"]
        saved_prompt_mask = state1["prompt_mask"]

        # ---- process folder ----
        total_with_detections = 0

        for idx, filename in enumerate(target_files, start=1):
            img2_path = os.path.join(target_folder, filename)
            stem, ext = os.path.splitext(filename)

            print(f"[{idx}/{len(target_files)}] Processing: {filename}")

            image2 = Image.open(img2_path).convert("RGB")
            w2, h2 = image2.size

            state2 = processor.set_image(image2)
            state2["prompt"] = saved_prompt
            state2["prompt_mask"] = saved_prompt_mask
            state2 = processor._forward_grounding(state2)

            detections = get_all_detected_boxes(
                state2,
                image_width=w2,
                image_height=h2,
                score_threshold=threshold,
            )

            if detections:
                total_with_detections += 1

            indexed_stem = f"{idx:04d}_{stem}"

            # File 1: image with boxes
            boxed_image_path = os.path.join(output_dir, f"{indexed_stem}{ext}")

            # File 2: coordinate text
            txt_path = os.path.join(output_dir, f"{indexed_stem}.txt")

            # Save boxed image
            plt.figure(figsize=(10, 10))
            plot_results(image2, state2)

            for det in detections:
                x1, y1, x2, y2 = det["box_xyxy"]
                plt.gca().add_patch(
                    plt.Rectangle(
                        (x1, y1),
                        x2 - x1,
                        y2 - y1,
                        edgecolor="red",
                        linewidth=2,
                        fill=False,
                    )
                )

            plt.title(f"Image {idx:04d}: {filename}")
            plt.axis("off")
            plt.savefig(boxed_image_path, bbox_inches="tight", dpi=150)
            plt.close()

            # Save coordinate file
            save_coordinate_txt(
                txt_path=txt_path,
                image_index=idx,
                image_name=filename,
                detections=detections,
                class_id=class_id,
            )

            print(f"  Detections: {len(detections)}")
            print(f"  Boxed image: {boxed_image_path}")
            print(f"  Coordinate txt: {txt_path}\n")

    print("Done!")
    print(f"Reference visualization: {reference_vis_path}")
    print(f"Images processed: {len(target_files)}")
    print(f"Images with detections: {total_with_detections}")
    print(f"Main output folder: {output_dir}")


if __name__ == "__main__":
    main()