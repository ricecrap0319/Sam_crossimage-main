
import sys
import os
from contextlib import nullcontext
from importlib.resources import files

import cv2
import matplotlib.pyplot as plt
import torch
from PIL import Image


sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import sam3
from sam3 import build_sam3_image_model
from sam3.model.box_ops import box_xywh_to_cxcywh
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualization_utils import normalize_bbox, plot_results, plot_bbox


def clean_path(p: str) -> str:
    return p.strip().strip('"').strip("'")


def target_present(state, score_threshold=0.5, mask_pixel_threshold=50):
    """
    Decide whether the target is present in this frame.

    This is written defensively because SAM3 output structure may vary.
    It tries score-based detection first, then falls back to mask size.
    """

    
    possible_score_keys = ["scores", "pred_scores", "confidence", "confidences"]

    for key in possible_score_keys:
        if key in state and state[key] is not None:
            scores = state[key]

          
            if hasattr(scores, "numel"):
                if scores.numel() == 0:
                    return False
                try:
                    return float(scores.max().item()) >= score_threshold
                except Exception:
                    pass

            # python list / tuple
            if isinstance(scores, (list, tuple)):
                if len(scores) == 0:
                    return False
                try:
                    return max(float(s) for s in scores) >= score_threshold
                except Exception:
                    pass

            # scalar
            try:
                return float(scores) >= score_threshold
            except Exception:
                pass

    # Fallback: check masks
    if "masks" in state and state["masks"] is not None:
        masks = state["masks"]

        # torch tensor masks
        if hasattr(masks, "numel"):
            if masks.numel() == 0:
                return False
            try:
                return float((masks > 0).sum().item()) >= mask_pixel_threshold
            except Exception:
                try:
                    return float(masks.sum().item()) >= mask_pixel_threshold
                except Exception:
                    pass

        # list-like masks
        try:
            return len(masks) > 0
        except Exception:
            pass

    return False


def save_overlay(image_pil, state, out_path, title_text):
    plt.figure(figsize=(10, 10))
    plot_results(image_pil, state)
    plt.title(title_text)
    plt.axis("off")
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()


def main():
    print("\n=== SAM3 Reference-to-Video Event Frame Saver ===\n")

    # -------- User input --------
    img1 = clean_path(input("Path to reference image: "))
    video_path = clean_path(input("Path to target video: "))

    print("\nEnter bounding box for reference image")
    x = float(input("x (top-left): "))
    y = float(input("y (top-left): "))
    w = float(input("width: "))
    h = float(input("height: "))

    threshold_in = input("Confidence threshold (default 0.5): ").strip()
    threshold = float(threshold_in) if threshold_in else 0.5

    cooldown_in = input(
        "Minimum frame gap between saved events (default 30): "
    ).strip()
    cooldown_frames = int(cooldown_in) if cooldown_in else 30

    absence_in = input(
        "Frames of absence needed before a new event (default 5): "
    ).strip()
    absence_frames_required = int(absence_in) if absence_in else 5

    default_output_dir = r"C:\Users\mingx\Downloads\Sam_crossimage-main\output"
    output_dir = clean_path(
        input(f"Output folder (default {default_output_dir}): ").strip()
    )
    if not output_dir:
        output_dir = default_output_dir
    os.makedirs(output_dir, exist_ok=True)

    save_overlay_choice = input(
        "Also save overlay images? (y/n, default y): "
    ).strip().lower()
    save_overlay_images = save_overlay_choice != "n"

    # -------- Output folders --------
    event_frames_dir = os.path.join(output_dir, "event_frames")
    os.makedirs(event_frames_dir, exist_ok=True)

    overlay_dir = os.path.join(output_dir, "event_overlays")
    if save_overlay_images:
        os.makedirs(overlay_dir, exist_ok=True)

    reference_vis_path = os.path.join(output_dir, "reference_prompt_result.png")

    # -------- Device setup --------
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"\nUsing device: {device}\n")

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    else:
        autocast_ctx = nullcontext()

    with autocast_ctx:
        # -------- Build model --------
        bpe_path = str(files("sam3").joinpath("assets/bpe_simple_vocab_16e6.txt.gz"))

        # Assumes sam3.pt is in the project root
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

        # -------- Load reference image --------
        image1 = Image.open(img1).convert("RGB")
        w1, h1 = image1.size
        box = [x, y, w, h]

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
        plt.title(f"Prompt on Reference Image: {os.path.basename(img1)}")
        plt.axis("off")
        plt.savefig(reference_vis_path, bbox_inches="tight", dpi=150)
        plt.close()

        saved_prompt = state1["prompt"]
        saved_prompt_mask = state1["prompt_mask"]

        # -------- Open video --------
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video:\n{video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Video FPS: {fps}")
        print(f"Total frames: {frame_count}\n")

        # -------- Event logic state --------
        frame_idx = 0
        event_count = 0

        target_was_present = False
        absence_counter = absence_frames_required
        last_saved_frame = -10**9

        print("Processing video...\n")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            print(f"Processing frame {frame_idx}/{frame_count}", end="\r")

            # Convert frame to PIL
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image2 = Image.fromarray(frame_rgb).convert("RGB")

            # Apply saved prompt
            state2 = processor.set_image(image2)
            state2["prompt"] = saved_prompt
            state2["prompt_mask"] = saved_prompt_mask
            state2 = processor._forward_grounding(state2)

            present_now = target_present(state2, score_threshold=threshold)

            if present_now:
                # New event only if:
                # 1) target was absent before long enough
                # 2) enough cooldown since last save
                is_new_event = (
                    (not target_was_present)
                    and (absence_counter >= absence_frames_required)
                    and ((frame_idx - last_saved_frame) >= cooldown_frames)
                )

                if is_new_event:
                    event_count += 1
                    last_saved_frame = frame_idx

                    frame_filename = (
                        f"event_{event_count:04d}_frame_{frame_idx:06d}.png"
                    )
                    frame_path = os.path.join(event_frames_dir, frame_filename)

                    # Save original frame
                    cv2.imwrite(frame_path, frame)

                    print(
                        f"\nSaved event frame {event_count} at frame {frame_idx}:"
                        f"\n{frame_path}"
                    )

                    if save_overlay_images:
                        overlay_filename = (
                            f"event_{event_count:04d}_frame_{frame_idx:06d}_overlay.png"
                        )
                        overlay_path = os.path.join(overlay_dir, overlay_filename)
                        save_overlay(
                            image2,
                            state2,
                            overlay_path,
                            f"Detected target - event {event_count}, frame {frame_idx}",
                        )

                target_was_present = True
                absence_counter = 0

            else:
                target_was_present = False
                absence_counter += 1

        cap.release()

    print("\n\nDone!")
    print(f"Reference visualization: {reference_vis_path}")
    print(f"Saved event frames folder: {event_frames_dir}")
    if save_overlay_images:
        print(f"Saved overlay folder: {overlay_dir}")
    print(f"Total events saved: {event_count}")


if __name__ == "__main__":
    main()