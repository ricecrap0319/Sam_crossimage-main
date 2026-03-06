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


def main():
    print("\n=== SAM3 Cross Image Prompt Tool ===\n")

    img1 = clean_path(input("Path to reference image (image 1): "))
    img2 = clean_path(input("Path to target image (image 2): "))

    print("\nEnter bounding box for image 1")
    x = float(input("x (top-left): "))
    y = float(input("y (top-left): "))
    w = float(input("width: "))
    h = float(input("height: "))

    threshold_in = input("Confidence threshold (default 0.5): ").strip()
    threshold = float(threshold_in) if threshold_in else 0.5

    # Change this if you want a different default folder
    default_output_dir = r"C:\Users\mingx\Downloads\Sam_crossimage-main\output"
    output_dir = clean_path(
        input(f"Output folder (default {default_output_dir}): ").strip()
    )
    if not output_dir:
        output_dir = default_output_dir
    os.makedirs(output_dir, exist_ok=True)

    output1 = clean_path(
        input("Output filename for image 1 result (default image1_segmentation.png): ")
    )
    output2 = clean_path(
        input("Output filename for image 2 result (default image2_segmentation.png): ")
    )

    if not output1:
        output1 = "image1_segmentation.png"
    if not output2:
        output2 = "image2_segmentation.png"

    output1 = os.path.join(output_dir, output1)
    output2 = os.path.join(output_dir, output2)

    box = [x, y, w, h]

    # Device setup
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
        bpe_path = str(files("sam3").joinpath("assets/bpe_simple_vocab_16e6.txt.gz"))

        # Assumes sam3.pt is in the project root:
        # C:\Users\mingx\Downloads\Sam_crossimage-main\Sam_crossimage-main\sam3.pt
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

        image1 = Image.open(img1).convert("RGB")
        image2 = Image.open(img2).convert("RGB")

        w1, h1 = image1.size

        print("Extracting prompt from image 1...")

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
        plt.savefig(output1, bbox_inches="tight", dpi=150)
        plt.close()

        saved_prompt = state1["prompt"]
        saved_prompt_mask = state1["prompt_mask"]

        print("Applying prompt to second image...")

        state2 = processor.set_image(image2)
        state2["prompt"] = saved_prompt
        state2["prompt_mask"] = saved_prompt_mask
        state2 = processor._forward_grounding(state2)

        plt.figure(figsize=(10, 10))
        plot_results(image2, state2)
        plt.title(
            f"Reused Prompt from {os.path.basename(img1)} on {os.path.basename(img2)}"
        )
        plt.axis("off")
        plt.savefig(output2, bbox_inches="tight", dpi=150)
        plt.close()

    print("\nDone!")
    print("Image 1 result:", output1)
    print("Image 2 result:", output2)


if __name__ == "__main__":
    main()