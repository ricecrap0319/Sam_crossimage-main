#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
import matplotlib.pyplot as plt
from importlib.resources import files
from PIL import Image

import sam3
from sam3 import build_sam3_image_model
from sam3.model.box_ops import box_xywh_to_cxcywh
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualization_utils import normalize_bbox, plot_results, plot_bbox


def main():

    print("\n=== SAM3 Cross Image Prompt Tool ===\n")

    img1 = input("Path to reference image (image 1): ").strip()
    img2 = input("Path to target image (image 2): ").strip()

    print("\nEnter bounding box for image 1")
    x = float(input("x (top-left): "))
    y = float(input("y (top-left): "))
    w = float(input("width: "))
    h = float(input("height: "))

    threshold = input("Confidence threshold (default 0.5): ").strip()
    threshold = float(threshold) if threshold else 0.5

    output1 = input("Output path for image1 result (default image1_segmentation.png): ").strip()
    output2 = input("Output path for image2 result (default image2_segmentation.png): ").strip()

    if not output1:
        output1 = "image1_segmentation.png"
    if not output2:
        output2 = "image2_segmentation.png"

    box = [x, y, w, h]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda":
        autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    else:
        from contextlib import nullcontext
        autocast = nullcontext()

    with autocast:

        bpe_path = str(files("sam3").joinpath("assets/bpe_simple_vocab_16e6.txt.gz"))

        model = build_sam3_image_model(bpe_path=bpe_path)
        processor = Sam3Processor(model, confidence_threshold=threshold)

        image1 = Image.open(img1)
        image2 = Image.open(img2)

        w1, h1 = image1.size

        print("\nExtracting prompt from image 1...")

        state1 = processor.set_image(image1)

        box_xywh = torch.tensor(box).view(-1, 4)
        box_cxcywh = box_xywh_to_cxcywh(box_xywh)

        norm_box = normalize_bbox(box_cxcywh, w1, h1).flatten().tolist()

        state1 = processor._add_box_prompt(box=norm_box, label=True, state=state1)

        state1_inference = processor._forward_grounding(state1.copy())

        plt.figure(figsize=(10, 10))
        plot_results(image1, state1_inference)
        plot_bbox(h1, w1, box, box_format="XYWH", color="yellow",
                  linestyle="dashed", text="PROMPT", relative_coords=False)
        plt.axis("off")
        plt.savefig(output1, bbox_inches='tight', dpi=150)
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
        plt.axis("off")
        plt.savefig(output2, bbox_inches='tight', dpi=150)
        plt.close()

    print("\nDone!")
    print("Image1 result:", output1)
    print("Image2 result:", output2)


if __name__ == "__main__":
    main()