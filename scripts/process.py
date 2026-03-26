#!/usr/bin/env python3
"""SAM3 text-prompt image segmentation example script.

Usage example:
python scripts/process.py \
  --image assets/images/test_image.jpg \
  --prompt shoe \
  --prompt person \
  --output-dir outputs
"""

import argparse
import json
import os
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image
import torch

import sam3
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SAM3 segmentation with text prompts on one image."
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input image.",
    )
    parser.add_argument(
        "--prompt",
        action="append",
        default=[],
        help="Text prompt to segment, can be repeated. Supports comma-separated values too.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/sam3_process",
        help="Directory to save results.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="assets/sam3.pt",
        help="Optional SAM3 checkpoint path. If omitted, default model loading is used.",
    )
    parser.add_argument(
        "--bpe-path",
        type=str,
        default="assets/bpe_simple_vocab_16e6.txt.gz",
        help="Optional tokenizer BPE path.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Inference device.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Confidence threshold used by Sam3Processor.",
    )
    return parser.parse_args()


def normalize_prompts(raw_prompts: List[str]) -> List[str]:
    prompts: List[str] = []
    for item in raw_prompts:
        for part in item.split(","):
            clean = part.strip()
            if clean:
                prompts.append(clean)
    return prompts


def sanitize_name(name: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name)
    return safe or "prompt"


def blend_mask_on_image(base_rgb: np.ndarray, mask: np.ndarray, color: np.ndarray) -> np.ndarray:
    out = base_rgb.copy()
    alpha = 0.45
    idx = mask.astype(bool)
    out[idx] = (1.0 - alpha) * out[idx] + alpha * color
    return out


def main() -> None:
    args = parse_args()
    prompts = normalize_prompts(args.prompt)
    if not prompts:
        raise ValueError("Please provide at least one prompt via --prompt")

    image_path = Path(args.image).expanduser().resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    out_root = Path(args.output_dir).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    sam3_root = Path(sam3.__file__).resolve().parent.parent
    checkpoint_path = args.checkpoint_path
    if checkpoint_path is not None:
        checkpoint_path = str(Path(checkpoint_path).expanduser().resolve())

    bpe_path = args.bpe_path
    if bpe_path is not None:
        bpe_path = str(Path(bpe_path).expanduser().resolve())

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_sam3_image_model(
        bpe_path=bpe_path,
        checkpoint_path=checkpoint_path,
        load_from_HF=checkpoint_path is None,
        device=device,
    )
    processor = Sam3Processor(model, confidence_threshold=args.threshold, device=device)

    image = Image.open(image_path).convert("RGB")
    base_np = np.asarray(image).astype(np.float32)

    state = processor.set_image(image)

    summary = {
        "image": str(image_path),
        "prompts": [],
    }

    palette = np.array(
        [
            [255, 87, 51],
            [67, 97, 238],
            [46, 196, 182],
            [255, 159, 28],
            [131, 56, 236],
            [42, 157, 143],
            [244, 114, 182],
            [14, 165, 233],
        ],
        dtype=np.float32,
    )

    for prompt in prompts:
        processor.reset_all_prompts(state)
        state = processor.set_text_prompt(prompt=prompt, state=state)

        prompt_safe = sanitize_name(prompt)
        prompt_dir = out_root / prompt_safe
        prompt_dir.mkdir(parents=True, exist_ok=True)

        masks = state["masks"].squeeze(1).detach().cpu().numpy().astype(np.uint8)
        boxes = state["boxes"].detach().cpu().numpy().tolist()
        scores = state["scores"].detach().cpu().numpy().tolist()

        overlay = base_np.copy()
        records = []

        for idx, mask in enumerate(masks):
            color = palette[idx % len(palette)]
            overlay = blend_mask_on_image(overlay, mask, color)

            mask_img = (mask * 255).astype(np.uint8)
            mask_path = prompt_dir / f"mask_{idx:03d}.png"
            Image.fromarray(mask_img, mode="L").save(mask_path)

            records.append(
                {
                    "id": idx,
                    "score": float(scores[idx]),
                    "box_xyxy": [float(v) for v in boxes[idx]],
                    "mask_path": str(mask_path),
                }
            )

        overlay_path = prompt_dir / "overlay.png"
        Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8)).save(overlay_path)

        prompt_json_path = prompt_dir / "result.json"
        prompt_json = {
            "prompt": prompt,
            "count": len(records),
            "results": records,
            "overlay_path": str(overlay_path),
        }
        with open(prompt_json_path, "w", encoding="utf-8") as f:
            json.dump(prompt_json, f, ensure_ascii=False, indent=2)

        summary["prompts"].append(
            {
                "prompt": prompt,
                "count": len(records),
                "result_json": str(prompt_json_path),
            }
        )

    summary_path = out_root / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Done. Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
