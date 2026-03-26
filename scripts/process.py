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
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image
import torch

import sam3
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


HUMAN_PROMPT_SET = [
    "person",
    "people",
    "man",
    "woman",
    "boy",
    "girl",
    "child",
    "adult",
]


def box_iou_xyxy(box1: List[float], box2: List[float]) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = area1 + area2 - inter
    if union <= 0:
        return 0.0
    return inter / union


def nms_indices(boxes: List[List[float]], scores: List[float], iou_threshold: float) -> List[int]:
    if not boxes:
        return []
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    keep: List[int] = []
    while order:
        cur = order.pop(0)
        keep.append(cur)
        remain: List[int] = []
        for i in order:
            if box_iou_xyxy(boxes[cur], boxes[i]) <= iou_threshold:
                remain.append(i)
        order = remain
    return keep


def expand_prompt(prompt: str, use_human_preset: bool) -> List[str]:
    prompt_clean = prompt.strip()
    prompt_lower = prompt_clean.lower()
    if use_human_preset and prompt_lower in {"human", "person", "people"}:
        return HUMAN_PROMPT_SET
    return [prompt_clean]


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
    parser.add_argument(
        "--human-preset",
        action="store_true",
        help="If prompt is human/person/people, expand to multiple human-related prompts and merge.",
    )
    parser.add_argument(
        "--nms-iou",
        type=float,
        default=0.5,
        help="NMS IoU threshold used when merging detections from expanded prompts.",
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
        candidate_prompts = expand_prompt(prompt, args.human_preset)

        prompt_safe = sanitize_name(prompt)
        prompt_dir = out_root / prompt_safe
        prompt_dir.mkdir(parents=True, exist_ok=True)

        merged_masks: List[np.ndarray] = []
        merged_boxes: List[List[float]] = []
        merged_scores: List[float] = []
        merged_sources: List[str] = []

        for cand_prompt in candidate_prompts:
            processor.reset_all_prompts(state)
            state = processor.set_text_prompt(prompt=cand_prompt, state=state)

            cand_masks = state["masks"].squeeze(1).detach().cpu().numpy().astype(np.uint8)
            cand_boxes = state["boxes"].detach().cpu().numpy().tolist()
            cand_scores = state["scores"].detach().cpu().numpy().tolist()

            for idx in range(len(cand_scores)):
                merged_masks.append(cand_masks[idx])
                merged_boxes.append([float(v) for v in cand_boxes[idx]])
                merged_scores.append(float(cand_scores[idx]))
                merged_sources.append(cand_prompt)

        keep = nms_indices(merged_boxes, merged_scores, args.nms_iou)
        keep_sorted = sorted(keep, key=lambda i: merged_scores[i], reverse=True)

        overlay = base_np.copy()
        records = []

        for out_idx, idx in enumerate(keep_sorted):
            mask = merged_masks[idx]
            color = palette[idx % len(palette)]
            overlay = blend_mask_on_image(overlay, mask, color)

            mask_img = (mask * 255).astype(np.uint8)
            mask_path = prompt_dir / f"mask_{out_idx:03d}.png"
            Image.fromarray(mask_img, mode="L").save(mask_path)

            records.append(
                {
                    "id": out_idx,
                    "score": float(merged_scores[idx]),
                    "box_xyxy": [float(v) for v in merged_boxes[idx]],
                    "matched_prompt": merged_sources[idx],
                    "mask_path": str(mask_path),
                }
            )

        overlay_path = prompt_dir / "overlay.png"
        Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8)).save(overlay_path)

        prompt_json_path = prompt_dir / "result.json"
        prompt_json = {
            "prompt": prompt,
            "candidate_prompts": candidate_prompts,
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
