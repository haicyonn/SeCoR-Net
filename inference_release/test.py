import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import ChangeDataset
from metrics import metrics_from_confusion, update_confusion


def parse_args():
    parser = argparse.ArgumentParser(description="Inference/evaluation for the SeCoR inference-only release package.")
    parser.add_argument("--checkpoint", required=True, help="Path to TorchScript .pt checkpoint.")
    parser.add_argument("--data_root", required=True, help="Dataset root. Supports root/test/{A,B,label}.")
    parser.add_argument("--output_dir", default="outputs", help="Directory for predictions and metrics.")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="cuda", help="cuda or cpu.")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--data_pipeline", default="methodA", help="Test preprocessing preset: methodA or methodB.")
    parser.add_argument("--preserve_ab_order", action="store_true")
    parser.add_argument("--no_save", action="store_true", help="Do not save per-image prediction masks.")
    return parser.parse_args()


def load_model(checkpoint, device):
    model = torch.jit.load(str(checkpoint), map_location=device)
    model.to(device)
    model.eval()
    return model


def main():
    args = parse_args()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    pred_dir = out_dir / "pred_masks"
    prob_dir = out_dir / "prob_npy"
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_save:
        pred_dir.mkdir(parents=True, exist_ok=True)
        prob_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(args.checkpoint, device=device)
    dataset = ChangeDataset(
        args.data_root,
        split="test",
        size=args.size,
        preserve_ab_order=args.preserve_ab_order,
        pipeline=args.data_pipeline,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    with torch.no_grad():
        for batch_idx, (image, target, names) in enumerate(loader, start=1):
            image = image.to(device, non_blocking=True)
            pre = image[:, 0:3]
            post = image[:, 3:6]
            logits = model(pre, post)
            prob = torch.sigmoid(logits).detach().cpu().numpy()
            pred = (prob >= float(args.threshold)).astype(np.uint8)
            target_np = target.numpy().astype(np.uint8)

            for i, name in enumerate(names):
                pred_i = pred[i, 0]
                prob_i = prob[i, 0]
                gt_i = target_np[i]
                update_confusion(pred_i, gt_i, counts)
                if not args.no_save:
                    stem = Path(name).stem
                    cv2.imwrite(str(pred_dir / (stem + ".png")), pred_i * 255)
                    np.save(str(prob_dir / (stem + ".npy")), prob_i.astype(np.float32))

            if batch_idx % 20 == 0:
                print("[batch {}/{}]".format(batch_idx, len(loader)), flush=True)

    metrics = metrics_from_confusion(counts)
    result = {
        "checkpoint": str(args.checkpoint),
        "data_root": str(args.data_root),
        "threshold": float(args.threshold),
        "data_pipeline": args.data_pipeline,
        "metrics": metrics,
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    with open(out_dir / "metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)

    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
