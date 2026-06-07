"""
Swin-B single-model inference: produces unified-format predictions + probabilities.
Usage (stage 2):
  python swin_inference.py \
    --model /path/to/best_model_xxx.pth \
    --data_root /path/to/stage-2 \
    --output_dir /path/to/output \
    --threshold 0.5
Usage (stage 1, top-k):
  python swin_inference.py ... --topk 101
"""
import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import swin_b
from tqdm import tqdm
from datetime import datetime

IMG_SIZE = 384
BATCH_SIZE = 32


def get_model(model_path, device):
    model = swin_b(weights=None)
    model.head = nn.Linear(model.head.in_features, 2)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device)
    model.eval()
    return model


def get_test_loader(data_root, dataset_module_path):
    """Create test DataLoader. dataset_module_path points to the directory containing dataset.py."""
    sys.path.insert(0, dataset_module_path)
    from dataset import StoneDataset

    val_tf = transforms.Compose([
        transforms.Resize(int(IMG_SIZE * 1.14)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    test_dataset = StoneDataset(root=data_root, split="test", transforms=val_tf)
    return DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)


def run_inference(model, loader, device, threshold=None, topk=None):
    """
    Run inference, return (filenames, probabilities, labels).
    If topk is set: use rank cutoff (stage 1, known distribution).
    If threshold is set: use fixed threshold (stage 2, unknown distribution).
    """
    model.eval()
    probs_list, paths_list = [], []

    with torch.no_grad():
        for images, meta in tqdm(loader, desc="Inference"):
            images = images.to(device)
            with torch.amp.autocast("cuda"):
                probs = torch.softmax(model(images), dim=1)[:, 1].cpu().numpy()
            probs_list.extend(probs.tolist())
            paths_list.extend(meta)

    probs_arr = np.array(probs_list)
    filenames = [os.path.basename(p) for p in paths_list]

    if topk is not None:
        sorted_indices = np.argsort(probs_arr)[::-1]
        labels = np.zeros(len(probs_arr), dtype=int)
        labels[sorted_indices[:topk]] = 1
        actual_threshold = float(probs_arr[sorted_indices[topk - 1]]) if len(sorted_indices) > topk else 0.5
        method = f"topk_{topk}"
    elif threshold is not None:
        labels = (probs_arr >= threshold).astype(int)
        actual_threshold = threshold
        method = f"threshold_{threshold}"
    else:
        labels = (probs_arr >= 0.5).astype(int)
        actual_threshold = 0.5
        method = "threshold_0.5"

    return filenames, probs_arr.tolist(), labels.tolist(), actual_threshold, method


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to Swin-B .pth checkpoint")
    parser.add_argument("--data_root", required=True,
                        help="Path to dataset root (contains train_images/, test_images/, train_labels.csv)")
    parser.add_argument("--output_dir", required=True, help="Output directory for predictions")
    parser.add_argument("--model_name", default=None, help="Short name (default: from checkpoint filename)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Fixed threshold for binary prediction (stage 2)")
    parser.add_argument("--topk", type=int, default=None,
                        help="Top-k rank cutoff for positive predictions (stage 1)")
    args = parser.parse_args()

    if args.threshold is None and args.topk is None:
        args.threshold = 0.5  # default for stage 2

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model_name = args.model_name or os.path.splitext(os.path.basename(args.model))[0]

    # Determine dataset module path from data_root
    dataset_module_path = args.data_root

    print(f"Loading model: {args.model}")
    model = get_model(args.model, device)

    print(f"Loading test data from: {args.data_root}")
    loader = get_test_loader(args.data_root, dataset_module_path)

    print("Running inference...")
    filenames, probs, labels, actual_threshold, method = run_inference(
        model, loader, device, threshold=args.threshold, topk=args.topk
    )

    # Save predictions CSV
    df = pd.DataFrame({"id": filenames, "label": labels, "prob_m": probs})
    pred_path = os.path.join(args.output_dir, f"{model_name}_test_preds.csv")
    df.to_csv(pred_path, index=False)
    print(f"Predictions saved to: {pred_path}")

    # Summary stats
    stats = {
        "model": model_name,
        "checkpoint": args.model,
        "timestamp": datetime.now().isoformat(),
        "num_test": len(filenames),
        "pred_pos_count": int(sum(labels)),
        "prob_mean": float(np.mean(probs)),
        "prob_std": float(np.std(probs)),
        "prob_min": float(np.min(probs)),
        "prob_max": float(np.max(probs)),
        "prediction_method": method,
        "threshold_or_topk": actual_threshold,
    }
    stats_path = os.path.join(args.output_dir, f"{model_name}_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Stats saved to: {stats_path}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
