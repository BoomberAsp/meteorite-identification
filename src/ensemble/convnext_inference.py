"""
ConvNeXt-Large single-model inference: produces unified-format predictions.
Usage (stage 2):
  python convnext_inference.py \
    --model /path/to/best_model.pth \
    --data_root /path/to/stage-2 \
    --output_dir /path/to/output \
    --threshold 0.5
"""
import os
import re
import sys
import json
import argparse
import numpy as np
import torch
import pandas as pd
import timm
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from datetime import datetime

IMG_SIZE = 224
BATCH_SIZE = 32


def remap_old_timm_keys(state_dict):
    """Remap old-style timm ConvNeXt keys to current timm format."""

    def _remap(key):
        key = key.replace("convnext.embeddings.patch_embeddings.", "stem.0.")
        key = key.replace("convnext.embeddings.layernorm.", "stem.1.")
        key = re.sub(
            r"convnext\.encoder\.stages\.(\d)\.layers\.(\d+)\.layer_scale_parameter",
            r"stages.\1.blocks.\2.gamma", key)
        key = re.sub(
            r"convnext\.encoder\.stages\.(\d)\.layers\.(\d+)\.dwconv",
            r"stages.\1.blocks.\2.conv_dw", key)
        key = re.sub(
            r"convnext\.encoder\.stages\.(\d)\.layers\.(\d+)\.layernorm",
            r"stages.\1.blocks.\2.norm", key)
        key = re.sub(
            r"convnext\.encoder\.stages\.(\d)\.layers\.(\d+)\.pwconv1",
            r"stages.\1.blocks.\2.mlp.fc1", key)
        key = re.sub(
            r"convnext\.encoder\.stages\.(\d)\.layers\.(\d+)\.pwconv2",
            r"stages.\1.blocks.\2.mlp.fc2", key)
        key = re.sub(
            r"convnext\.encoder\.stages\.(\d)\.downsampling_layer\.(\d)",
            r"stages.\1.downsample.\2", key)
        key = key.replace("convnext.layernorm.", "head.norm.")
        key = key.replace("classifier.", "head.fc.")
        return key

    return {_remap(k): v for k, v in state_dict.items()}


def get_model(model_path, device):
    model = timm.create_model("convnext_large", pretrained=False, num_classes=2)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    checkpoint = remap_old_timm_keys(checkpoint)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    return model


def get_test_loader(data_root, dataset_module_path):
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
    parser.add_argument("--model", required=True, help="Path to ConvNeXt .pth checkpoint")
    parser.add_argument("--data_root", required=True,
                        help="Dataset root (contains train_images/, test_images/, train_labels.csv)")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--model_name", default=None, help="Short name (default: from checkpoint filename)")
    parser.add_argument("--threshold", type=float, default=None, help="Fixed threshold (stage 2)")
    parser.add_argument("--topk", type=int, default=None, help="Top-k cutoff (stage 1)")
    args = parser.parse_args()

    if args.threshold is None and args.topk is None:
        args.threshold = 0.5

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model_name = args.model_name or os.path.splitext(os.path.basename(args.model))[0]

    print(f"Loading model: {args.model}")
    model = get_model(args.model, device)

    print(f"Loading test data from: {args.data_root}")
    loader = get_test_loader(args.data_root, args.data_root)

    print("Running inference...")
    filenames, probs, labels, actual_threshold, method = run_inference(
        model, loader, device, threshold=args.threshold, topk=args.topk
    )

    df = pd.DataFrame({"id": filenames, "label": labels, "prob_m": probs})
    pred_path = os.path.join(args.output_dir, f"{model_name}_test_preds.csv")
    df.to_csv(pred_path, index=False)
    print(f"Predictions saved to: {pred_path}")

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
