"""
Qwen3-VL inference with probability extraction from first-token logits.
Usage:
  python qwen_inference.py \
    --base_model /path/to/Qwen3-VL-8B-Instruct \
    --adapter /path/to/final_adapter \
    --data_root /path/to/data \
    --output_dir /path/to/output
"""
import os
import sys
import json
import argparse
import numpy as np
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
)
from peft import PeftModel


DEFAULT_INSTRUCTION = (
    "请观察这张图片，它有60%可能为非陨石（或不是石头），有40%概率为陨石。"
    "请你判断它是否为陨石？回答\"是\"或\"否\"。"
)


def load_model_and_processor(base_model_id, adapter_path, device="cuda"):
    """Load Qwen3-VL with LoRA adapter, return (model, processor)."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    base_model = AutoModelForImageTextToText.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    processor = AutoProcessor.from_pretrained(base_model_id, trust_remote_code=True)
    processor.tokenizer.padding_side = "left"

    return model, processor


def extract_meteor_probability(model, processor, image, instruction, device):
    """
    Extract meteorite probability from first-token logits.
    Returns probability of "是" (meteorite) in [0, 1].
    """
    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": None},
            {"type": "text", "text": instruction},
        ]},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=1,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
        )

    first_scores = generated.scores[0][0]  # logits for first token

    yes_id = processor.tokenizer.encode("是", add_special_tokens=False)[0]
    no_id = processor.tokenizer.encode("否", add_special_tokens=False)[0]

    yes_logit = first_scores[yes_id].item()
    no_logit = first_scores[no_id].item()

    # Binary softmax: p(是) = exp(yes) / (exp(yes) + exp(no))
    yes_prob = np.exp(yes_logit) / (np.exp(yes_logit) + np.exp(no_logit))
    return float(yes_prob)


class TestImageDataset(Dataset):
    """Simple dataset for test images, returns (filename, PIL.Image)."""

    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.image_files = []
        for f in sorted(os.listdir(image_dir)):
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                self.image_files.append(f)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        filename = self.image_files[idx]
        path = os.path.join(self.image_dir, filename)
        image = Image.open(path).convert("RGB")
        image.thumbnail((720, 720), Image.Resampling.LANCZOS)
        return filename, image


def run_inference(model, processor, image_dir, device, threshold=0.5, instruction=DEFAULT_INSTRUCTION):
    """
    Run inference on all test images, return (filenames, probabilities, labels).
    """
    dataset = TestImageDataset(image_dir)
    filenames_all = []
    probs_all = []

    for i in tqdm(range(len(dataset)), desc="Qwen inference"):
        filename, image = dataset[i]
        prob = extract_meteor_probability(model, processor, image, instruction, device)
        filenames_all.append(filename)
        probs_all.append(prob)

    probs_arr = np.array(probs_all)
    labels = (probs_arr >= threshold).astype(int)

    return filenames_all, probs_arr.tolist(), labels.tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True, help="Path to Qwen3-VL-8B-Instruct base model")
    parser.add_argument("--adapter", required=True, help="Path to LoRA adapter directory")
    parser.add_argument("--data_root", required=True, help="Path to dataset root (contains test_images/)")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--model_name", default=None, help="Short name (default: from adapter dirname)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for binary classification")
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION,
                        help="Prompt used for Qwen inference. Default matches test_model.py's 60:40 prior prompt.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model_name = args.model_name or os.path.basename(os.path.dirname(args.adapter))

    print(f"Loading model from: {args.base_model}")
    print(f"Loading adapter from: {args.adapter}")
    model, processor = load_model_and_processor(args.base_model, args.adapter, device)

    image_dir = os.path.join(args.data_root, "test_images")
    print(f"Running inference on: {image_dir}")
    print(f"Instruction: {args.instruction}")
    filenames, probs, labels = run_inference(
        model, processor, image_dir, device,
        threshold=args.threshold,
        instruction=args.instruction,
    )

    # Save predictions
    df = pd.DataFrame({"id": filenames, "label": labels, "prob_m": probs})
    pred_path = os.path.join(args.output_dir, f"{model_name}_test_preds.csv")
    df.to_csv(pred_path, index=False)
    print(f"Predictions saved to: {pred_path}")

    stats = {
        "model": model_name,
        "base_model": args.base_model,
        "adapter": args.adapter,
        "timestamp": datetime.now().isoformat(),
        "instruction": args.instruction,
        "num_test": len(filenames),
        "pred_pos_count": int(sum(labels)),
        "prob_mean": float(np.mean(probs)),
        "prob_std": float(np.std(probs)),
        "prob_min": float(np.min(probs)),
        "prob_max": float(np.max(probs)),
    }
    stats_path = os.path.join(args.output_dir, f"{model_name}_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Stats saved to: {stats_path}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
