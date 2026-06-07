"""
gen_submission_aug.py

使用增强训练模型对 test_images-v2 进行推理，输出两个版本：
1. 无 TTA：固定 720 尺寸，单次推理
2. TTA：5 种变换版本（3 尺度 × 翻转），概率平均

提示词与训练数据一致：先回答"是/否"，再给出理由。
"""

import os
import json
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel

# ================= 路径配置 =================
PROJECT_ROOT = "/root/Projects/meteorite-identification"
MODEL_ROOT = os.path.join(PROJECT_ROOT, "models")
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")

CHECKPOINT_PATH = os.path.join(
    MODEL_ROOT, "qwen3-vl-lora-meteorite_vc2_20260602_180603", "checkpoint-516")
BASE_MODEL_ID = os.path.join(MODEL_ROOT, "Qwen3-VL-8B-Instruct")
TEST_DIR = os.path.join(DATA_ROOT, "test_images-v2")
TEMPLATE_CSV = os.path.join(DATA_ROOT, "sample_submission_v2.csv")
OUTPUT_DIR = os.path.join(DATA_ROOT, "logs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================= 提示词（与训练数据一致）=================
INSTRUCTION = (
    "请观察这张图片中石头的视觉特征（颜色、纹理、形状），"
    "判断其中是否有陨石？注意：忽略图片中的所有文字、标签、标尺等信息，"
    "仅根据石头本身的外观特征判断。"
    "回答时首先给出\"是\"或\"否\"，然后简要说明理由。"
)

TTA_CONFIGS = [
    {"scale": 720, "flip_h": False, "label": "orig-720"},
    {"scale": 512, "flip_h": False, "label": "orig-512"},
    {"scale": 384, "flip_h": False, "label": "orig-384"},
    {"scale": 720, "flip_h": True,  "label": "flip-720"},
    {"scale": 512, "flip_h": True,  "label": "flip-512"},
]


# ================= Dataset =================
class TestImageDataset(Dataset):
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.image_files = sorted([
            f for f in os.listdir(image_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        filename = self.image_files[idx]
        path = os.path.join(self.image_dir, filename)
        image = Image.open(path).convert("RGB")
        return filename, image


def collate_fn(batch):
    return batch


def load_model():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    base_model = AutoModelForImageTextToText.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    model = PeftModel.from_pretrained(base_model, CHECKPOINT_PATH)
    model.eval()

    processor = AutoProcessor.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    processor.tokenizer.padding_side = 'left'

    return model, processor


def apply_transform(img, scale, flip_h):
    """应用尺度 + 翻转变换"""
    img_t = img.copy()
    if flip_h:
        img_t = img_t.transpose(Image.FLIP_LEFT_RIGHT)
    img_t.thumbnail((scale, scale), Image.Resampling.LANCZOS)
    return img_t


# ================= 无 TTA 推理 =================
def predict_no_tta(model, processor, test_loader, device):
    """generate 后提取首字符答案（"是"/"否"）"""
    model.eval()
    id_to_pred = {}

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predict (no-TTA)"):
            filenames, images = zip(*batch)
            scaled = [apply_transform(img, 720, False) for img in images]

            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": None},
                    {"type": "text", "text": INSTRUCTION}
                ]
            }]
            texts = [processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
                for _ in range(len(scaled))]

            inputs = processor(text=texts, images=scaled,
                               return_tensors="pt", padding=True).to(device)

            generated = model.generate(**inputs, max_new_tokens=256, do_sample=False)

            for fn, gen_id in zip(filenames, generated):
                gen_text = processor.decode(
                    gen_id[inputs['input_ids'].shape[1]:], skip_special_tokens=True)
                # 答案在前格式："是。理由：..." / "否。理由：..."
                if gen_text.startswith("是"):
                    id_to_pred[fn] = 1
                elif gen_text.startswith("否"):
                    id_to_pred[fn] = 0
                elif "因此判断为：是" in gen_text:
                    id_to_pred[fn] = 1
                elif "因此判断为：否" in gen_text:
                    id_to_pred[fn] = 0
                else:
                    id_to_pred[fn] = 1 if "是" in gen_text else 0

    return id_to_pred


# ================= TTA 推理 =================
def predict_with_tta(model, processor, test_loader, device):
    """
    对每种 TTA 配置分别推理，取 "是"/"否" token 的 logits 差值平均。
    """
    model.eval()

    yes_token_id = processor.tokenizer.encode("是", add_special_tokens=False)[0]
    no_token_id  = processor.tokenizer.encode("否", add_special_tokens=False)[0]

    filename_scores = {}
    n_configs = len(TTA_CONFIGS)

    with torch.no_grad():
        for config in TTA_CONFIGS:
            desc = f"TTA {config['label']}"
            for batch in tqdm(test_loader, desc=desc, leave=False):
                filenames, images = zip(*batch)
                scaled = [apply_transform(img, config["scale"], config["flip_h"])
                          for img in images]

                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "image", "image": None},
                        {"type": "text", "text": INSTRUCTION}
                    ]
                }]
                texts = [processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True)
                    for _ in range(len(scaled))]

                inputs = processor(text=texts, images=scaled,
                                   return_tensors="pt", padding=True).to(device)

                outputs = model(**inputs)
                logits = outputs.logits  # [B, seq, vocab]
                input_len = inputs["input_ids"].shape[1]

                for i, fn in enumerate(filenames):
                    # logits[t] 预测 token[t+1]，所以 input_len-1 预测第一个生成 token
                    first_gen_logits = logits[i, input_len - 1, :]
                    yes_logit = first_gen_logits[yes_token_id].item()
                    no_logit  = first_gen_logits[no_token_id].item()

                    if fn not in filename_scores:
                        filename_scores[fn] = 0.0
                    filename_scores[fn] += (yes_logit - no_logit)

                del inputs, outputs, logits
                torch.cuda.empty_cache()

    id_to_pred = {}
    for fn, score in filename_scores.items():
        id_to_pred[fn] = 1 if score / n_configs > 0 else 0

    return id_to_pred


def make_submission(id_to_pred, output_path):
    ids_df = pd.read_csv(TEMPLATE_CSV)
    sub = ids_df.copy()
    sub["label"] = sub["id"].map(id_to_pred)
    missing = sub["label"].isna().sum()
    if missing > 0:
        print(f"⚠️  {missing} 条缺失，填 0")
        sub["label"] = sub["label"].fillna(0).astype(int)
    else:
        sub["label"] = sub["label"].astype(int)
    sub.to_csv(output_path, index=False)
    return sub


def main():
    print("=" * 60)
    print(f"增强模型推理：{os.path.basename(CHECKPOINT_PATH)}")
    print(f"  测试集: {TEST_DIR}")
    print(f"  提示词与训练一致（答案在前，理由在后）")
    print("=" * 60)

    model, processor = load_model()
    device = next(model.parameters()).device

    dataset = TestImageDataset(TEST_DIR)
    print(f"\n测试图像: {len(dataset)}")

    test_loader = DataLoader(
        dataset, batch_size=4, shuffle=False,
        num_workers=2, collate_fn=collate_fn)

    # ===== 1. No TTA =====
    print("\n[1/2] 无 TTA 推理...")
    pred_no = predict_no_tta(model, processor, test_loader, device)
    csv_no = os.path.join(OUTPUT_DIR, "submission_aug_no_tta.csv")
    sub_no = make_submission(pred_no, csv_no)
    yes_c = (sub_no["label"] == 1).sum()
    no_c  = (sub_no["label"] == 0).sum()
    print(f"  ✅ 无 TTA: {csv_no}")
    print(f"     是: {yes_c} ({yes_c/len(sub_no)*100:.1f}%) | 否: {no_c} ({no_c/len(sub_no)*100:.1f}%)")

    # ===== 2. With TTA =====
    print("\n[2/2] TTA 推理 (5 transforms)...")
    pred_tta = predict_with_tta(model, processor, test_loader, device)
    csv_tta = os.path.join(OUTPUT_DIR, "submission_aug_tta.csv")
    sub_tta = make_submission(pred_tta, csv_tta)
    yes_c = (sub_tta["label"] == 1).sum()
    no_c  = (sub_tta["label"] == 0).sum()
    print(f"  ✅ TTA: {csv_tta}")
    print(f"     是: {yes_c} ({yes_c/len(sub_tta)*100:.1f}%) | 否: {no_c} ({no_c/len(sub_tta)*100:.1f}%)")

    # ===== 对比 =====
    diff = (sub_no["label"] != sub_tta["label"]).sum()
    print(f"\n  差异样本数: {diff} / {len(sub_no)}")

    if diff > 0:
        print(f"  差异样本（无TTA vs TTA）:")
        for i in range(min(10, diff)):
            row = sub_no[sub_no["label"] != sub_tta["label"]].iloc[i]
            tta_label = sub_tta[sub_tta["id"] == row["id"]]["label"].values[0]
            print(f"    {row['id']}: no_tta={row['label']} vs tta={tta_label}")

    print("\n" + "=" * 60)
    print("✅ 完成!")


if __name__ == "__main__":
    main()
