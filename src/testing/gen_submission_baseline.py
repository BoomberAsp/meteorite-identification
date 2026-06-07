"""
gen_submission_baseline.py

使用基线 LoRA 模型在 test_images-v2 上推理，生成 submission 文件。
提示词、数据路径、输出路径均已修正为基线格式。
"""

import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel

# ================= 路径配置 =================
PROJECT_ROOT = "/root/Projects/meteorite-identification"
MODEL_ROOT = os.path.join(PROJECT_ROOT, "models")
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")

# 基线模型路径（自动选择最新的）
import glob
BASELINE_DIRS = sorted(glob.glob(
    os.path.join(MODEL_ROOT, "qwen3-vl-lora-meteorite_baseline_*/final_adapter_*")))
if not BASELINE_DIRS:
    raise RuntimeError("未找到基线模型权重")
ADAPTER_PATH = BASELINE_DIRS[-1]
print(f"使用基线模型: {ADAPTER_PATH}")

BASE_MODEL_ID = os.path.join(MODEL_ROOT, "Qwen3-VL-8B-Instruct")

# 测试集
TEST_DIR = os.path.join(DATA_ROOT, "test_images-v2")
TEMPLATE_CSV = os.path.join(DATA_ROOT, "sample_submission_v2.csv")

# 输出
OUTPUT_DIR = os.path.join(DATA_ROOT, "logs")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "submission_baseline.csv")

# ================= 提示词 =================
# 与 merged 训练数据一致：先回答"是/否"，再给理由
INSTRUCTION = (
    '请观察这张图片中石头的视觉特征（颜色、纹理、形状），'
    '判断其中是否有陨石？注意：忽略图片中的所有文字、标签、标尺等信息，'
    '仅根据石头本身的外观特征判断。回答时首先给出"是"或"否"，'
    '然后简要说明理由。'
)


class TestImageDataset(Dataset):
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.image_files = sorted([
            f for f in os.listdir(image_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))
        ])
        if not self.image_files:
            raise ValueError(f"No image files found in {image_dir}")

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

    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()

    processor = AutoProcessor.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    processor.tokenizer.padding_side = 'left'

    return model, processor


def predict(model, processor, test_loader, device):
    """
    对测试集推理。基线模型输出为详细 reasoning 文本，
    提取末尾的"是"或"否"作为预测。
    """
    model.eval()
    id_to_pred = {}

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting"):
            filenames, images = zip(*batch)
            filenames = list(filenames)
            scaled_images = []
            for img in images:
                img.thumbnail((720, 720), Image.Resampling.LANCZOS)
                scaled_images.append(img)

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": None},
                        {"type": "text", "text": INSTRUCTION}
                    ]
                }
            ]
            texts = [
                processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True)
                for _ in range(len(images))
            ]

            inputs = processor(
                text=texts,
                images=scaled_images,
                return_tensors="pt",
                padding=True
            ).to(device)

            generated_ids = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
            )

            for filename, gen_id in zip(filenames, generated_ids):
                generated_text = processor.decode(
                    gen_id[inputs['input_ids'].shape[1]:], skip_special_tokens=True)

                # 答案在前格式："是。理由：..." 或 "否。理由：..."
                # 兼容理由在后格式："...因此判断为：是/否。"
                if generated_text.startswith("是"):
                    pred_label = 1
                elif generated_text.startswith("否"):
                    pred_label = 0
                elif "因此判断为：是" in generated_text:
                    pred_label = 1
                elif "因此判断为：否" in generated_text:
                    pred_label = 0
                else:
                    # fallback
                    pred_label = 1 if "是" in generated_text else 0

                id_to_pred[filename] = pred_label

    return id_to_pred


def make_submission(id_to_pred, template_csv_path, output_path):
    """生成 submission CSV"""
    ids_df = pd.read_csv(template_csv_path)
    submission_df = ids_df.copy()
    submission_df["label"] = submission_df["id"].map(id_to_pred)

    missing = submission_df["label"].isna().sum()
    if missing > 0:
        missing_ids = submission_df.loc[submission_df["label"].isna(), "id"].head(5).tolist()
        print(f"⚠️  缺失 {missing} 条预测，示例: {missing_ids}")
        submission_df["label"] = submission_df["label"].fillna(0).astype(int)
    else:
        submission_df["label"] = submission_df["label"].astype(int)

    submission_df.to_csv(output_path, index=False)


def main():
    print("=" * 60)
    print("基线模型推理：test_images-v2")
    print(f"  模型: {ADAPTER_PATH}")
    print(f"  测试集: {TEST_DIR}")
    print(f"  输出: {OUTPUT_CSV}")
    print("=" * 60)

    model, processor = load_model()
    device = next(model.parameters()).device

    dataset = TestImageDataset(TEST_DIR)
    print(f"\n测试图像数量: {len(dataset)}")

    test_loader = DataLoader(
        dataset, batch_size=4, shuffle=False,
        num_workers=2, collate_fn=collate_fn
    )

    print("\n开始推理...")
    id_to_pred = predict(model, processor, test_loader, device)
    print(f"预测完成，共 {len(id_to_pred)} 条结果")

    # 统计分布
    yes_count = sum(1 for v in id_to_pred.values() if v == 1)
    no_count = len(id_to_pred) - yes_count
    print(f"  预测为'是': {yes_count} ({yes_count/len(id_to_pred)*100:.1f}%)")
    print(f"  预测为'否': {no_count} ({no_count/len(id_to_pred)*100:.1f}%)")

    # 生成 submission
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    make_submission(id_to_pred, TEMPLATE_CSV, OUTPUT_CSV)
    print(f"\n✅ Submission 已保存至: {OUTPUT_CSV}")

    # 预览
    print(f"\n前 5 条预测结果:")
    df = pd.read_csv(OUTPUT_CSV)
    print(df.head().to_string(index=False))


if __name__ == "__main__":
    main()
