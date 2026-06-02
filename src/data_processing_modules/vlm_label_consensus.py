"""
vlm_label_consensus.py

用 Qwen3-VL-8B-Instruct（多模态，能看图）为共识样本生成详细 reasoning。
已知人类标注的 answer（是/否），让 VLM 观察图片特征后写出判断理由。

输入: data/logs/consensus_99.txt + data/human_labeled/sample_submission_560.csv
输出: data/human_labeled/consensus_with_reasoning.jsonl
"""

import os, json
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

PROJECT_ROOT = "/root/Projects/meteorite-identification"
MODEL_ROOT = os.path.join(PROJECT_ROOT, "models")
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
MODEL_ID = os.path.join(MODEL_ROOT, "Qwen3-VL-8B-Instruct")

TEST_DIR = os.path.join(DATA_ROOT, "test_images-v2")
LABEL_CSV = os.path.join(DATA_ROOT, "human_labeled", "sample_submission_560.csv")
CONSENSUS_FILE = os.path.join(DATA_ROOT, "logs", "consensus_99.txt")
OUTPUT_FILE = os.path.join(DATA_ROOT, "human_labeled", "consensus_with_reasoning.jsonl")

# 指令（与训练一致：先回答再给理由）
INSTRUCTION = (
    "请观察这张图片中石头的视觉特征（颜色、纹理、形状），"
    "判断其中是否有陨石？注意：忽略图片中的所有文字、标签、标尺等信息，"
    "仅根据石头本身的外观特征判断。"
    "回答时首先给出\"是\"或\"否\"，然后简要说明理由。"
)

VLM_PROMPTS = {
    1: (
        "请观察这张石头图片。已知它是陨石。请以地质学专家视角，"
        "基于图片中实际可见的鉴别特征（如熔壳、气印、球粒结构、颜色、纹理、形状等），"
        "生成一段判断理由。不要编造不存在的特征，只描述你真正看到的。"
        "格式：是。理由：..."
    ),
    0: (
        "请观察这张石头图片。已知它不是陨石，是一块普通地球岩石。"
        "请以地质学专家视角，基于图片中实际可见的特征"
        "（如缺乏熔壳、显晶质结构、沉积层理、颜色、纹理、形状等），"
        "生成一段判断理由。不要编造不存在的特征，只描述你真正看到的。"
        "格式：否。理由：..."
    ),
}


def load_model():
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, quantization_config=bnb, device_map="auto", trust_remote_code=True
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    return model, processor


def generate_reasoning(model, processor, image_path, label):
    """VLM 看图生成 reasoning"""
    # 先让 VLM 看一眼生成模板
    prompt = VLM_PROMPTS[label]
    img = Image.open(image_path).convert("RGB")
    img.thumbnail((720, 720), Image.Resampling.LANCZOS)

    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": prompt},
        ]}
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[img], return_tensors="pt", padding=True).to(model.device)

    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=512, do_sample=True, temperature=0.7, top_p=0.9)

    result = processor.decode(generated[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()

    # 确保以 是/否 开头
    if result.startswith("是") or result.startswith("否"):
        return result
    # 尝试修复
    if "是" in result[:3]:
        return result
    return f"{'是' if label == 1 else '否'}。理由：{result.lstrip('是。理由：否。理由：')}"


def main():
    print("=" * 60)
    print("VLM 为共识样本生成 reasoning")

    # 加载标注和共识列表
    import pandas as pd
    labels = pd.read_csv(LABEL_CSV).set_index("id")

    with open(CONSENSUS_FILE) as f:
        consensus_ids = [line.strip() for line in f if line.strip()]

    print(f"  共识样本: {len(consensus_ids)} 张")

    # 加载模型
    print("\n加载 Qwen3-VL-8B-Instruct...")
    model, processor = load_model()
    print("模型就绪")

    # 生成 reasoning
    records = []
    for fid in tqdm(consensus_ids, desc="生成 reasoning"):
        img_path = os.path.join(TEST_DIR, fid)
        if not os.path.exists(img_path):
            print(f"  图片不存在: {img_path}")
            continue

        label = int(labels.loc[fid, "label"])
        reasoning = generate_reasoning(model, processor, img_path, label)

        records.append({
            "image_path": os.path.abspath(img_path),
            "instruction": INSTRUCTION,
            "output": reasoning,
            "source": "consensus_vlm",
            "label": label,
        })

    # 保存
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    yes_c = sum(1 for r in records if r["output"].startswith("是"))
    no_c = sum(1 for r in records if r["output"].startswith("否"))
    print(f"\n完成! {len(records)} 条")
    print(f"  是: {yes_c}  否: {no_c}")
    print(f"  输出: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
