"""
eval_train_val.py

评估模型在训练集和验证集上的表现，诊断过拟合问题。
"""

import os
import json
import torch
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset, Dataset
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix


# ================= 路径配置 =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(BASE_DIR, "..", "..")
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
MODEL_ROOT = os.path.join(PROJECT_ROOT, "models")

BASE_MODEL_ID = os.path.join(MODEL_ROOT, "Qwen3-VL-8B-Instruct")
DATA_PATH = os.path.join(DATA_ROOT, "qwen3_sft_train_simple.jsonl")

# 默认测试的 checkpoint
DEFAULT_CHECKPOINT = "qwen3-vl-lora-meteorite_20260506_150221_extractedButOverfitting/checkpoint-2380"


def load_model(checkpoint_path: str):
    """
    加载基础模型和 LoRA adapter
    """
    adapter_path = os.path.join(MODEL_ROOT, checkpoint_path)

    print(f"加载基础模型: {BASE_MODEL_ID}")
    print(f"加载 LoRA adapter: {adapter_path}")

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

    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    processor = AutoProcessor.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    processor.tokenizer.padding_side = 'left'

    device = "cuda"
    return model, processor, device


def split_dataset(dataset: Dataset, val_ratio: float = 0.1, seed: int = 42):
    """
    与训练脚本相同的数据集划分方式
    """
    val_size = int(len(dataset) * val_ratio)
    split = dataset.train_test_split(test_size=val_size, seed=seed)
    return split["train"], split["test"]


def extract_label(output: str) -> int:
    """
    从模型输出中提取标签
    """
    output = output.strip()
    if "是" in output and "否" not in output:
        return 1
    elif "否" in output:
        return 0
    else:
        # 无法解析，返回 -1
        return -1


def evaluate_on_dataset(model, processor, device, dataset, dataset_name: str, batch_size: int = 4):
    """
    在指定数据集上评估模型
    """
    print(f"\n{'='*60}")
    print(f"评估 {dataset_name} ({len(dataset)} 样本)")
    print(f"{'='*60}")

    instruction = "请观察这张图片，判断是否为陨石？回答\"是\"或\"否\"。"

    predictions = []
    ground_truths = []
    invalid_count = 0

    # 分批处理
    for i in tqdm(range(0, len(dataset), batch_size), desc=f"评估 {dataset_name}"):
        batch_indices = range(i, min(i + batch_size, len(dataset)))
        batch_data = [dataset[j] for j in batch_indices]

        # 加载图像
        images = []
        labels = []
        for item in batch_data:
            try:
                img = Image.open(item["image_path"]).convert("RGB")
                img.thumbnail((720, 720), Image.Resampling.LANCZOS)
                images.append(img)
                # 从 output 字段提取真实标签
                gt_label = extract_label(item["output"])
                labels.append(gt_label)
            except Exception as e:
                print(f"加载图像失败: {item['image_path']}, {e}")
                continue

        if not images:
            continue

        # 构造输入
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": None},
                    {"type": "text", "text": instruction}
                ]
            }
        ]
        texts = [processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                 for _ in range(len(images))]

        inputs = processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True
        ).to(device)

        # 生成
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=False,
                temperature=None,
                top_p=None
            )

        # 解码并提取预测
        for j, gen_id in enumerate(generated_ids):
            generated_text = processor.decode(
                gen_id[inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            pred_label = extract_label(generated_text)

            if pred_label == -1:
                invalid_count += 1
                print(f"  无法解析输出: '{generated_text}' (真实标签: {labels[j]})")

            predictions.append(pred_label)
            ground_truths.append(labels[j])

    # 计算指标（排除无效预测）
    valid_preds = [p for p, g in zip(predictions, ground_truths) if p != -1]
    valid_gts = [g for p, g in zip(predictions, ground_truths) if p != -1]

    if len(valid_preds) == 0:
        print("❌ 没有有效的预测结果")
        return None

    accuracy = accuracy_score(valid_gts, valid_preds)
    f1 = f1_score(valid_gts, valid_preds, average='binary')
    precision = precision_score(valid_gts, valid_preds, average='binary', zero_division=0)
    recall = recall_score(valid_gts, valid_preds, average='binary', zero_division=0)
    cm = confusion_matrix(valid_gts, valid_preds)

    # 打印结果
    print(f"\n{dataset_name} 评估结果:")
    print(f"  总样本数: {len(dataset)}")
    print(f"  有效预测: {len(valid_preds)}")
    print(f"  无效预测: {invalid_count}")
    print(f"  准确率 (Accuracy): {accuracy:.4f}")
    print(f"  F1-score: {f1:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"\n混淆矩阵:")
    print(f"  预测为0 | 预测为1")
    print(f"  真实为0 | {cm[0][0]} | {cm[0][1]}")
    print(f"  真实为1 | {cm[1][0]} | {cm[1][1]}")

    return {
        "dataset_name": dataset_name,
        "total_samples": len(dataset),
        "valid_predictions": len(valid_preds),
        "invalid_predictions": invalid_count,
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "confusion_matrix": cm.tolist()
    }


def main(checkpoint_path: str = DEFAULT_CHECKPOINT):
    print("="*60)
    print("训练集 vs 验证集 评估")
    print("="*60)
    print(f"Checkpoint: {checkpoint_path}")

    # 1. 加载模型
    print("\n加载模型...")
    model, processor, device = load_model(checkpoint_path)

    # 2. 加载并划分数据集（与训练脚本相同的方式）
    print("\n加载数据集...")
    raw_dataset = load_dataset("json", data_files=DATA_PATH, split="train")
    print(f"原始数据集大小: {len(raw_dataset)}")

    train_dataset, val_dataset = split_dataset(raw_dataset, val_ratio=0.1, seed=42)
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")

    # 3. 评估训练集（采样部分样本，避免太慢）
    TRAIN_EVAL_SIZE = 500  # 评估训练集的样本数
    train_subset = train_dataset.shuffle(seed=42).select(range(min(TRAIN_EVAL_SIZE, len(train_dataset))))
    train_results = evaluate_on_dataset(model, processor, device, train_subset, "训练集子集", batch_size=4)

    # 4. 评估验证集
    val_results = evaluate_on_dataset(model, processor, device, val_dataset, "验证集", batch_size=4)

    # 5. 汇总对比
    print("\n" + "="*60)
    print("结果对比")
    print("="*60)

    if train_results and val_results:
        print(f"\n指标              训练集子集        验证集")
        print(f"-" * 50)
        print(f"准确率            {train_results['accuracy']:.4f}          {val_results['accuracy']:.4f}")
        print(f"F1-score          {train_results['f1']:.4f}          {val_results['f1']:.4f}")
        print(f"Precision         {train_results['precision']:.4f}          {val_results['precision']:.4f}")
        print(f"Recall            {train_results['recall']:.4f}          {val_results['recall']:.4f}")

        # 判断是否过拟合
        if train_results['accuracy'] - val_results['accuracy'] > 0.1:
            print("\n⚠️ 可能存在过拟合：训练集表现明显优于验证集")
        elif train_results['accuracy'] < val_results['accuracy']:
            print("\n⚠️ 异常：验证集表现优于训练集，请检查数据划分")
        else:
            print("\n✅ 训练集和验证集表现相近，无明显过拟合")

    # 6. 保存结果
    output_dir = os.path.join(DATA_ROOT, "logs")
    os.makedirs(output_dir, exist_ok=True)

    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"eval_train_val_{timestamp}.json")

    results = {
        "checkpoint": checkpoint_path,
        "timestamp": timestamp,
        "train_subset_size": TRAIN_EVAL_SIZE,
        "train_results": train_results,
        "val_results": val_results
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n结果已保存至: {output_file}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="评估模型在训练集和验证集上的表现")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT,
                        help="LoRA checkpoint 路径（相对于 models/ 目录）")
    args = parser.parse_args()

    main(args.checkpoint)