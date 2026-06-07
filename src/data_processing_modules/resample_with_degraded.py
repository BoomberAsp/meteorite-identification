"""
resample_with_degraded.py

构建包含原始 + 增强 + 降质图像的 5:1 训练数据（答案在前，理由在后）：
1. 以 training_data_merged.jsonl 为源（含详细 reasoning）
2. 包含 train_images（原始）、train_images_augmented（旋转/缩放）、train_images_degraded（域降质）
3. 增强/降质图像继承原图的 instruction 和 output
4. 按 5:1 比例重采样（非陨石:陨石）
5. 输出到 qwen3_sft_train_resampled_aug.jsonl
"""

import os
import json
import random
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = "/root/Projects/meteorite-identification"
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")

# 输入
MERGED_JSONL = os.path.join(DATA_ROOT, "merged", "training_data_merged.jsonl")
ORIGINAL_IMG_DIR = os.path.join(DATA_ROOT, "train_images")
AUGMENTED_IMG_DIR = os.path.join(DATA_ROOT, "train_images_augmented")
DEGRADED_IMG_DIR = os.path.join(DATA_ROOT, "train_images_degraded")

# 输出
OUTPUT_FILE = os.path.join(DATA_ROOT, "qwen3_sft_train_resampled_aug.jsonl")

# 目标比例：非陨石:陨石 = 5:1
TARGET_RATIO = 5.0


def load_merged_data():
    """加载 merged JSONL，构建 {img_name: {instruction, output, label}} 字典"""
    data_dict = {}
    with open(MERGED_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            img_name = os.path.basename(record["image_path"])
            output_text = record.get("output", "")
            label = 1 if output_text.startswith("是") else 0
            data_dict[img_name] = {
                "instruction": record.get("instruction", ""),
                "output": output_text,
                "label": label,
            }
    return data_dict


def _extract_original_id(img_name, label_dict):
    """从文件名提取原始 ID，在 label_dict 中查找匹配:
    000001_aug0.jpg -> 尝试 000001.jpg / .jpeg / .png
    000001.jpg      -> 直接返回 000001.jpg
    """
    name, ext = os.path.splitext(img_name)

    for sep in ['_aug', '_deg']:
        if sep in name:
            name = name.split(sep)[0]
            for e in ['.jpg', '.jpeg', '.png']:
                candidate = name + e
                if candidate in label_dict:
                    return candidate
            return None

    if img_name in label_dict:
        return img_name
    return None


def _process_directory(img_dir, data_dict, meteorite_data, non_meteorite_data, desc="图片"):
    """扫描目录，对每张图查找对应的 merged 数据，增强/降质图继承原图"""
    if not os.path.exists(img_dir):
        print(f"  目录不存在，跳过: {img_dir}")
        return
    images = sorted([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
    print(f"  图片总数: {len(images)}")

    matched = 0
    unmatched = 0
    for img_name in tqdm(images, desc=desc):
        original_id = _extract_original_id(img_name, data_dict)
        record = data_dict.get(original_id)
        if record is None:
            unmatched += 1
            continue

        matched += 1
        data_entry = {
            "image_path": os.path.join(img_dir, img_name),
            "instruction": record["instruction"],
            "output": record["output"],
        }
        if record["label"] == 1:
            meteorite_data.append(data_entry)
        else:
            non_meteorite_data.append(data_entry)

    if unmatched > 0:
        print(f"  ⚠️  未匹配: {unmatched}/{len(images)} 张（merged 中无对应记录）")


def main():
    print("=" * 60)
    print("构建增强训练数据（答案在前，理由在后；含原始 + 旋转缩放 + 域降质）")
    print("=" * 60)

    # 加载 merged 数据
    print(f"\n加载 merged 数据: {MERGED_JSONL}")
    data_dict = load_merged_data()
    print(f"  加载 {len(data_dict)} 条记录")
    yes_count = sum(1 for v in data_dict.values() if v["label"] == 1)
    no_count = len(data_dict) - yes_count
    print(f"  陨石: {yes_count}, 非陨石: {no_count}")

    meteorite_data = []
    non_meteorite_data = []

    # 1. 原始图片
    print(f"\n[1/3] 处理原始图片...")
    _process_directory(ORIGINAL_IMG_DIR, data_dict, meteorite_data, non_meteorite_data,
                       desc="原始图片")

    # 2. 旋转/缩放增强图片
    print(f"\n[2/3] 处理旋转/缩放增强图片...")
    _process_directory(AUGMENTED_IMG_DIR, data_dict, meteorite_data, non_meteorite_data,
                       desc="旋转/缩放增强")

    # 3. 域降质图片
    print(f"\n[3/3] 处理域降质图片...")
    _process_directory(DEGRADED_IMG_DIR, data_dict, meteorite_data, non_meteorite_data,
                       desc="域降质图片")

    # 统计
    total = len(meteorite_data) + len(non_meteorite_data)
    print(f"\n全部数据分布:")
    print(f"  陨石: {len(meteorite_data)} ({len(meteorite_data)/total*100:.1f}%)")
    print(f"  非陨石: {len(non_meteorite_data)} ({len(non_meteorite_data)/total*100:.1f}%)")
    print(f"  总计: {total}")

    # 计算目标陨石样本数
    target_meteorite_count = int(len(non_meteorite_data) / TARGET_RATIO)
    target_meteorite_count = min(target_meteorite_count, len(meteorite_data))
    print(f"\n目标分布 (非陨石:陨石 = {TARGET_RATIO}:1):")
    print(f"  非陨石: {len(non_meteorite_data)} (保留全部)")
    print(f"  陨石: {target_meteorite_count} (随机采样)")
    print(f"  总计: {len(non_meteorite_data) + target_meteorite_count}")

    random.seed(42)
    sampled_meteorite = random.sample(meteorite_data, target_meteorite_count)
    all_data = non_meteorite_data + sampled_meteorite
    random.shuffle(all_data)

    # 写入输出
    print(f"\n写入数据: {OUTPUT_FILE}")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for data in tqdm(all_data, desc="写入数据"):
            f.write(json.dumps(data, ensure_ascii=False) + '\n')

    # 验证
    with open(OUTPUT_FILE, 'r') as f:
        output_count = sum(1 for _ in f)
    print(f"\n最终: {output_count} 行")

    # 来源统计
    total_orig = sum(1 for d in all_data if '_aug' not in os.path.basename(d['image_path'])
                     and '_deg' not in os.path.basename(d['image_path']))
    total_aug = sum(1 for d in all_data if '_aug' in os.path.basename(d['image_path']))
    total_deg = sum(1 for d in all_data if '_deg' in os.path.basename(d['image_path']))
    print(f"\n来源统计:")
    print(f"  原始图片: {total_orig} ({total_orig/output_count*100:.1f}%)")
    print(f"  旋转/缩放增强: {total_aug} ({total_aug/output_count*100:.1f}%)")
    print(f"  域降质: {total_deg} ({total_deg/output_count*100:.1f}%)")

    print(f"\n输出文件: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
