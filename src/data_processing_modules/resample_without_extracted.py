"""
resample_without_extracted.py

使用原始未分割图片（包含原始+增强）构建训练数据：
1. 基于 train_labels.csv 标签信息
2. 包含 train_images 目录（原始图片）和 train_images_augmented 目录（增强图片）
3. 简化输出为"是"/"否"
4. 按5:1比例重采样（非陨石:陨石）
5. 输出到 qwen3_sft_train_resampled_without_extracted.jsonl
"""

import os
import json
import random
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = "/root/Projects/meteorite-identification"
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")

# 输入
TRAIN_CSV = os.path.join(DATA_ROOT, "train_labels.csv")  # 标签信息
ORIGINAL_IMG_DIR = os.path.join(DATA_ROOT, "train_images")  # 原始图片目录
AUGMENTED_IMG_DIR = os.path.join(DATA_ROOT, "train_images_augmented")  # 增强图片目录

# 输出
OUTPUT_FILE = os.path.join(DATA_ROOT, "qwen3_sft_train_resampled_without_extracted.jsonl")

# 目标比例：非陨石:陨石 = 5:1
TARGET_RATIO = 5.0

# 简化的 instruction
SIMPLE_INSTRUCTION = "请观察这张图片，判断是否为陨石？回答\"是\"或\"否\"。"


def main():
    print("="*60)
    print("构建未分割图片的5:1训练数据（含原始+增强）")
    print("="*60)

    # 读取 train_labels.csv 获取标签
    print(f"\n读取标签文件: {TRAIN_CSV}")
    train_df = pd.read_csv(TRAIN_CSV)
    print(f"  总样本数: {len(train_df)}")
    print(f"  列: {train_df.columns.tolist()}")

    # 统计原始分布
    meteorite_count = (train_df['label'] == 1).sum()
    non_meteorite_count = (train_df['label'] == 0).sum()
    print(f"\n原始标签分布:")
    print(f"  陨石 (label=1): {meteorite_count} ({meteorite_count/len(train_df)*100:.1f}%)")
    print(f"  非陨石 (label=0): {non_meteorite_count} ({non_meteorite_count/len(train_df)*100:.1f}%)")
    print(f"  非陨石:陨石 = {non_meteorite_count/meteorite_count:.2f}:1")

    # 构建标签字典，方便快速查找
    label_dict = {}
    for row in train_df.itertuples():
        img_id = row.id  # 格式如 "000001.jpg"
        label_dict[img_id] = row.label

    meteorite_data = []
    non_meteorite_data = []

    # 1. 处理原始图片目录
    print(f"\n扫描原始图片目录: {ORIGINAL_IMG_DIR}")
    original_images = [f for f in os.listdir(ORIGINAL_IMG_DIR)
                       if f.endswith(('.jpg', '.jpeg', '.png'))]
    print(f"  图片总数: {len(original_images)}")

    print(f"\n处理原始图片...")
    for img_name in tqdm(original_images, desc="原始图片"):
        label = label_dict.get(img_name)
        if label is None:
            continue

        output = "是" if label == 1 else "否"
        data_entry = {
            "image_path": os.path.join(ORIGINAL_IMG_DIR, img_name),
            "instruction": SIMPLE_INSTRUCTION,
            "output": output
        }

        if label == 1:
            meteorite_data.append(data_entry)
        else:
            non_meteorite_data.append(data_entry)

    # 2. 处理增强图片目录
    print(f"\n扫描增强图片目录: {AUGMENTED_IMG_DIR}")
    augmented_images = [f for f in os.listdir(AUGMENTED_IMG_DIR)
                        if f.endswith(('.jpg', '.jpeg', '.png'))]
    print(f"  图片总数: {len(augmented_images)}")

    print(f"\n处理增强图片...")
    for img_name in tqdm(augmented_images, desc="增强图片"):
        # 图片命名格式: 000001_aug0.jpg 或 000001_aug1.jpg
        base_num = img_name.split('_aug')[0]  # 得到如 000001
        # 构造原始图片名格式
        img_id_str = f"{base_num}.jpg"

        label = label_dict.get(img_id_str)
        if label is None:
            # 也尝试 .jpeg 和 .png
            img_id_str_jpeg = f"{base_num}.jpeg"
            img_id_str_png = f"{base_num}.png"
            label = label_dict.get(img_id_str_jpeg) or label_dict.get(img_id_str_png)
            if label is None:
                continue

        output = "是" if label == 1 else "否"
        data_entry = {
            "image_path": os.path.join(AUGMENTED_IMG_DIR, img_name),
            "instruction": SIMPLE_INSTRUCTION,
            "output": output
        }

        if label == 1:
            meteorite_data.append(data_entry)
        else:
            non_meteorite_data.append(data_entry)

    print(f"\n全部数据分布:")
    total = len(meteorite_data) + len(non_meteorite_data)
    print(f"  陨石: {len(meteorite_data)} ({len(meteorite_data)/total*100:.1f}%)")
    print(f"  非陨石: {len(non_meteorite_data)} ({len(non_meteorite_data)/total*100:.1f}%)")
    print(f"  非陨石:陨石 = {len(non_meteorite_data)/len(meteorite_data):.2f}:1")
    print(f"  总计: {total}")

    # 计算目标陨石样本数（保留全部非陨石）
    target_meteorite_count = int(len(non_meteorite_data) / TARGET_RATIO)
    print(f"\n目标分布 (非陨石:陨石 = {TARGET_RATIO}:1):")
    print(f"  非陨石: {len(non_meteorite_data)} (保留全部)")
    print(f"  陨石: {target_meteorite_count} (随机采样)")
    print(f"  总计: {len(non_meteorite_data) + target_meteorite_count}")

    # 随机采样陨石样本
    random.seed(42)
    sampled_meteorite = random.sample(meteorite_data, target_meteorite_count)

    # 合并并shuffle
    all_data = non_meteorite_data + sampled_meteorite
    random.shuffle(all_data)

    # 写入输出文件
    print(f"\n写入数据: {OUTPUT_FILE}")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for data in tqdm(all_data, desc="写入数据"):
            f.write(json.dumps(data, ensure_ascii=False) + '\n')

    # 验证输出
    print(f"\n验证输出文件:")
    with open(OUTPUT_FILE, 'r') as f:
        output_count = sum(1 for _ in f)
    print(f"  文件行数: {output_count}")

    # 统计新分布
    meteorite_out = sum(1 for d in all_data if d["output"] == "是")
    non_meteorite_out = sum(1 for d in all_data if d["output"] == "否")

    print(f"\n最终分布:")
    print(f"  陨石: {meteorite_out} ({meteorite_out/output_count*100:.1f}%)")
    print(f"  非陨石: {non_meteorite_out} ({non_meteorite_out/output_count*100:.1f}%)")
    print(f"  非陨石:陨石 = {non_meteorite_out/meteorite_out:.2f}:1")

    # 显示示例（区分原始和增强）
    print(f"\n数据示例:")
    orig_count = 0
    aug_count = 0
    for d in all_data[:6]:
        path = d['image_path']
        img_name = os.path.basename(path)
        if '_aug' in img_name:
            source = '增强'
            aug_count += 1
        else:
            source = '原始'
            orig_count += 1
        print(f"  [{source}] {img_name} -> {d['output']}")

    # 统计来源分布
    total_orig = sum(1 for d in all_data if '_aug' not in os.path.basename(d['image_path']))
    total_aug = sum(1 for d in all_data if '_aug' in os.path.basename(d['image_path']))
    print(f"\n来源统计:")
    print(f"  原始图片: {total_orig} ({total_orig/output_count*100:.1f}%)")
    print(f"  增强图片: {total_aug} ({total_aug/output_count*100:.1f}%)")

    print("\n" + "="*60)
    print("✅ 完成!")
    print("="*60)
    print(f"\n输出文件: {OUTPUT_FILE}")
    print("可在 train_lora.py 中设置 DATA_PATH 使用此文件")


if __name__ == "__main__":
    main()