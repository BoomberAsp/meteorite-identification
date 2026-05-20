"""
resample_training_data.py

重采样训练数据，使非陨石:陨石比例接近测试集分布 (4:1)

策略：
- 保留全部非陨石样本（7121条）
- 随机采样约1/4的陨石样本（约1780条）
- 最终比例约 4:1，与测试集一致
"""

import os
import json
import random
from tqdm import tqdm

PROJECT_ROOT = "/root/Projects/meteorite-identification"
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")

INPUT_FILE = os.path.join(DATA_ROOT, "qwen3_sft_train_simple.jsonl")
OUTPUT_FILE = os.path.join(DATA_ROOT, "qwen3_sft_train_resampled.jsonl")

# 目标比例：非陨石:陨石 = 4:1
TARGET_RATIO = 4.0  # 非陨石数量 / 陨石数量


def main():
    print("="*60)
    print("重采样训练数据")
    print("="*60)

    # 读取原始数据
    print(f"\n读取原始数据: {INPUT_FILE}")
    meteorite_data = []
    non_meteorite_data = []

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="读取数据"):
            data = json.loads(line.strip())
            if data["output"] == "是":
                meteorite_data.append(data)
            else:
                non_meteorite_data.append(data)

    print(f"\n原始数据分布:")
    print(f"  陨石: {len(meteorite_data)} ({len(meteorite_data)/(len(meteorite_data)+len(non_meteorite_data))*100:.1f}%)")
    print(f"  非陨石: {len(non_meteorite_data)} ({len(non_meteorite_data)/(len(meteorite_data)+len(non_meteorite_data))*100:.1f}%)")
    print(f"  非陨石:陨石 = {len(non_meteorite_data)/len(meteorite_data):.2f}:1")

    # 计算目标陨石样本数
    # 保留全部非陨石，采样陨石使比例达到 TARGET_RATIO
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
    print(f"\n写入重采样数据: {OUTPUT_FILE}")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for data in tqdm(all_data, desc="写入数据"):
            f.write(json.dumps(data, ensure_ascii=False) + '\n')

    # 验证
    print(f"\n验证输出文件:")
    with open(OUTPUT_FILE, 'r') as f:
        output_count = sum(1 for _ in f)
    print(f"  文件行数: {output_count}")

    # 统计新分布
    meteorite_out = sum(1 for d in all_data if d["output"] == "是")
    non_meteorite_out = sum(1 for d in all_data if d["output"] == "否")

    print(f"\n重采样后分布:")
    print(f"  陨石: {meteorite_out} ({meteorite_out/output_count*100:.1f}%)")
    print(f"  非陨石: {non_meteorite_out} ({non_meteorite_out/output_count*100:.1f}%)")
    print(f"  非陨石:陨石 = {non_meteorite_out/meteorite_out:.2f}:1")

    print("\n" + "="*60)
    print("✅ 重采样完成!")
    print("="*60)
    print(f"\n新训练数据文件: {OUTPUT_FILE}")
    print("可在 train_lora.py 中设置 DATA_PATH 使用此文件")


if __name__ == "__main__":
    main()