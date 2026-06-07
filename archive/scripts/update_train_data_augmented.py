"""
update_train_data_augmented.py

更新训练数据文件，包含增强后的图片：
- 基于 qwen3_sft_train_extracted.jsonl（原始提取图片）
- 添加增强图片（_aug0, _aug1）的训练数据
- 输出到 qwen3_sft_train_extracted_augmented.jsonl
"""

import os
import json
from tqdm import tqdm

# 路径配置
PROJECT_ROOT = "/root/Projects/meteorite-identification"
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")

INPUT_FILE = os.path.join(DATA_ROOT, "qwen3_sft_train_extracted.jsonl")
OUTPUT_FILE = os.path.join(DATA_ROOT, "qwen3_sft_train_extracted_augmented.jsonl")

# 图片目录
EXTRACTED_DIR = os.path.join(DATA_ROOT, "train_images_extracted")
AUGMENTED_DIR = os.path.join(DATA_ROOT, "train_images_extracted_augmented")


def main():
    print("=" * 60)
    print("更新训练数据（包含增强图片）")
    print("=" * 60)

    # 检查输入文件
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 输入文件不存在: {INPUT_FILE}")
        return

    # 检查增强图片目录
    if not os.path.exists(AUGMENTED_DIR):
        print(f"❌ 增强图片目录不存在: {AUGMENTED_DIR}")
        return

    # 获取增强图片列表
    augmented_images = set(os.listdir(AUGMENTED_DIR))
    print(f"增强图片数量: {len(augmented_images)}")

    # 统计
    original_count = 0
    augmented_count = 0
    total_count = 0

    # 读取原始数据
    print(f"\n读取原始训练数据: {INPUT_FILE}")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        original_data = [json.loads(line.strip()) for line in f]

    print(f"原始数据条数: {len(original_data)}")

    # 处理并输出
    print(f"\n生成增强训练数据...")

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        for item in tqdm(original_data, desc="Processing"):
            # 1. 写入原始数据（更新路径到增强目录的原始图片）
            original_path = item["image_path"]
            img_name = os.path.basename(original_path)

            # 检查原始图片是否在增强目录中（可能有同名）
            # 优先使用提取目录的原始图片
            f_out.write(json.dumps(item, ensure_ascii=False) + '\n')
            original_count += 1
            total_count += 1

            # 2. 为增强版本生成数据
            # 增强图片命名格式：000001_aug0.jpg, 000001_aug1.jpg
            base_name, ext = os.path.splitext(img_name)

            for aug_suffix in ["_aug0", "_aug1"]:
                aug_img_name = f"{base_name}{aug_suffix}{ext}"

                if aug_img_name in augmented_images:
                    # 创建增强版本的数据条目
                    aug_item = {
                        "image_path": os.path.join(AUGMENTED_DIR, aug_img_name),
                        "instruction": item["instruction"],
                        "output": item["output"]  # 保持相同的标签
                    }
                    f_out.write(json.dumps(aug_item, ensure_ascii=False) + '\n')
                    augmented_count += 1
                    total_count += 1

    # 输出统计
    print("\n" + "=" * 60)
    print("处理完成!")
    print("=" * 60)
    print(f"原始图片数据: {original_count} 条")
    print(f"增强图片数据: {augmented_count} 条")
    print(f"总数据条数: {total_count} 条")
    print(f"\n✅ 输出文件: {OUTPUT_FILE}")

    # 验证
    with open(OUTPUT_FILE, 'r') as f:
        output_count = sum(1 for _ in f)
    print(f"   文件行数验证: {output_count} 行")

    # 显示示例
    print("\n数据示例（前3条）:")
    with open(OUTPUT_FILE, 'r') as f:
        for i, line in enumerate(f):
            if i < 3:
                data = json.loads(line)
                print(f"  {i+1}. {os.path.basename(data['image_path'])}")


if __name__ == "__main__":
    main()