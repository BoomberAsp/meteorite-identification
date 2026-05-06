"""
update_train_data_path.py

更新训练数据文件中的图片路径：
- 将 train_images/ 替换为 train_images_extracted/
- 检查新路径是否存在
- 输出到 qwen3_sft_train_extracted.jsonl
"""

import os
import json
from tqdm import tqdm

# 路径配置
PROJECT_ROOT = "/root/Projects/meteorite-identification"
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")

INPUT_FILE = os.path.join(DATA_ROOT, "qwen3_sft_train.jsonl")
OUTPUT_FILE = os.path.join(DATA_ROOT, "qwen3_sft_train_extracted.jsonl")

# 路径替换规则
OLD_DIR = "train_images"
NEW_DIR = "train_images_extracted"


def main():
    print("=" * 60)
    print("更新训练数据图片路径")
    print("=" * 60)

    # 检查输入文件
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 输入文件不存在: {INPUT_FILE}")
        return

    # 检查提取图片目录
    extracted_dir = os.path.join(DATA_ROOT, NEW_DIR)
    if not os.path.exists(extracted_dir):
        print(f"❌ 提取图片目录不存在: {extracted_dir}")
        return

    # 获取已提取图片列表
    extracted_images = set(os.listdir(extracted_dir))
    print(f"已提取图片数量: {len(extracted_images)}")

    # 统计
    total_count = 0
    success_count = 0
    missing_count = 0
    missing_images = []

    # 处理数据
    print(f"\n读取输入文件: {INPUT_FILE}")

    with open(INPUT_FILE, 'r', encoding='utf-8') as f_in:
        lines = f_in.readlines()

    print(f"原始数据条数: {len(lines)}")
    print(f"\n开始处理...")

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        for line in tqdm(lines, desc="Processing"):
            total_count += 1

            try:
                data = json.loads(line.strip())
            except json.JSONDecodeError as e:
                print(f"JSON 解析错误: {e}")
                continue

            # 获取原始图片路径
            old_path = data.get("image_path", "")

            # 替换路径
            new_path = old_path.replace(OLD_DIR, NEW_DIR)

            # 获取图片文件名
            img_name = os.path.basename(new_path)

            # 检查新路径是否存在
            if img_name in extracted_images:
                # 更新路径
                data["image_path"] = new_path
                f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
                success_count += 1
            else:
                missing_count += 1
                missing_images.append(img_name)
                # 可选：跳过不存在的图片，或保留原路径
                # 这里选择跳过

    # 输出统计
    print("\n" + "=" * 60)
    print("处理完成!")
    print("=" * 60)
    print(f"原始数据条数: {total_count}")
    print(f"成功更新条数: {success_count}")
    print(f"缺失图片条数: {missing_count}")
    print(f"输出文件: {OUTPUT_FILE}")

    if missing_count > 0:
        print(f"\n⚠️ 缺失图片示例 (前10个):")
        for img in missing_images[:10]:
            print(f"   {img}")

        # 保存缺失图片列表
        missing_file = os.path.join(DATA_ROOT, "missing_images.txt")
        with open(missing_file, 'w') as f:
            for img in missing_images:
                f.write(img + '\n')
        print(f"\n缺失图片完整列表已保存至: {missing_file}")

    print(f"\n✅ 新训练数据文件: {OUTPUT_FILE}")
    print(f"   可用于 train_lora.py 的 DATA_PATH 配置")


if __name__ == "__main__":
    main()