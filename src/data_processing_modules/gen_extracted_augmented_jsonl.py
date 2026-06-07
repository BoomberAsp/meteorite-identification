"""
gen_extracted_augmented_jsonl.py

基于 merged 训练数据（答案在前，理由在后），
为 SAM2 提取后的石头图像 + 增强图像生成完整的 JSONL 标注。

包含:
- train_images_extracted/ (原始提取)
- train_images_extracted_augmented/ (旋转 + 水平翻转增强)
按 5:1 比例重采样。
"""

import os
import json
import random
from tqdm import tqdm

PROJECT_ROOT = "/root/Projects/meteorite-identification"
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")

# 输入
MERGED_JSONL = os.path.join(DATA_ROOT, "merged", "training_data_merged.jsonl")
ORIGINAL_DIR = os.path.join(DATA_ROOT, "train_images_extracted")
AUGMENTED_DIR = os.path.join(DATA_ROOT, "train_images_extracted_augmented")

# 输出
OUTPUT_FILE = os.path.join(DATA_ROOT, "qwen3_sft_train_extracted_augmented.jsonl")

# 目标比例
TARGET_RATIO = 5.0


def load_merged_data():
    """加载 merged JSONL，构建 {img_name: {instruction, output, label}}"""
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


def _extract_original_id(img_name, data_dict):
    """从文件名提取原始 ID，在 data_dict 中查找匹配:
    000001_aug0.jpg -> 尝试 000001.jpg / .jpeg / .png
    000001.jpg      -> 直接返回 000001.jpg
    """
    # 分离主名和扩展名
    name, ext = os.path.splitext(img_name)

    # 如果是增强/降质文件，去掉后缀前缀
    for sep in ['_aug', '_deg']:
        if sep in name:
            name = name.split(sep)[0]
            # 尝试所有扩展名
            for e in ['.jpg', '.jpeg', '.png']:
                candidate = name + e
                if candidate in data_dict:
                    return candidate
            return None

    # 原始文件名，直接返回
    if img_name in data_dict:
        return img_name
    return None


def _scan_directory(img_dir, data_dict, meteorite_data, non_meteorite_data, desc=""):
    if not os.path.exists(img_dir):
        print(f"  目录不存在: {img_dir}")
        return
    images = sorted([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
    print(f"  图片数: {len(images)}")

    matched = 0
    for img_name in tqdm(images, desc=desc):
        original_id = _extract_original_id(img_name, data_dict)
        record = data_dict.get(original_id)
        if record is None:
            continue
        matched += 1
        entry = {
            "image_path": os.path.join(img_dir, img_name),
            "instruction": record["instruction"],
            "output": record["output"],
        }
        if record["label"] == 1:
            meteorite_data.append(entry)
        else:
            non_meteorite_data.append(entry)

    print(f"  匹配成功: {matched}/{len(images)}")


def main():
    print("=" * 60)
    print("构建提取图像 + 增强图像的 JSONL 标注（答案在前，理由在后）")
    print("=" * 60)

    print(f"\n加载 merged 数据: {MERGED_JSONL}")
    data_dict = load_merged_data()
    yes_c = sum(1 for v in data_dict.values() if v["label"] == 1)
    no_c = len(data_dict) - yes_c
    print(f"  加载 {len(data_dict)} 条 | 陨石: {yes_c} | 非陨石: {no_c}")

    meteorite_data = []
    non_meteorite_data = []

    # 原始提取图像
    print(f"\n[1/2] 扫描原始提取图像...")
    _scan_directory(ORIGINAL_DIR, data_dict, meteorite_data, non_meteorite_data,
                    desc="提取图像")

    # 增强图像
    print(f"\n[2/2] 扫描增强图像...")
    _scan_directory(AUGMENTED_DIR, data_dict, meteorite_data, non_meteorite_data,
                    desc="增强图像")

    total = len(meteorite_data) + len(non_meteorite_data)
    print(f"\n全部数据: {total} 条（不重采样，保留全部）")
    print(f"  陨石: {len(meteorite_data)} | 非陨石: {len(non_meteorite_data)}")

    all_data = meteorite_data + non_meteorite_data
    random.shuffle(all_data)

    # 写入
    print(f"\n写入: {OUTPUT_FILE}")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for d in tqdm(all_data, desc="写入"):
            f.write(json.dumps(d, ensure_ascii=False) + '\n')

    # 验证
    with open(OUTPUT_FILE, 'r') as f:
        count = sum(1 for _ in f)
    print(f"最终行数: {count}")

    orig_c = sum(1 for d in all_data if '_aug' not in os.path.basename(d['image_path']))
    aug_c = sum(1 for d in all_data if '_aug' in os.path.basename(d['image_path']))
    print(f"  原始提取: {orig_c} | 增强: {aug_c}")


if __name__ == "__main__":
    main()
