"""
fix_merged_paths.py

将 training_data_merged.jsonl（答案在前，理由在后）中的相对 image_path
修正为指向 data/train_images/ 的绝对路径，输出为可直接训练的 JSONL。
"""

import os
import json

PROJECT_ROOT = "/root/Projects/meteorite-identification"
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")

INPUT_FILE = os.path.join(DATA_ROOT, "merged", "training_data_merged.jsonl")
OUTPUT_FILE = os.path.join(DATA_ROOT, "qwen3_sft_train_baseline.jsonl")
TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train_images")


def main():
    assert os.path.exists(TRAIN_IMG_DIR), f"训练图片目录不存在: {TRAIN_IMG_DIR}"

    count = 0
    not_found = 0

    with open(INPUT_FILE, "r", encoding="utf-8") as fin, \
         open(OUTPUT_FILE, "w", encoding="utf-8") as fout:

        for line in fin:
            record = json.loads(line)
            img_name = os.path.basename(record["image_path"])
            abs_path = os.path.join(TRAIN_IMG_DIR, img_name)

            if not os.path.exists(abs_path):
                not_found += 1
                continue

            record["image_path"] = abs_path
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    print(f"处理完成: {count} 条有效记录, {not_found} 条图片未找到")
    print(f"输出文件: {OUTPUT_FILE}")

    # 统计标签分布
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        yes_count = sum(1 for l in f if json.loads(l)["output"].startswith("是"))
    print(f"  其中判断为'是'的样本: {yes_count}")
    print(f"  其中判断为'否'的样本: {count - yes_count}")


if __name__ == "__main__":
    main()
