"""
api_rewrite_for_augmented.py

使用 DashScope API 为每条训练数据改写 reasoning 理由，
改写后的文本直接分配给对应的增强图像（_aug0, _aug1）。

流程：
- 原始图像 → 原始 reasoning
- 增强图 _aug0 → 改写版本1
- 增强图 _aug1 → 改写版本2

输出：直接写入完整的 JSONL（含原始 + 增强的完整标注）

用法:
    export DASHSCOPE_API_KEY='sk-xxx'
    python api_rewrite_for_augmented.py -w 16
    python api_rewrite_for_augmented.py --dry-run
"""

import os
import re
import json
import argparse
import concurrent.futures
from tqdm import tqdm

PROJECT_ROOT = "/root/Projects/meteorite-identification"
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")

API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")

# 输入输出
MERGED_JSONL = os.path.join(DATA_ROOT, "merged", "training_data_merged.jsonl")
ORIGINAL_DIR = os.path.join(DATA_ROOT, "train_images_extracted_white_bg")
AUGMENTED_DIR = os.path.join(DATA_ROOT, "train_images_extracted_augmented")
OUTPUT_FILE = os.path.join(DATA_ROOT, "qwen3_sft_train_extracted_augmented.jsonl")

REWRITE_SYSTEM = (
    "你是一个地质学专家。你的任务是用不同的专业术语和句式重写石头判断理由，"
    "保持结论（是/否）不变，保持推理思路不变。只输出改写后的理由，不要添加任何解释。"
)

REWRITE_PROMPTS = {
    "是": (
        "以下是一段关于某块石头被判断为陨石的推理理由。\n"
        "请保持结论\"是\"不变，用不同的专业术语和句式重新表述。\n"
        "要求：保持原有判断依据（如熔壳、气印、球粒等），长度相近。\n"
        "格式与原文一致，以\"是\"开头。\n\n"
        "原始：\n{original}"
    ),
    "否": (
        "以下是一段关于某块石头被判断为非陨石的推理理由。\n"
        "请保持结论\"否\"不变，用不同的专业术语和句式重新表述。\n"
        "要求：保持原有判断依据（如无熔壳、地球岩石特征等），长度相近。\n"
        "格式与原文一致，以\"否\"开头。\n\n"
        "原始：\n{original}"
    ),
}


def call_deepseek(text: str, label: str,
                  model: str = "deepseek-v4-pro",
                  base_url: str = "https://api.deepseek.com/v1") -> str:
    """调用 DeepSeek API 改写文本（OpenAI 兼容格式）"""
    try:
        from openai import OpenAI

        client = OpenAI(api_key=API_KEY, base_url=base_url)
        prompt = REWRITE_PROMPTS[label].format(original=text)

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": REWRITE_SYSTEM},
                {"role": "user", "content": prompt}
            ],
            max_tokens=512,
            temperature=0.7,
            top_p=0.9,
        )

        result = response.choices[0].message.content.strip()
        if result.startswith("是") or result.startswith("否"):
            return result
        return text

    except Exception as e:
        # 429 速率限制时指数退避重试
        if "429" in str(e) or "rate_limit" in str(e).lower():
            import time
            time.sleep(2)
            try:
                return call_deepseek(text, label, model, base_url)
            except Exception:
                pass
        print(f"  API 调用失败: {e}")
        return text


def get_augmented_files(img_name, aug_dir, max_aug=2):
    """查找某张原图对应的增强文件"""
    base = os.path.splitext(img_name)[0]
    aug_files = []
    for i in range(max_aug):
        for ext in ['.jpg', '.jpeg', '.png']:
            aug_name = f"{base}_aug{i}{ext}"
            aug_path = os.path.join(aug_dir, aug_name)
            if os.path.exists(aug_path):
                aug_files.append(aug_path)
                break
    return aug_files


def process_single(idx, record, aug_dir, model_name="deepseek-v4-pro"):
    """处理单条记录：改写 + 分配增强图"""
    img_name = os.path.basename(record["image_path"])
    output_text = record.get("output", "")
    instruction = record.get("instruction", "")
    label = "是" if output_text.startswith("是") else "否"

    # 查找增强图
    aug_files = get_augmented_files(img_name, aug_dir)
    n_aug = len(aug_files)

    results = []

    # 1. 原始图像 → 原始 reasoning
    orig_path = os.path.join(ORIGINAL_DIR, img_name)
    if os.path.exists(orig_path):
        results.append({
            "image_path": orig_path,
            "instruction": instruction,
            "output": output_text,
        })

    # 2. 改写 reasoning 并分配给增强图
    for i, aug_path in enumerate(aug_files):
        rewritten = call_deepseek(output_text, label, model=model_name)
        results.append({
            "image_path": aug_path,
            "instruction": instruction,
            "output": rewritten,
        })

    return (idx, results, n_aug)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--workers", type=int, default=8,
                        help="并行工作线程数")
    parser.add_argument("-m", "--model", default="deepseek-v4-pro",
                        help="DeepSeek 模型名称")
    parser.add_argument("--dry-run", action="store_true",
                        help="预览前3条，不写文件")
    args = parser.parse_args()

    if not API_KEY:
        print("⚠️  未设置 DEEPSEEK_API_KEY 环境变量")
        print("   export DEEPSEEK_API_KEY='sk-xxx'")
        return

    print("=" * 60)
    print("API 改写理由 + 分配增强图")
    print(f"  模型: {args.model}")
    print(f"  并行数: {args.workers}")
    print("=" * 60)

    # 读取 merged 数据
    records = []
    with open(MERGED_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    # 过滤：只保留在 train_images_extracted_white_bg 中存在的
    valid_records = []
    for r in records:
        img_name = os.path.basename(r["image_path"])
        if os.path.exists(os.path.join(ORIGINAL_DIR, img_name)):
            valid_records.append(r)

    print(f"有效记录数: {len(valid_records)} / {len(records)}")

    yes_c = sum(1 for r in valid_records if r.get("output", "").startswith("是"))
    no_c = len(valid_records) - yes_c
    print(f"  是: {yes_c} | 否: {no_c}")

    # Dry run
    if args.dry_run:
        print("\n--- 预览前3条 ---")
        for i, r in enumerate(valid_records[:3]):
            label = "是" if r["output"].startswith("是") else "否"
            print(f"\n[{i}] 原始 ({label}):\n  {r['output'][:80]}...")
            aug_files = get_augmented_files(
                os.path.basename(r["image_path"]), AUGMENTED_DIR)
            print(f"  增强图: {len(aug_files)} 张")
            if aug_files:
                rewritten = call_deepseek(r["output"], label, model=args.model)
                print(f"  改写1:\n  {rewritten[:80]}...")
        return

    # 并行改写
    all_records = []
    success_count = 0
    fail_count = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_single, i, r, AUGMENTED_DIR, args.model): i
            for i, r in enumerate(valid_records)
        }

        for future in tqdm(concurrent.futures.as_completed(futures),
                           total=len(futures), desc="改写进度"):
            idx, results, n_aug = future.result()
            all_records.extend(results)
            success_count += 1

    # 打乱
    import random
    random.seed(42)
    random.shuffle(all_records)

    # 保存
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n完成!")
    print(f"  总记录数: {len(all_records)}")
    print(f"  原始图: {sum(1 for r in all_records if '_aug' not in r['image_path'])}")
    print(f"  增强图: {sum(1 for r in all_records if '_aug' in r['image_path'])}")
    print(f"  输出: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
