"""
api_rewrite_reasoning.py

使用 DashScope API 并行改写 SFT 数据的理由文本。
每条原始 reasoning 生成多个改写版本，保持"是/否"结论不变。

用法:
    python api_rewrite_reasoning.py -i input.jsonl -o output.jsonl -n 2 --workers 8
    python api_rewrite_reasoning.py --dry-run   # 预览前3条
"""

import os
import re
import json
import argparse
import concurrent.futures
from tqdm import tqdm

PROJECT_ROOT = "/root/Projects/meteorite-identification"
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")

# DashScope API Key（从环境变量读取，也可硬编码）
API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")

REWRITE_SYSTEM = (
    "你是一个地质学专家。你的任务是用不同的专业术语和句式重写石头判断理由，"
    "保持结论（是/否）不变。只输出改写后的理由，不要添加任何解释。"
)

REWRITE_PROMPTS = {
    "是": (
        "以下是一段关于某块石头被判断为陨石的推理理由。\n"
        "请保持结论\"是\"不变，用不同的专业术语和句式重新表述。\n"
        "要求：保持原有判断依据（如熔壳、气印、球粒等），长度相近。\n"
        "格式与原文一致，以\"是。理由：\"开头。\n\n"
        "原始：\n{original}"
    ),
    "否": (
        "以下是一段关于某块石头被判断为非陨石的推理理由。\n"
        "请保持结论\"否\"不变，用不同的专业术语和句式重新表述。\n"
        "要求：保持原有判断依据（如无熔壳、地球岩石特征等），长度相近。\n"
        "格式与原文一致，以\"否。理由：\"开头。\n\n"
        "原始：\n{original}"
    ),
}


def call_dashscope(text: str, label: str, model: str = "qwen3.6-plus") -> str:
    """调用 DashScope API 改写文本"""
    try:
        import dashscope
        from dashscope import Generation
        dashscope.api_key = API_KEY

        prompt = REWRITE_PROMPTS[label].format(original=text)

        response = Generation.call(
            model=model,
            messages=[
                {"role": "system", "content": REWRITE_SYSTEM},
                {"role": "user", "content": prompt}
            ],
            max_tokens=512,
            temperature=0.7,
            top_p=0.9,
        )

        if response.status_code == 200 and response.output.choices:
            result = response.output.choices[0].message.content.strip()
            # 验证以"是"或"否"开头
            if result.startswith("是") or result.startswith("否"):
                return result
            # 尝试提取开头
            match = re.match(r"^[是否][。:.，,]?\s*(.*)", result)
            if match:
                return f"{label}。理由：{match.group(1)}"
        return text  # 失败返回原文本

    except Exception as e:
        print(f"  API 调用失败: {e}")
        return text


def rewrite_batch(batch, label, n_rewrites):
    """批量改写一组样本"""
    results = []
    for idx, original_text in batch:
        rewrites = []
        for i in range(n_rewrites):
            rewritten = call_dashscope(original_text, label)
            rewrites.append(rewritten)
        results.append((idx, original_text, rewrites))
    return results


def main():
    parser = argparse.ArgumentParser(description="DashScope API 批量改写理由文本")
    parser.add_argument("-i", "--input",
                        default=os.path.join(DATA_ROOT, "merged", "training_data_merged.jsonl"),
                        help="输入 JSONL 文件路径")
    parser.add_argument("-o", "--output",
                        default=os.path.join(DATA_ROOT, "merged", "training_data_rewritten.jsonl"),
                        help="输出 JSONL 文件路径")
    parser.add_argument("-n", "--rewrite-count", type=int, default=2,
                        help="每条数据生成的改写版本数")
    parser.add_argument("-w", "--workers", type=int, default=8,
                        help="并行工作线程数")
    parser.add_argument("-m", "--model", default="qwen3.6-plus",
                        help="DashScope 模型名称")
    parser.add_argument("--dry-run", action="store_true",
                        help="预览前3条，不写入文件")
    args = parser.parse_args()

    if not API_KEY:
        print("⚠️  未设置 DASHSCOPE_API_KEY 环境变量")
        print("   export DASHSCOPE_API_KEY='sk-xxx'")
        return

    print("=" * 60)
    print("DashScope API 批量改写理由文本")
    print(f"  模型: {args.model}")
    print(f"  每条改写数: {args.rewrite_count}")
    print(f"  并行数: {args.workers}")
    print("=" * 60)

    # 读取输入
    records = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    print(f"输入记录数: {len(records)}")

    # 按 label 分组
    yes_records = [(i, r) for i, r in enumerate(records)
                   if r.get("output", "").startswith("是")]
    no_records = [(i, r) for i, r in enumerate(records)
                  if r.get("output", "").startswith("否")]
    print(f"  是: {len(yes_records)} | 否: {len(no_records)}")

    # Dry run
    if args.dry_run:
        print("\n--- 预览 ---")
        for idx, r in records[:3]:
            label = "是" if r["output"].startswith("是") else "否"
            print(f"\n原始 ({label}):\n  {r['output'][:100]}...")
        return

    # 改写
    output_records = []
    total_rewrites = len(records) * args.rewrite_count

    def process_record(idx_record):
        idx, record = idx_record
        output_text = record.get("output", "")
        label = "是" if output_text.startswith("是") else "否"
        rewrites = []
        for i in range(args.rewrite_count):
            rewritten = call_dashscope(output_text, label, model=args.model)
            rewrites.append(rewritten)
        return (idx, record, rewrites)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = list(tqdm(
            executor.map(process_record, enumerate(records)),
            total=len(records),
            desc="改写进度"
        ))

        for idx, record, rewrites in futures:
            # 原始记录
            output_records.append(record)
            # 改写记录
            for i, rewritten in enumerate(rewrites):
                rew_record = record.copy()
                rew_record["output"] = rewritten
                rew_record["source_index"] = record.get("source_index", idx + 1) * 100 + i + 1
                output_records.append(rew_record)

    # 保存
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for r in output_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n完成! 输出 {len(output_records)} 条")
    print(f"  原始: {len(records)}")
    print(f"  改写: {len(output_records) - len(records)}")
    print(f"  输出: {args.output}")


if __name__ == "__main__":
    main()
