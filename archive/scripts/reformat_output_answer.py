"""
reformat_output_answer.py

将 merged JSONL 中 assistant 回答的格式从"先答案后理由"改为"先理由后答案"。

变更前 output 格式:
    "是。理由：这块石头具有..."
    "否。理由：这块石头的颜色..."

变更后 output 格式:
    "理由：这块石头具有... 因此判断为：是。"
    "理由：这块石头的颜色... 因此判断为：否。"

同时更新 instruction 字段以匹配新的回答格式要求。
"""

import os
import re
import json
import argparse


NEW_INSTRUCTION = (
    '请观察这张图片中石头的视觉特征（颜色、纹理、形状），'
    '判断其中是否有陨石？注意：忽略图片中的所有文字、标签、标尺等信息，'
    '仅根据石头本身的外观特征判断。请详细分析陨石的典型鉴别特征（熔壳、气印、'
    '颜色、纹理、形状等），先给出推理过程和理由，最后在末尾明确回答'
    '”是”或”否”。'
)


def reformat_output_text(text: str, label: str) -> str:
    """
    提取理由部分并重组为"理由... 因此判断为：{label}。"格式。

    能处理的输入模式:
    - "是。理由：..."
    - "否。理由：..."
    - "是。 理由：..."
    - "否。\n理由：..."
    - "是，理由：..."

    如果无法匹配模式，返回原始文本。
    """
    # 匹配 "是/否" 开头 + 分隔符 + "理由："
    pattern = r"^[是否]\s*[。,\.\n，]+\s*理由[：:]\s*(.+)$"
    match = re.match(pattern, text, re.DOTALL)
    if not match:
        return text

    reason = match.group(1).strip()
    # 清理理由末尾的句号，统一由新格式控制
    reason = reason.rstrip("。")
    return f"理由：{reason} 因此判断为：{label}。"


def process(input_path: str, output_path: str, dry_run: bool = False):
    records = []
    skipped = 0

    with open(input_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            record = json.loads(line)
            old_output = record.get("output", "")
            label = "是" if old_output.startswith("是") else "否"

            new_output = reformat_output_text(old_output, label)
            if new_output == old_output:
                skipped += 1

            record["output"] = new_output
            record["instruction"] = NEW_INSTRUCTION
            records.append(record)

            if dry_run and line_num <= 3:
                print(f"--- 样本 {line_num} ---")
                print(f"  旧: {old_output[:80]}...")
                print(f"  新: {new_output[:80]}...")

    if dry_run:
        print(f"\n总计 {len(records)} 条，{skipped} 条未能自动重格式化")
        return

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"处理完成: {len(records)} 条")
    print(f"  成功重格式化: {len(records) - skipped} 条")
    print(f"  跳过 (未匹配模式): {skipped} 条")
    print(f"  输出文件: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="重格式化 SFT 数据的 answer 顺序")
    parser.add_argument("input", help="输入 JSONL 文件路径")
    parser.add_argument("-o", "--output", default=None,
                        help="输出 JSONL 文件路径（默认: {input}.reformatted）")
    parser.add_argument("--dry-run", action="store_true",
                        help="仅打印前 3 条变更预览，不写入文件")
    args = parser.parse_args()

    output = args.output or args.input + ".reformatted"
    process(args.input, output, dry_run=args.dry_run)
