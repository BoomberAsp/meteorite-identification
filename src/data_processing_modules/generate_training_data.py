"""
陨石图片数据集转换脚本
将图片 + labels.csv 转换为带模型理由的训练数据 CSV 文件
使用 GPT-4o-mini 多模态模型生成每张图片对应的「是/否 + 理由」
"""

import os
import csv
import json
import base64
import time
from pathlib import Path
from openai import OpenAI

# ─── 配置 ────────────────────────────────────────────────────────────────────
IMAGE_DIR    = "./test/images"        # 图片目录
LABELS_CSV   = "./test/labels.csv"        # 标签文件
OUTPUT_CSV   = "./test/training_data.csv" # 输出文件
MODEL        = "gpt-4o-mini"
BASE_URL     = "https://api.ai-gaochao.cn/v1"  # 第三方中转平台
INSTRUCTION  = "请观察这张图片中石头的视觉特征（颜色、纹理、形状），判断其中是否有陨石？注意：忽略图片中的所有文字、标签、标尺等信息，仅根据石头本身的外观特征判断。回答时首先给出”是”或”否”，然后简要说明理由。"
RETRY_LIMIT  = 3                     # 单张图片最大重试次数
RETRY_DELAY  = 5                     # 重试间隔（秒）
SLEEP_BETWEEN = 1.0                  # 每次请求后的间隔（秒，避免限流）
# ─────────────────────────────────────────────────────────────────────────────


def encode_image_to_base64(image_path: str) -> str:
    """将图片文件编码为 base64 字符串"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_image_media_type(image_path: str) -> str:
    """根据文件扩展名返回 MIME 类型"""
    ext = Path(image_path).suffix.lower()
    mapping = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    return mapping.get(ext, "image/jpeg")


def build_prompt(label: int) -> str:
    """
    根据标签构建带约束的 prompt，确保模型输出与已知标签一致，
    同时生成合理的理由。
    """
    if label == 1:
        constraint = (
            "注意：这张图片已被专家确认为陨石图像，请以此为前提，"
            "详细描述图片中哪些视觉特征（如熔壳、维德曼花纹、不规则形状、颜色、纹理等）"
            "表明这是一块陨石。"
        )
        answer_prefix = "是"
    else:
        constraint = (
            "注意：这张图片已被专家确认为非陨石图像，请以此为前提，"
            "详细描述图片中哪些视觉特征（如形状、颜色、纹理、表面特征等）"
            "表明这不是陨石。"
        )
        answer_prefix = "否"

    return (
        f"{INSTRUCTION}\n\n"
        f"{constraint}\n\n"
        f"请严格按照以下格式输出，不要输出其他内容：\n"
        f"{answer_prefix}。理由：[你的详细理由]"
    )


def call_gpt4o_mini(client: OpenAI, image_path: str, label: int) -> str:
    """调用 GPT-4o-mini，返回模型生成的「是/否 + 理由」字符串"""
    image_b64 = encode_image_to_base64(image_path)
    media_type = get_image_media_type(image_path)
    prompt = build_prompt(label)

    for attempt in range(1, RETRY_LIMIT + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{media_type};base64,{image_b64}",
                                    "detail": "high",
                                },
                            },
                            {
                                "type": "text",
                                "text": prompt,
                            },
                        ],
                    }
                ],
                max_tokens=512,
                temperature=0.3,
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            print(f"  [尝试 {attempt}/{RETRY_LIMIT}] 请求失败：{e}")
            if attempt < RETRY_LIMIT:
                time.sleep(RETRY_DELAY)
            else:
                return f"{'是' if label == 1 else '否'}。理由：（模型调用失败，无法生成理由）"


def load_labels(labels_csv: str) -> dict:
    """读取 labels.csv，返回 {filename: label} 字典"""
    labels = {}
    with open(labels_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row["id"].strip().strip('"')
            label    = int(row["label"].strip().strip('"'))
            labels[filename] = label
    return labels


def main():
    # 初始化 OpenAI 客户端（从环境变量读取 OPENAI_API_KEY）
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "未找到 OPENAI_API_KEY 环境变量，请先设置：\n"
            "  export OPENAI_API_KEY='sk-...'"
        )
    client = OpenAI(api_key=api_key, base_url=BASE_URL)

    # 加载标签
    print(f"📂 加载标签文件：{LABELS_CSV}")
    labels = load_labels(LABELS_CSV)
    print(f"   共找到 {len(labels)} 条标签记录。")

    # 准备输出
    results = []
    total   = len(labels)

    for idx, (filename, label) in enumerate(labels.items(), start=1):
        image_path = os.path.join(IMAGE_DIR, filename)

        if not os.path.isfile(image_path):
            print(f"[{idx}/{total}] ⚠️  图片不存在，跳过：{image_path}")
            continue

        print(f"[{idx}/{total}] 🔍 处理：{filename}（label={label}）")
        output_text = call_gpt4o_mini(client, image_path, label)
        print(f"        ✅ 输出：{output_text[:80]}{'...' if len(output_text) > 80 else ''}")

        results.append({
            "image_path":  image_path,
            "instruction": INSTRUCTION,
            "output":      output_text,
        })

        time.sleep(SLEEP_BETWEEN)

    # 写入 CSV
    print(f"\n💾 写入训练数据到：{OUTPUT_CSV}")
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["image_path", "instruction", "output"]
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(results)

    print(f"✅ 完成！共生成 {len(results)} 条训练数据。")

    # 同时输出一个 JSON Lines 文件（便于后续微调框架使用）
    jsonl_path = OUTPUT_CSV.replace(".csv", ".jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"✅ 同时输出 JSONL 格式：{jsonl_path}")


if __name__ == "__main__":
    main()