"""
陨石图片数据集转换脚本
将图片 + labels.csv 转换为带模型理由的训练数据 CSV 文件
使用多模态模型生成每张图片对应的「是/否 + 理由」
"""

import os
import csv
import json
import base64
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock, Semaphore
from openai import OpenAI

# ─── 配置 ────────────────────────────────────────────────────────────────────
IMAGE_DIR    = "./test/images"        # 图片目录
LABELS_CSV   = "./test/labels.csv"        # 标签文件
OUTPUT_CSV   = "./test/training_data_from_2501.csv" # 输出文件
MODEL        = "moonshotai/kimi-k2.6"
BASE_URL     = "https://openrouter.ai/api/v1"  # 第三方中转平台
INSTRUCTION  = "请观察这张图片中石头的视觉特征（颜色、纹理、形状），判断其中是否有陨石？注意：忽略图片中的所有文字、标签、标尺等信息，仅根据石头本身的外观特征判断。回答时首先给出”是”或”否”，然后简要说明理由。"
RETRY_LIMIT  = 3                     # 单张图片最大重试次数
RETRY_DELAY  = 5                     # 重试间隔（秒）
SLEEP_BETWEEN = 1.0                  # 每个 key 两次请求之间的最小间隔（秒，避免限流）
START_INDEX  = 2501                  # 从 labels.csv 的第几条数据开始生成（从 1 开始计数）
MAX_SAMPLES  = None                  # 最多生成多少条；None 表示从 START_INDEX 生成到文件末尾
PER_KEY_CONCURRENCY = 3              # 每个 API key 同时处理的请求数，建议先保持 1
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


def call_vl_model(client: OpenAI, image_path: str, label: int) -> str:
    """调用多模态模型，返回模型生成的「是/否 + 理由」字符串"""
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


def load_labels(labels_csv: str) -> list[tuple[str, int]]:
    """读取 labels.csv，按文件顺序返回 [(filename, label), ...]"""
    labels = []
    with open(labels_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row["id"].strip().strip('"')
            label    = int(row["label"].strip().strip('"'))
            labels.append((filename, label))
    return labels


def load_api_keys() -> list[str]:
    """从 OPENAI_API_KEYS 或 OPENAI_API_KEY 读取 API key。"""
    keys_text = os.getenv("OPENAI_API_KEYS")
    if keys_text:
        return [key.strip() for key in keys_text.split(",") if key.strip()]

    single_key = os.getenv("OPENAI_API_KEY")
    return [single_key.strip()] if single_key else []


def load_partial_results(partial_jsonl: str) -> dict[int, dict]:
    """读取已完成的 partial 结果，便于中断后继续。"""
    results = {}
    if not os.path.exists(partial_jsonl):
        return results

    with open(partial_jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            results[int(row["source_index"])] = row
    return results


def append_jsonl(path: str, row: dict, lock: Lock) -> None:
    with lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def process_one(
    client: OpenAI,
    semaphore: Semaphore,
    key_index: int,
    idx: int,
    total: int,
    filename: str,
    label: int,
) -> dict | None:
    image_path = os.path.join(IMAGE_DIR, filename)

    if not os.path.isfile(image_path):
        print(f"[{idx}/{total}] 图片不存在，跳过：{image_path}")
        return None

    with semaphore:
        print(f"[{idx}/{total}] key#{key_index + 1} 处理：{filename}（label={label}）")
        output_text = call_vl_model(client, image_path, label)
        print(f"[{idx}/{total}] 完成：{output_text[:80]}{'...' if len(output_text) > 80 else ''}")
        time.sleep(SLEEP_BETWEEN)

    return {
        "source_index": idx,
        "image_path": image_path,
        "instruction": INSTRUCTION,
        "output": output_text,
    }


def main():
    # 初始化 OpenAI 客户端（从环境变量读取 OPENAI_API_KEYS 或 OPENAI_API_KEY）
    api_keys = load_api_keys()
    if not api_keys:
        raise EnvironmentError(
            "未找到 API key，请先设置其中一种：\n"
            "  export OPENAI_API_KEYS='sk-or-v1-key1,sk-or-v1-key2'\n"
            "  export OPENAI_API_KEY='sk-or-v1-key1'"
        )

    clients = [
        OpenAI(
            api_key=api_key,
            base_url=BASE_URL,
            default_headers={
                "HTTP-Referer": "http://localhost",
                "X-OpenRouter-Title": "meteorite-data-generation",
            },
        )
        for api_key in api_keys
    ]
    semaphores = [Semaphore(PER_KEY_CONCURRENCY) for _ in api_keys]
    max_workers = max(1, len(api_keys) * PER_KEY_CONCURRENCY)
    print(f"🔑 已加载 {len(api_keys)} 个 API key，并发 worker 数：{max_workers}")

    # 加载标签
    print(f"📂 加载标签文件：{LABELS_CSV}")
    labels = load_labels(LABELS_CSV)
    print(f"   共找到 {len(labels)} 条标签记录。")

    total = len(labels)
    start_offset = max(START_INDEX - 1, 0)
    end_offset = None if MAX_SAMPLES is None else start_offset + MAX_SAMPLES
    selected_labels = labels[start_offset:end_offset]

    partial_jsonl = OUTPUT_CSV.replace(".csv", ".partial.jsonl")
    completed_results = load_partial_results(partial_jsonl)
    completed_indexes = set(completed_results)

    tasks = []
    for offset, (filename, label) in enumerate(selected_labels):
        idx = START_INDEX + offset
        if idx in completed_indexes:
            continue
        tasks.append((idx, filename, label))

    print(
        f"   本次区间从第 {START_INDEX} 条开始，共 {len(selected_labels)} 条；"
        f"已完成 {len(completed_indexes)} 条，待生成 {len(tasks)} 条。"
    )
    print(f"   partial 结果会实时写入：{partial_jsonl}")

    write_lock = Lock()
    generated_results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for task_pos, (idx, filename, label) in enumerate(tasks):
            key_index = task_pos % len(clients)
            futures.append(
                executor.submit(
                    process_one,
                    clients[key_index],
                    semaphores[key_index],
                    key_index,
                    idx,
                    total,
                    filename,
                    label,
                )
            )

        for future in as_completed(futures):
            row = future.result()
            if row is None:
                continue
            generated_results.append(row)
            append_jsonl(partial_jsonl, row, write_lock)

    all_results = list(completed_results.values()) + generated_results
    all_results.sort(key=lambda row: int(row["source_index"]))

    # 写入 CSV
    print(f"\n💾 写入训练数据到：{OUTPUT_CSV}")
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["source_index", "image_path", "instruction", "output"]
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(all_results)

    print(f"✅ 完成！共生成/汇总 {len(all_results)} 条训练数据。")

    # 同时输出一个 JSON Lines 文件（便于后续微调框架使用）
    jsonl_path = OUTPUT_CSV.replace(".csv", ".jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for row in all_results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"✅ 同时输出 JSONL 格式：{jsonl_path}")


if __name__ == "__main__":
    main()