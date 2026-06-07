"""
修复 training_data JSONL 中模型调用失败的样本。

默认读取 test/training_data_from_2501.partial.jsonl，查找 output 中包含
"（模型调用失败，无法生成理由）" 的行，按 source_index 从 labels.csv 找回标签，
重新调用多模态模型生成理由，并更新对应 JSONL 行。

用法：
  export OPENAI_API_KEYS='sk-or-v1-key1,sk-or-v1-key2'
  python3 src/data_processing_modules/repair_failed_training_data.py

建议不要在主生成脚本仍然写同一个 partial.jsonl 时运行本脚本。
"""

import argparse
import base64
import csv
import json
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Semaphore

from openai import OpenAI


DEFAULT_JSONL = "./test/training_data_from_2501.partial.jsonl"
DEFAULT_LABELS_CSV = "./test/labels.csv"
DEFAULT_IMAGE_DIR = "./test/images"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = os.getenv("OPENROUTER_MODEL", "moonshotai/kimi-k2.6")
FAILED_MARKER = "（模型调用失败，无法生成理由）"
VALID_ENDING = "。"
INSTRUCTION = "请观察这张图片中石头的视觉特征（颜色、纹理、形状），判断其中是否有陨石？注意：忽略图片中的所有文字、标签、标尺等信息，仅根据石头本身的外观特征判断。回答时首先给出”是”或”否”，然后简要说明理由。"

RETRY_LIMIT = 4
RETRY_DELAY = 8
SLEEP_BETWEEN = 1.0
PER_KEY_CONCURRENCY = 3


def encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_image_media_type(image_path: str) -> str:
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
    if label == 1:
        constraint = (
            "注意：这张图片已被专家确认为陨石图像，请以此为前提，"
            "详细描述图片中哪些视觉特征（如熔壳、气印、不规则形状、颜色、纹理等）"
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
        "请严格按照以下格式输出，不要输出其他内容：\n"
        f"{answer_prefix}。理由：[你的详细理由]"
    )


def load_api_keys() -> list[str]:
    keys_text = os.getenv("OPENAI_API_KEYS")
    if keys_text:
        return [key.strip() for key in keys_text.split(",") if key.strip()]

    single_key = os.getenv("OPENAI_API_KEY")
    return [single_key.strip()] if single_key else []


def load_labels(labels_csv: str) -> list[tuple[str, int]]:
    labels = []
    with open(labels_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row["id"].strip().strip('"')
            label = int(row["label"].strip().strip('"'))
            labels.append((filename, label))
    return labels


def read_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row["__line_no"] = line_no
            rows.append(row)
    return rows


def write_jsonl(path: str, rows: list[dict]) -> None:
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        for row in rows:
            clean_row = {k: v for k, v in row.items() if not k.startswith("__")}
            f.write(json.dumps(clean_row, ensure_ascii=False) + "\n")
    os.replace(tmp_path, path)


def is_failed(row: dict, also_short: bool, min_chars: int) -> bool:
    output = (row.get("output") or "").strip()
    if FAILED_MARKER in output:
        return True
    if output and not output.endswith(VALID_ENDING):
        return True
    if also_short and len(output) < min_chars:
        return True
    return False


def call_vl_model(client: OpenAI, model: str, image_path: str, label: int) -> str | None:
    image_b64 = encode_image_to_base64(image_path)
    media_type = get_image_media_type(image_path)
    prompt = build_prompt(label)

    for attempt in range(1, RETRY_LIMIT + 1):
        try:
            response = client.chat.completions.create(
                model=model,
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
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
                max_tokens=512,
                temperature=0.2,
            )
            content = response.choices[0].message.content
            content = content.strip() if content else ""
            if not content:
                raise ValueError("empty model content")
            if FAILED_MARKER in content:
                raise ValueError("model returned failure marker")
            if not content.endswith(VALID_ENDING):
                raise ValueError("model output does not end with Chinese period")
            return content
        except Exception as e:
            print(f"    尝试 {attempt}/{RETRY_LIMIT} 失败：{e}")
            if attempt < RETRY_LIMIT:
                time.sleep(RETRY_DELAY * attempt)

    return None


def repair_one(
    row: dict,
    labels: list[tuple[str, int]],
    image_dir: str,
    client: OpenAI,
    model: str,
    semaphore: Semaphore,
    key_index: int,
) -> tuple[int, str | None]:
    source_index = int(row["source_index"])
    label_pos = source_index - 1
    if label_pos < 0 or label_pos >= len(labels):
        print(f"[{source_index}] labels.csv 中找不到对应行，跳过")
        return source_index, None

    filename, label = labels[label_pos]
    image_path = os.path.join(image_dir, filename)
    if not os.path.isfile(image_path):
        print(f"[{source_index}] 图片不存在：{image_path}")
        return source_index, None

    with semaphore:
        print(f"[{source_index}] key#{key_index + 1} 重新生成：{filename}（label={label}）")
        output = call_vl_model(client, model, image_path, label)
        time.sleep(SLEEP_BETWEEN)

    return source_index, output


def main() -> None:
    parser = argparse.ArgumentParser(description="重新修复 JSONL 中模型调用失败的训练数据。")
    parser.add_argument("--jsonl", default=DEFAULT_JSONL, help="需要修复的 JSONL 文件")
    parser.add_argument("--labels-csv", default=DEFAULT_LABELS_CSV, help="labels.csv 路径")
    parser.add_argument("--image-dir", default=DEFAULT_IMAGE_DIR, help="图片目录")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="OpenRouter/OpenAI compatible base_url")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="用于重试的模型")
    parser.add_argument("--max-fixes", type=int, default=None, help="最多修复多少条，默认修复全部失败行")
    parser.add_argument("--also-short", action="store_true", help="同时重试特别短的输出")
    parser.add_argument("--min-chars", type=int, default=30, help="--also-short 的最短字符阈值")
    parser.add_argument("--no-backup", action="store_true", help="不生成 .bak 备份")
    args = parser.parse_args()

    api_keys = load_api_keys()
    if not api_keys:
        raise EnvironmentError(
            "未找到 API key，请设置：\n"
            "  export OPENAI_API_KEYS='sk-or-v1-key1,sk-or-v1-key2'\n"
            "或：\n"
            "  export OPENAI_API_KEY='sk-or-v1-key1'"
        )

    rows = read_jsonl(args.jsonl)
    labels = load_labels(args.labels_csv)
    failed_rows = [row for row in rows if is_failed(row, args.also_short, args.min_chars)]
    if args.max_fixes is not None:
        failed_rows = failed_rows[: args.max_fixes]

    print(f"读取 {len(rows)} 行；待修复 {len(failed_rows)} 行；模型：{args.model}")
    if not failed_rows:
        print("没有找到需要修复的失败行。")
        return

    if not args.no_backup:
        backup_path = f"{args.jsonl}.bak"
        shutil.copy2(args.jsonl, backup_path)
        print(f"已备份原文件：{backup_path}")

    clients = [
        OpenAI(
            api_key=key,
            base_url=args.base_url,
            default_headers={
                "HTTP-Referer": "http://localhost",
                "X-OpenRouter-Title": "meteorite-data-repair",
            },
        )
        for key in api_keys
    ]
    semaphores = [Semaphore(PER_KEY_CONCURRENCY) for _ in api_keys]
    max_workers = max(1, len(api_keys) * PER_KEY_CONCURRENCY)

    fixed_outputs: dict[int, str] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for pos, row in enumerate(failed_rows):
            key_index = pos % len(clients)
            futures.append(
                executor.submit(
                    repair_one,
                    row,
                    labels,
                    args.image_dir,
                    clients[key_index],
                    args.model,
                    semaphores[key_index],
                    key_index,
                )
            )

        for future in as_completed(futures):
            source_index, output = future.result()
            if output:
                fixed_outputs[source_index] = output
                print(f"[{source_index}] 修复成功：{output[:80]}{'...' if len(output) > 80 else ''}")
            else:
                print(f"[{source_index}] 修复失败，保留原输出")

    for row in rows:
        source_index = int(row.get("source_index", -1))
        if source_index in fixed_outputs:
            row["output"] = fixed_outputs[source_index]

    write_jsonl(args.jsonl, rows)
    print(f"完成：成功修复 {len(fixed_outputs)}/{len(failed_rows)} 条，已更新 {args.jsonl}")


if __name__ == "__main__":
    main()
