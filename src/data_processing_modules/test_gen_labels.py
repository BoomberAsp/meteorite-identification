"""
使用视觉大模型 (如 GPT-4V、Qwen-VL 等) 对陨石图片进行推理，
生成详细的判断理由，用于数据增强或模型蒸馏。

支持 OpenAI 格式的 API 调用（兼容 GPT-4V、qwen-vl-max 等）
"""

import os
import json
import base64
from typing import List, Optional, Dict, Any
from pathlib import Path
from tqdm import tqdm
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================== 配置区域 ====================

# TODO: 填写你的 API 配置
API_KEY = "sk-sp-062e57d4c0a1499ba21bfb7e35ce3388"  # 例如："sk-xxx" 或 "qwen-xxx"
API_BASE_URL = "https://coding.dashscope.aliyuncs.com/v1"  # 例如："https://api.openai.com/v1" 或 "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL_NAME = "qwen3.5-plus"  # 例如："gpt-4o" 或 "qwen-vl-max-latest"

# 图片目录
DATA_ROOT = os.path.join(os.path.dirname(__file__), "../../data")
TRAIN_IMAGES_DIR = os.path.join(DATA_ROOT, "train_images")

# 输出文件
OUTPUT_JSONL = os.path.join(DATA_ROOT, "vl_model_reasoning.jsonl")

# 每个类别的处理数量（用于测试）
MAX_SAMPLES_PER_CLASS = 5  # 设为 None 处理全部

# =================================================


def image_to_base64(image_path: str) -> str:
    """将图片转换为 base64 编码"""
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")

    # 检测图片格式
    ext = Path(image_path).suffix.lower()
    if ext == ".jpg" or ext == ".jpeg":
        mime_type = "image/jpeg"
    elif ext == ".png":
        mime_type = "image/png"
    elif ext == ".webp":
        mime_type = "image/webp"
    else:
        mime_type = "image/jpeg"  # 默认

    return f"data:{mime_type};base64,{image_data}"


def call_vl_api(
    image_path: str,
    api_key: str,
    api_base: str,
    model: str,
    prompt: Optional[str] = None,
    known_label: Optional[int] = None
) -> Dict[str, Any]:
    """
    调用视觉语言模型 API

    Args:
        image_path: 图片路径
        api_key: API 密钥
        api_base: API 基础 URL
        model: 模型名称
        prompt: 自定义提示词（可选）
        known_label: 已知标签（0=非陨石，1=陨石），如果提供则让模型解释原因

    Returns:
        包含推理结果的字典
    """
    # 根据是否有已知标签，使用不同的提示词
    if prompt is None:
        if known_label is not None:
            # 已知答案模式：让模型解释原因
            label_text = "是陨石" if known_label == 1 else "不是陨石"
            prompt = f"""请仔细观察这张图片中的岩石样本。已知该样本的科学分类为：{label_text}。

【任务】
请分析该样本的视觉特征，解释为什么它{'属于' if known_label == 1 else '不属于'}的科学分类：

仅基于以下视觉特征进行解释：
- 熔壳：是否有黑色/深褐色的玻璃质熔化外壳
- 气印：是否有像手指按压面团般的凹坑
- 颜色：颜色特征是什么
- 纹理：表面纹理、晶体结构、金属光泽等
- 形状：形状特征，是否有流线型或圆润边缘

【注意】
- 忽略图片中的所有文字、标签、标尺等信息
- 仅根据岩石本身的外观特征解释

请按照以下 JSON 格式回答（不要添加额外内容）：
{{
    "has_meteorite": {str(known_label == 1).lower()},
    "confidence": 0.0-1.0,
    "reasons": ["理由 1", "理由 2", ...],
    "features_observed": {{
        "color": "描述",
        "texture": "描述",
        "shape": "描述",
        "other": "描述"
    }}
}}"""
        else:
            # 未知答案模式：让模型自己判断
            prompt = """请仔细观察这张图片中的岩石样本，完成以下任务：

【重要要求】
1. 忽略图片中的所有文字、标签、标尺、水印等信息，仅根据岩石本身的视觉特征进行判断
2. 假设你不知道任何关于样本来源的信息，纯粹从外观判断

【判断任务】
1. 判断图片中是否有陨石
2. 详细说明你的判断理由，仅基于以下视觉特征：
   - 熔壳：是否有黑色/深褐色的玻璃质熔化外壳
   - 气印：是否有像手指按压面团般的凹坑
   - 颜色：是否有金属灰色、深褐色或氧化后的红褐色
   - 纹理：是否有球粒结构、金属光泽或铁镍基质
   - 形状：是否有经过大气层烧蚀的流线型或圆润边缘

【注意】
- 不要提及图片中的文字、编号、标签、博物馆名称等信息
- 这些外部信息对判断没有帮助，你必须仅靠岩石外观特征判断

请按照以下 JSON 格式回答（不要添加额外内容）：
{
    "has_meteorite": true/false,
    "confidence": 0.0-1.0,
    "reasons": ["理由 1", "理由 2", ...],
    "features_observed": {
        "color": "描述",
        "texture": "描述",
        "shape": "描述",
        "other": "描述"
    }
}"""

    # 编码图片
    image_base64 = image_to_base64(image_path)

    # 构造 OpenAI 格式的请求
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_base64
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ],
        "max_tokens": 1024,
        "temperature": 0.7
    }

    # 发送请求
    try:
        import requests
        response = requests.post(
            f"{api_base}/chat/completions",
            headers=headers,
            json=payload,
            timeout=60
        )
        response.raise_for_status()

        result = response.json()
        content = result["choices"][0]["message"]["content"]

        # 尝试解析 JSON
        try:
            # 清理可能的 markdown 格式
            if content.startswith("```json"):
                content = content.replace("```json", "").replace("```", "").strip()
            elif content.startswith("```"):
                content = content.replace("```", "").strip()

            parsed = json.loads(content)
            return {
                "success": True,
                "image_path": image_path,
                "prediction": parsed,
                "raw_response": content
            }
        except json.JSONDecodeError as e:
            logger.warning(f"JSON 解析失败：{e}, 原始回复：{content[:200]}...")
            return {
                "success": True,
                "image_path": image_path,
                "prediction": {"raw_text": content},
                "raw_response": content,
                "parse_error": str(e)
            }

    except Exception as e:
        logger.error(f"API 调用失败：{e}")
        return {
            "success": False,
            "image_path": image_path,
            "error": str(e)
        }


def get_labeled_images(csv_path: str, max_per_class: Optional[int] = None) -> List[Dict[str, str]]:
    """
    从 CSV 标签文件中获取带标签的图片路径

    Returns:
        [{"id": "000001.jpg", "label": 1, "path": "..."}, ...]
    """
    import pandas as pd

    df = pd.read_csv(csv_path)

    # 按类别分组采样
    samples = []
    for label in [0, 1]:
        label_df = df[df["label"] == label]
        if max_per_class:
            label_df = label_df.head(max_per_class)

        for _, row in label_df.iterrows():
            samples.append({
                "id": row["id"],
                "label": int(row["label"]),
                "path": os.path.join(TRAIN_IMAGES_DIR, row["id"])
            })

    return samples


def main():
    """主函数"""
    print("=" * 60)
    print("🔭 视觉大模型陨石推理")
    print("=" * 60)

    # 检查配置
    if not API_KEY:
        print("\n⚠️  请先在脚本中配置 API_KEY")
        return
    if not API_BASE_URL:
        print("\n⚠️  请先在脚本中配置 API_BASE_URL")
        return
    if not MODEL_NAME:
        print("\n⚠️  请先在脚本中配置 MODEL_NAME")
        return

    # 检查图片目录
    if not os.path.exists(TRAIN_IMAGES_DIR):
        print(f"\n❌ 图片目录不存在：{TRAIN_IMAGES_DIR}")
        return

    # 加载带标签的图片
    csv_path = os.path.join(DATA_ROOT, "train_labels.csv")
    if not os.path.exists(csv_path):
        print(f"\n❌ 标签文件不存在：{csv_path}")
        return

    samples = get_labeled_images(csv_path, MAX_SAMPLES_PER_CLASS)
    print(f"\n📂 将处理 {len(samples)} 张图片:")
    print(f"   - 陨石样本：{sum(1 for s in samples if s['label'] == 1)}")
    print(f"   - 非陨石样本：{sum(1 for s in samples if s['label'] == 0)}")

    # 处理图片
    results = []
    for sample in tqdm(samples, desc="推理中"):
        result = call_vl_api(
            image_path=sample["path"],
            api_key=API_KEY,
            api_base=API_BASE_URL,
            model=MODEL_NAME,
            known_label=sample["label"]  # 传入已知标签，让模型解释原因
        )
        result["ground_truth_label"] = sample["label"]
        results.append(result)

        # 每处理 5 张保存一次，防止中断丢失
        if len(results) % 5 == 0:
            with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
                for r in results:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 保存结果
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n✅ 推理完成，结果保存至：{OUTPUT_JSONL}")

    # 统计成功率
    success_count = sum(1 for r in results if r.get("success"))
    print(f"   成功：{success_count}/{len(results)}")

    # 打印一个样例
    if results:
        print("\n📋 样例输出:")
        sample_result = results[0]
        if sample_result.get("success"):
            print(json.dumps(sample_result.get("prediction"), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
