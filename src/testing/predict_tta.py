"""
predict_tta.py

测试时增强（TTA）推理模块。
对每张测试图施加多种变换（多尺度 + 水平翻转），
对 "是"/"否" token 的 logits 取平均后判定。

替代 test_model.py 中的 predict() 函数。
"""

import torch
from PIL import Image
from tqdm import tqdm


# TTA 配置：5 种变换版本
TTA_CONFIGS = [
    {"scale": 720, "flip_h": False, "label": "orig-720"},
    {"scale": 512, "flip_h": False, "label": "orig-512"},
    {"scale": 384, "flip_h": False, "label": "orig-384"},
    {"scale": 720, "flip_h": True,  "label": "flip-720"},
    {"scale": 512, "flip_h": True,  "label": "flip-512"},
]


def predict_with_tta(model, processor, test_loader, device,
                     tta_configs=TTA_CONFIGS,
                     instruction=None):
    """
    带 TTA 的预测：对每张图的多种变换版本分别推理，
    取 "是"/"否" token 的 softmax 概率平均后判定。

    Args:
        model: 已加载 LoRA 的模型
        processor: 处理器
        test_loader: DataLoader，每个 batch 为 (filename, PIL.Image) 列表
        device: 设备
        tta_configs: TTA 变换配置列表
        instruction: 推理提示词，默认使用与训练一致的格式

    Returns:
        dict {filename: label}
    """
    if instruction is None:
        instruction = (
            "请观察这张图片中石头的视觉特征（颜色、纹理、形状），"
            "判断其中是否有陨石？注意：忽略图片中的所有文字、标签、标尺等信息，"
            "仅根据石头本身的外观特征判断。回答时首先给出“是”或“否”，"
            "然后简要说明理由。"
        )

    model.eval()

    # 找到 "是" 和 "否" 的 token ID
    yes_token_id = processor.tokenizer.encode("是", add_special_tokens=False)[0]
    no_token_id = processor.tokenizer.encode("否", add_special_tokens=False)[0]

    # 文件名 -> 累计 "是" 的 log 概率（logits 差值近似）
    filename_scores = {}

    with torch.no_grad():
        for config in tta_configs:
            desc = f"TTA {config['label']} (scale={config['scale']}, flip={config['flip_h']})"
            for batch in tqdm(test_loader, desc=desc):
                filenames, images = zip(*batch)
                filenames = list(filenames)
                scaled_images = []
                for img in images:
                    img = img.copy()
                    if config["flip_h"]:
                        img = img.transpose(Image.FLIP_LEFT_RIGHT)
                    img.thumbnail((config["scale"], config["scale"]), Image.Resampling.LANCZOS)
                    scaled_images.append(img)

                # 构建输入
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": None},
                            {"type": "text", "text": instruction}
                        ]
                    }
                ]
                texts = [
                    processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True)
                    for _ in range(len(images))
                ]

                inputs = processor(
                    text=texts,
                    images=scaled_images,
                    return_tensors="pt",
                    padding=True
                ).to(device)

                # 获取模型输出 logits
                outputs = model(**inputs)
                logits = outputs.logits  # [B, seq_len, vocab]

                # 对每个样本，找到生成的第一个 token 位置，取 "是"/"否" 的 logits 差值
                input_len = inputs["input_ids"].shape[1]
                for i, fn in enumerate(filenames):
                    # 取生成序列的第一个 token 位置的 logits
                    first_gen_pos = input_len  # 第一个生成的 token
                    token_logits = logits[i, first_gen_pos, :]  # [vocab]

                    yes_logit = token_logits[yes_token_id].item()
                    no_logit = token_logits[no_token_id].item()

                    if fn not in filename_scores:
                        filename_scores[fn] = 0.0

                    # 累加 logit 差值（相当于 log 概率比的近似）
                    filename_scores[fn] += (yes_logit - no_logit)

                # 每个 config 之间清理显存
                del inputs, outputs, logits
                torch.cuda.empty_cache()

    # 平均后判定
    id_to_pred = {}
    for fn, score in filename_scores.items():
        avg_score = score / len(tta_configs)
        id_to_pred[fn] = 1 if avg_score > 0 else 0

    return id_to_pred


def predict_no_tta(model, processor, test_loader, device, instruction=None):
    """
    不使用 TTA 的推理（与原有 predict() 兼容），
    使用 generate 方法，按规则提取答案。
    """
    if instruction is None:
        instruction = (
            "请观察这张图片中石头的视觉特征（颜色、纹理、形状），"
            "判断其中是否有陨石？注意：忽略图片中的所有文字、标签、标尺等信息，"
            "仅根据石头本身的外观特征判断。回答时首先给出“是”或“否”，"
            "然后简要说明理由。"
        )

    model.eval()
    id_to_pred = {}

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting"):
            filenames, images = zip(*batch)
            filenames = list(filenames)
            scaled_images = []
            for img in images:
                img.thumbnail((720, 720), Image.Resampling.LANCZOS)
                scaled_images.append(img)

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": None},
                        {"type": "text", "text": instruction}
                    ]
                }
            ]
            texts = [
                processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True)
                for _ in range(len(images))
            ]

            inputs = processor(
                text=texts,
                images=scaled_images,
                return_tensors="pt",
                padding=True
            ).to(device)

            generated_ids = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
            )

            for filename, gen_id in zip(filenames, generated_ids):
                generated_text = processor.decode(
                    gen_id[inputs['input_ids'].shape[1]:], skip_special_tokens=True)
                # 答案在前格式："是。理由：..." / "否。理由：..."
                if generated_text.startswith("是"):
                    pred_label = 1
                elif generated_text.startswith("否"):
                    pred_label = 0
                else:
                    pred_label = 1 if "是" in generated_text else 0
                id_to_pred[filename] = pred_label

    return id_to_pred
