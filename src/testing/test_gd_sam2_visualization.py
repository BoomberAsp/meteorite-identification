"""
test_gd_sam2_visualization.py

分别测试 GroundingDINO 和 SAM2.1 的单独效果，输出可视化图片用于对比分析。

输出：
- GroundingDINO 检测结果：在原图上绘制检测框
- SAM2.1 分割结果：使用图像中心点作为 prompt，输出分割掩码
- 组合结果：GroundingDINO 检测框 + SAM2.1 分割掩码
"""

import os
import sys
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional

# 路径配置
PROJECT_ROOT = "/root/Projects/meteorite-identification"
SAM2_ROOT = "/root/Projects/sam2"
GD_MODEL_PATH = os.path.join(PROJECT_ROOT, "models/grounding_dino")
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data/visualization_results")

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_grounding_dino():
    """加载 GroundingDINO 模型"""
    from transformers import AutoModel, AutoProcessor

    print("加载 GroundingDINO...")
    model = AutoModel.from_pretrained(
        GD_MODEL_PATH,
        local_files_only=True,
        trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(
        GD_MODEL_PATH,
        local_files_only=True,
        trust_remote_code=True
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    print(f"GroundingDINO 加载成功 (device: {device})")
    return model, processor, device


def load_sam2():
    """加载 SAM2.1 large 模型"""
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    print("加载 SAM2.1 large...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 切换到 sam2 目录
    original_dir = os.getcwd()
    os.chdir(SAM2_ROOT)

    sam2_model = build_sam2(
        "configs/sam2.1/sam2.1_hiera_l.yaml",
        os.path.join(SAM2_ROOT, "checkpoints/sam2.1_hiera_large.pt"),
        device=device
    )
    predictor = SAM2ImagePredictor(sam2_model)

    os.chdir(original_dir)
    print(f"SAM2.1 large 加载成功 (device: {device})")
    return predictor, device


def detect_with_grounding_dino(
    model,
    processor,
    device,
    image_pil: Image.Image,
    prompts: List[str] = ["rock", "stone", "meteorite", "rock . stone", "stone . meteorite"]
) -> dict:
    """
    使用 GroundingDINO 检测，尝试多种 prompt

    Returns:
        {
            'bboxes': [{'bbox': [x1,y1,x2,y2], 'score': float, 'prompt': str}, ...],
            'best_bbox': [x1, y1, x2, y2] 或 None
        }
    """
    w, h = image_pil.size
    results = {'bboxes': [], 'best_bbox': None}

    for prompt in prompts:
        try:
            inputs = processor(
                images=image_pil,
                text=prompt,
                return_tensors="pt"
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)

            # 解析输出
            logits = outputs.encoder_logits[0]  # [900, 256]
            boxes = outputs.encoder_pred_boxes[0]  # [900, 4] CXCYWH normalized

            # 计算分数
            scores = torch.sigmoid(logits.max(dim=-1).values)

            # 获取最高分的框
            best_idx = scores.argmax()
            best_score = scores[best_idx].item()
            best_box = boxes[best_idx]

            # 转换 CXCYWH -> XYXY (pixel)
            cx, cy, cw, ch = best_box.tolist()
            x1 = int((cx - cw / 2) * w)
            y1 = int((cy - ch / 2) * h)
            x2 = int((cx + cw / 2) * w)
            y2 = int((cy + ch / 2) * h)

            # 边界检查
            x1 = max(0, min(w, x1))
            y1 = max(0, min(h, y1))
            x2 = max(0, min(w, x2))
            y2 = max(0, min(h, y2))

            results['bboxes'].append({
                'bbox': [x1, y1, x2, y2],
                'score': best_score,
                'prompt': prompt
            })

        except Exception as e:
            print(f"  Prompt '{prompt}' 处理失败: {e}")

    # 选择最佳结果（score 最高）
    if results['bboxes']:
        best = max(results['bboxes'], key=lambda x: x['score'])
        results['best_bbox'] = best['bbox']
        results['best_prompt'] = best['prompt']
        results['best_score'] = best['score']

    return results


def segment_with_sam2(
    predictor,
    image_np: np.ndarray,
    bbox: Optional[List[int]] = None,
    point: Optional[Tuple[int, int]] = None,
) -> dict:
    """
    使用 SAM2 分割

    Args:
        image_np: 图像数组
        bbox: 边界框 [x1, y1, x2, y2]
        point: 点提示 (x, y)

    Returns:
        {
            'masks': np.ndarray (N, H, W),
            'scores': np.ndarray (N,),
            'best_mask': np.ndarray (H, W)
        }
    """
    predictor.set_image(image_np)

    if bbox is not None:
        masks, scores, logits = predictor.predict(
            box=np.array(bbox),
            multimask_output=True
        )
        prompt_type = 'box'
    elif point is not None:
        masks, scores, logits = predictor.predict(
            point_coords=np.array([[point[0], point[1]]]),
            point_labels=np.array([1]),
            multimask_output=True
        )
        prompt_type = 'point'
    else:
        # 使用图像中心点
        h, w = image_np.shape[:2]
        center = (w // 2, h // 2)
        masks, scores, logits = predictor.predict(
            point_coords=np.array([[center[0], center[1]]]),
            point_labels=np.array([1]),
            multimask_output=True
        )
        prompt_type = 'center_point'

    best_idx = scores.argmax()

    return {
        'masks': masks.astype(bool),  # 转换为 bool 类型
        'scores': scores,
        'best_mask': masks[best_idx].astype(bool),  # 确保 bool 类型
        'best_score': scores[best_idx],
        'prompt_type': prompt_type
    }


def visualize_detection(
    image_pil: Image.Image,
    detection_results: dict,
    output_path: str
):
    """
    可视化 GroundingDINO 检测结果
    在原图上绘制所有检测框，标注 prompt 和 score
    """
    img_draw = image_pil.copy()
    draw = ImageDraw.Draw(img_draw)

    # 尝试加载字体
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
    except:
        font = ImageFont.load_default()

    colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan']

    for i, result in enumerate(detection_results['bboxes']):
        bbox = result['bbox']
        score = result['score']
        prompt = result['prompt']

        color = colors[i % len(colors)]

        # 绘制框
        draw.rectangle(bbox, outline=color, width=3)

        # 标注文字
        text = f"{prompt}: {score:.2f}"
        draw.text((bbox[0], bbox[1] - 25), text, fill=color, font=font)

    # 标注最佳结果
    if detection_results['best_bbox']:
        best_bbox = detection_results['best_bbox']
        draw.rectangle(best_bbox, outline='yellow', width=5)
        text = f"BEST: {detection_results['best_prompt']} ({detection_results['best_score']:.2f})"
        draw.text((best_bbox[0], best_bbox[3] + 5), text, fill='yellow', font=font)

    img_draw.save(output_path, quality=95)
    print(f"  检测结果保存: {output_path}")


def visualize_segmentation(
    image_pil: Image.Image,
    segmentation_result: dict,
    output_path: str,
    title: str = "SAM2 Segmentation"
):
    """
    可视化 SAM2 分割结果
    显示掩码覆盖效果
    """
    image_np = np.array(image_pil)
    best_mask = segmentation_result['best_mask'].astype(bool)  # 确保 bool 类型

    # 创建掩码可视化
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 原图
    axes[0].imshow(image_pil)
    axes[0].set_title("Original")
    axes[0].axis('off')

    # 掩码
    axes[1].imshow(best_mask, cmap='gray')
    axes[1].set_title(f"Mask (score: {segmentation_result['best_score']:.3f})")
    axes[1].axis('off')

    # 掩码覆盖
    overlay = image_np.copy()
    # 绿色覆盖掩码区域
    overlay[best_mask] = [0, 255, 0]  # 绿色表示分割区域
    # 半透明混合
    blended = image_np.copy()
    blended[best_mask] = (image_np[best_mask] * 0.5 + overlay[best_mask] * 0.5).astype(np.uint8)

    axes[2].imshow(blended)
    axes[2].set_title(f"Overlay ({segmentation_result['prompt_type']})")
    axes[2].axis('off')

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"  分割结果保存: {output_path}")


def visualize_combined(
    image_pil: Image.Image,
    detection_results: dict,
    segmentation_result: dict,
    output_path: str
):
    """
    可视化组合结果：检测框 + 分割掩码
    """
    image_np = np.array(image_pil)
    best_mask = segmentation_result['best_mask'].astype(bool)  # 确保 bool 类型

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # 检测框 + 分割掩码
    overlay = image_np.copy()
    overlay[best_mask] = [0, 255, 0]
    blended = (image_np * 0.6 + overlay * 0.4).astype(np.uint8)
    blended_pil = Image.fromarray(blended)
    draw = ImageDraw.Draw(blended_pil)

    # 绘制最佳检测框
    if detection_results['best_bbox']:
        bbox = detection_results['best_bbox']
        draw.rectangle(bbox, outline='red', width=3)

    axes[0].imshow(blended_pil)
    axes[0].set_title(f"GD Box + SAM2 Mask\nGD score: {detection_results['best_score']:.2f}\nSAM2 score: {segmentation_result['best_score']:.3f}")
    axes[0].axis('off')

    # 分割掩码的边界框
    mask_bbox = get_mask_bbox(best_mask)
    cropped = image_np[mask_bbox[1]:mask_bbox[3], mask_bbox[0]:mask_bbox[2]]
    mask_cropped = best_mask[mask_bbox[1]:mask_bbox[3], mask_bbox[0]:mask_bbox[2]]

    # 填充背景
    cropped_with_bg = cropped.copy()
    cropped_with_bg[~mask_cropped] = 128

    axes[1].imshow(cropped_with_bg)
    axes[1].set_title(f"Cropped Result\nSize: {mask_bbox[2]-mask_bbox[0]}x{mask_bbox[3]-mask_bbox[1]}")
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"  组合结果保存: {output_path}")


def get_mask_bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """从掩码获取边界框"""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    y_indices = np.where(rows)[0]
    x_indices = np.where(cols)[0]

    if len(y_indices) == 0 or len(x_indices) == 0:
        return (0, 0, mask.shape[1], mask.shape[0])

    y1, y2 = y_indices[0], y_indices[-1] + 1
    x1, x2 = x_indices[0], x_indices[-1] + 1

    return (int(x1), int(y1), int(x2), int(y2))


def test_single_image(
    image_path: str,
    gd_model,
    gd_processor,
    gd_device,
    sam2_predictor,
    prompts: List[str]
):
    """测试单张图片，输出所有可视化结果"""
    img_name = os.path.basename(image_path)
    print(f"\n处理: {img_name}")

    # 加载图像
    image_pil = Image.open(image_path).convert("RGB")
    image_np = np.array(image_pil)
    w, h = image_pil.size
    print(f"  图像尺寸: {w}x{h}")

    # 1. GroundingDINO 检测
    print("  运行 GroundingDINO...")
    detection_results = detect_with_grounding_dino(
        gd_model, gd_processor, gd_device, image_pil, prompts
    )

    if detection_results['bboxes']:
        print(f"  检测到 {len(detection_results['bboxes'])} 个框")
        for r in detection_results['bboxes']:
            print(f"    - {r['prompt']}: box={r['bbox']}, score={r['score']:.3f}")
    else:
        print("  未检测到任何框")

    # 保存检测结果可视化
    gd_output = os.path.join(OUTPUT_DIR, f"{img_name}_gd_detection.jpg")
    visualize_detection(image_pil, detection_results, gd_output)

    # 2. SAM2 分割 - 使用中心点
    print("  运行 SAM2 (中心点 prompt)...")
    center_result = segment_with_sam2(
        sam2_predictor, image_np,
        point=(w // 2, h // 2)
    )
    print(f"  中心点分割 scores: {center_result['scores']}")

    sam2_center_output = os.path.join(OUTPUT_DIR, f"{img_name}_sam2_center.png")
    visualize_segmentation(
        image_pil, center_result, sam2_center_output,
        title=f"SAM2 Center Point ({w//2}, {h//2})"
    )

    # 3. SAM2 分割 - 使用检测框（如果有）
    if detection_results['best_bbox']:
        print("  运行 SAM2 (检测框 prompt)...")
        box_result = segment_with_sam2(
            sam2_predictor, image_np,
            bbox=detection_results['best_bbox']
        )
        print(f"  检测框分割 scores: {box_result['scores']}")

        sam2_box_output = os.path.join(OUTPUT_DIR, f"{img_name}_sam2_box.png")
        visualize_segmentation(
            image_pil, box_result, sam2_box_output,
            title=f"SAM2 Box Prompt {detection_results['best_bbox']}"
        )

        # 4. 组合结果
        combined_output = os.path.join(OUTPUT_DIR, f"{img_name}_combined.png")
        visualize_combined(image_pil, detection_results, box_result, combined_output)
    else:
        print("  无法运行检测框分割（无检测结果）")

    # 5. SAM2 分割 - 使用整图作为框（备用方案）
    print("  运行 SAM2 (整图框 prompt)...")
    full_box = [0, 0, w, h]
    full_box_result = segment_with_sam2(
        sam2_predictor, image_np,
        bbox=full_box
    )
    print(f"  整图框分割 scores: {full_box_result['scores']}")

    sam2_full_output = os.path.join(OUTPUT_DIR, f"{img_name}_sam2_fullbox.png")
    visualize_segmentation(
        image_pil, full_box_result, sam2_full_output,
        title=f"SAM2 Full Image Box [0,0,{w},{h}]"
    )


def main():
    """主测试函数"""
    print("=" * 60)
    print("GroundingDINO + SAM2.1 单独效果测试")
    print("=" * 60)

    # 设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    # 加载模型
    gd_model, gd_processor, gd_device = load_grounding_dino()
    sam2_predictor, sam2_device = load_sam2()

    # 测试图片
    test_dir = os.path.join(DATA_ROOT, "test_images")
    test_images = sorted([
        f for f in os.listdir(test_dir)
        if f.lower().endswith('.jpg')
    ])

    # 测试前 10 张
    test_count = min(10, len(test_images))
    print(f"\n测试图片数量: {test_count}")

    # 多种 prompt 测试
    prompts = [
        "rock",
        "stone",
        "meteorite",
        "rock . stone",
        "stone . meteorite",
        "a rock",
        "a stone",
        "rock in the image",
        "stone object",
        "geological sample"
    ]

    print(f"测试 prompts: {prompts}")
    print(f"输出目录: {OUTPUT_DIR}")

    # 处理每张图片
    for i in range(test_count):
        image_path = os.path.join(test_dir, test_images[i])
        test_single_image(
            image_path,
            gd_model, gd_processor, gd_device,
            sam2_predictor,
            prompts
        )

        # 定期清理显存
        if (i + 1) % 5 == 0:
            torch.cuda.empty_cache()

    # 总结
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)
    print(f"\n输出文件位于: {OUTPUT_DIR}")
    print("文件命名规则:")
    print("  *_gd_detection.jpg  - GroundingDINO 检测框可视化")
    print("  *_sam2_center.png   - SAM2 中心点分割结果")
    print("  *_sam2_box.png      - SAM2 检测框分割结果")
    print("  *_sam2_fullbox.png  - SAM2 整图框分割结果")
    print("  *_combined.png      - 组合结果可视化")

    # 释放资源
    print("\n释放模型...")
    gd_model.to("cpu")
    del gd_model
    sam2_predictor.model.to("cpu")
    del sam2_predictor
    torch.cuda.empty_cache()
    print("完成!")


if __name__ == "__main__":
    main()