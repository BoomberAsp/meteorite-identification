"""
convert_gray_to_white.py

将 train_images_extracted/ 中的 SAM2 提取图像背景
从灰色 (128,128,128) 转换为白色 (255,255,255)，
以匹配测试集提取图的纯白背景。

由于 SAM2 提取时背景是均匀填充的灰色像素，
可以通过颜色距离阈值法精确识别并替换。
"""

import os
import numpy as np
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = "/root/Projects/meteorite-identification"
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
INPUT_DIR = os.path.join(DATA_ROOT, "train_images_extracted")
OUTPUT_DIR = os.path.join(DATA_ROOT, "train_images_extracted_white_bg")

# 灰色背景值
GRAY_BG = np.array([128, 128, 128], dtype=np.float32)
# 白色背景
WHITE_BG = np.array([255, 255, 255], dtype=np.float32)
# 距离阈值：与灰色的 L2 距离小于此值视为背景像素
DISTANCE_THRESHOLD = 40.0


def convert_bg(img: Image.Image) -> Image.Image:
    """将灰色背景替换为白色"""
    arr = np.array(img, dtype=np.float32)
    dist = np.linalg.norm(arr - GRAY_BG, axis=-1)
    mask = dist < DISTANCE_THRESHOLD
    # 替换为白色
    arr[mask] = WHITE_BG
    return Image.fromarray(arr.astype(np.uint8))


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    images = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith(('.jpg', '.jpeg', '.png'))])
    print(f"输入: {INPUT_DIR}")
    print(f"输出: {OUTPUT_DIR}")
    print(f"图片数: {len(images)}")

    success = 0
    fail = 0
    for fname in tqdm(images, desc="转换背景"):
        try:
            img = Image.open(os.path.join(INPUT_DIR, fname)).convert("RGB")
            converted = convert_bg(img)
            out_name = fname.rsplit('.', 1)[0] + '.jpg'
            converted.save(os.path.join(OUTPUT_DIR, out_name), format="JPEG", quality=95)
            success += 1
        except Exception as e:
            fail += 1
            print(f"\n  失败: {fname} - {e}")

    print(f"\n完成: {success} 成功, {fail} 失败")

    # 验证：检查输出图是否还有灰色背景
    if success > 0:
        sample = np.array(Image.open(os.path.join(OUTPUT_DIR, "000001.jpg")))
        gray_px = np.sum((sample.astype(float) - GRAY_BG)**2, axis=-1) < 1600
        print(f"验证 000001.jpg: 剩余灰色像素 {gray_px.sum()}/{sample.shape[0]*sample.shape[1]}")


if __name__ == "__main__":
    main()
