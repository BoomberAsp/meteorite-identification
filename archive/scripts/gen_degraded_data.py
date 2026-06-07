"""
gen_degraded_data.py

生成域降质增强图像，输出到 data/train_images_degraded/
从 data/train_images/ 读取原始图像，施加 5 种降质操作中的 1~3 种组合。

用法:
    python gen_degraded_data.py                    # augmentation_factor=2.0
    python gen_degraded_data.py --factor 3.0       # 每张图 3 份降质副本
"""

import os
import argparse

from src.data_processing_modules.DomainDegradationAugmentor import DomainDegradationAugmentor

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
IMG_DIR = os.path.join(DATA_ROOT, "train_images")
OUTPUT_DIR = os.path.join(DATA_ROOT, "train_images_degraded")


def main():
    parser = argparse.ArgumentParser(description="生成域降质增强图像")
    parser.add_argument("--factor", type=float, default=2.0,
                        help="每张图生成的降质副本数（支持浮点，如 1.5 = 50% 图生成 2 份）")
    parser.add_argument("--skip-existing", action="store_true",
                        help="跳过已存在的输出文件")
    args = parser.parse_args()

    if not os.path.exists(IMG_DIR):
        print(f"错误: 找不到原始图像目录 {IMG_DIR}")
        return

    image_paths = [os.path.join(IMG_DIR, f) for f in os.listdir(IMG_DIR)
                   if f.endswith(('.jpg', '.jpeg', '.png'))]
    print(f"原始图像数量: {len(image_paths)}")
    print(f"增强因子: {args.factor}")
    print(f"输出目录: {OUTPUT_DIR}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    augmentor = DomainDegradationAugmentor()
    count, fail = augmentor.process_dataset(
        image_paths, OUTPUT_DIR,
        augmentation_factor=args.factor,
        skip_existing=args.skip_existing)

    print(f"\n降质增强完成: {count} 成功, {fail} 失败")
    print(f"输出文件数: {len(os.listdir(OUTPUT_DIR))}")


if __name__ == "__main__":
    main()
