"""
check_extract_quality.py

检查 SAM2 提取质量，筛选出：
1. 提取后尺寸过小的图像（可能分割失败）
2. 提取后与原图尺寸差异过大的图像（可能裁剪过度）
输出可疑图像列表，供人工检查。
"""

import os
import json
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = "/root/Projects/meteorite-identification"
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")

# 阈值配置
MIN_EXTRACT_SIZE = 50        # 提取后短边小于此值视为过小
MAX_SIZE_RATIO = 0.15        # 提取面积 / 原图面积 < 此值视为裁剪过度
MIN_SIZE_RATIO = 0.05        # 提取面积 / 原图面积 < 此值视为严重异常

# 检查模式：train / test / both
CHECK_MODE = "both"


def check_directory(original_dir, extracted_dir, output_prefix="train"):
    """检查一个目录的提取质量"""
    if not os.path.exists(original_dir) or not os.path.exists(extracted_dir):
        print(f"  目录不存在，跳过: {original_dir}")
        return [], []

    images = sorted([f for f in os.listdir(original_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
    print(f"  图片总数: {len(images)}")

    too_small = []    # 提取尺寸过小
    too_cropped = []  # 裁剪过度

    for img_name in tqdm(images, desc=f"{output_prefix}"):
        orig_path = os.path.join(original_dir, img_name)
        ext_path = os.path.join(extracted_dir, img_name)

        if not os.path.exists(ext_path):
            continue

        try:
            orig = Image.open(orig_path)
            ext = Image.open(ext_path)

            ow, oh = orig.size
            ew, eh = ext.size
            orig_area = ow * oh
            ext_area = ew * eh
            area_ratio = ext_area / orig_area if orig_area > 0 else 0

            # 规则1: 提取后短边过小
            if min(ew, eh) < MIN_EXTRACT_SIZE:
                too_small.append({
                    "filename": img_name,
                    "original_size": f"{ow}x{oh}",
                    "extracted_size": f"{ew}x{eh}",
                    "area_ratio": f"{area_ratio:.4f}",
                    "reason": f"短边过小: min({ew},{eh})={min(ew,eh)} < {MIN_EXTRACT_SIZE}",
                })

            # 规则2: 面积比例过小
            if area_ratio < MAX_SIZE_RATIO:
                too_cropped.append({
                    "filename": img_name,
                    "original_size": f"{ow}x{oh}",
                    "extracted_size": f"{ew}x{eh}",
                    "area_ratio": f"{area_ratio:.4f}",
                    "reason": f"面积比 {area_ratio:.1%} < {MAX_SIZE_RATIO:.0%}",
                })

        except Exception as e:
            print(f"  错误: {img_name}: {e}")

    return too_small, too_cropped


def main():
    print("=" * 60)
    print("SAM2 提取质量检查")
    print(f"  最小短边: {MIN_EXTRACT_SIZE}px")
    print(f"  最大裁剪比: {MAX_SIZE_RATIO:.0%}")
    print(f"  严重异常比: {MIN_SIZE_RATIO:.0%}")
    print("=" * 60)

    all_too_small = []
    all_too_cropped = []

    # 检查训练集
    if CHECK_MODE in ("train", "both"):
        print(f"\n[1/2] 检查训练集...")
        small, cropped = check_directory(
            os.path.join(DATA_ROOT, "train_images"),
            os.path.join(DATA_ROOT, "train_images_extracted"),
            "train"
        )
        all_too_small.extend(small)
        all_too_cropped.extend(cropped)

    # 检查测试集
    if CHECK_MODE in ("test", "both"):
        print(f"\n[2/2] 检查测试集...")
        small, cropped = check_directory(
            os.path.join(DATA_ROOT, "test_images"),
            os.path.join(DATA_ROOT, "test_images_extracted"),
            "test"
        )
        all_too_small.extend(small)
        all_too_cropped.extend(cropped)

    # 汇总
    print(f"\n{'='*60}")
    print(f"检查结果:")
    print(f"  短边过小: {len(all_too_small)} 张")
    print(f"  裁剪过度: {len(all_too_cropped)} 张")

    # 输出严重异常（面积比 < MIN_SIZE_RATIO）
    severe = [c for c in all_too_cropped if float(c["area_ratio"]) < MIN_SIZE_RATIO]
    if severe:
        print(f"\n⚠️  严重异常 ({len(severe)} 张):")
        for item in severe[:20]:
            print(f"  {item['filename']}: {item['original_size']} -> {item['extracted_size']} ({item['reason']})")

    # 保存结果
    output_dir = os.path.join(DATA_ROOT, "extract_quality_report")
    os.makedirs(output_dir, exist_ok=True)

    report = {
        "too_small": all_too_small,
        "too_cropped": all_too_cropped,
        "severe": severe,
        "summary": {
            "total_checked": len(all_too_small) + len(all_too_cropped),
            "too_small_count": len(all_too_small),
            "too_cropped_count": len(all_too_cropped),
            "severe_count": len(severe),
        }
    }

    report_path = os.path.join(output_dir, "quality_report.json")
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n报告已保存至: {report_path}")

    # 生成可疑图像列表文件（方便手动检查）
    suspicious_files = set()
    for item in all_too_small + all_too_cropped:
        suspicious_files.add(item["filename"])

    list_path = os.path.join(output_dir, "suspicious_files.txt")
    with open(list_path, 'w', encoding='utf-8') as f:
        for fn in sorted(suspicious_files):
            f.write(fn + '\n')
    print(f"可疑文件列表: {list_path} ({len(suspicious_files)} 个文件)")

    # 打印前 20 个可疑文件
    if suspicious_files:
        print(f"\n前 20 个可疑文件:")
        for fn in sorted(suspicious_files)[:20]:
            print(f"  {fn}")


if __name__ == "__main__":
    main()
