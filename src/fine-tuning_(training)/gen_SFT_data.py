import os
import json
import pandas as pd
from tqdm import tqdm

# 导入数据增强模块
from src.data_processing_modules.ImageAugmentor import ImageAugmentor



def gen_sft_data(csv_path, image_dir, output_jsonl, augment_factor: float = 0.0):
    """
    根据博文流程，生成 Qwen3-VL-8B 专用的 SFT JSONL 数据集

    Args:
        csv_path: CSV 标注文件路径
        image_dir: 图像目录
        output_jsonl: 输出 JSONL 文件路径
        augment_factor: 数据增强因子，0 表示不增强，1-3 表示生成 1-3 倍增强数据
    """
    # 1. 检查并读取 CSV 标注
    if not os.path.exists(csv_path):
        print(f"错误：找不到标注文件 {csv_path}")
        return

    df = pd.read_csv(csv_path)
    sft_records = []

    print(f"正在为 Qwen3-VL 转换数据，总计：{len(df)} 条记录")

    # 数据增强处理
    augmented_image_dir = None
    if augment_factor > 0:
        print(f"启动数据增强，增强因子：{augment_factor}...")
        augmentor = ImageAugmentor(
            rotation_range=(-30, 30),
            scale_range=(0.8, 1.2),
            fill_color=(128, 128, 128),
            save_quality=95,
        )
        # 获取原始图像路径列表
        original_image_paths = [
            os.path.join(image_dir, str(row['id']))
            for _, row in df.iterrows()
            if os.path.exists(os.path.join(image_dir, str(row['id'])))
        ]
        # 生成增强图像到临时目录
        augmented_image_dir = os.path.join(os.path.dirname(output_jsonl), "train_images_augmented")
        os.makedirs(augmented_image_dir, exist_ok=True)
        aug_count, aug_fail = augmentor.process_dataset(
            image_paths=original_image_paths,
            output_dir=augmented_image_dir,
            augmentation_factor=augment_factor,
            skip_existing=False,
        )
        print(f"数据增强完成：生成 {aug_count} 张增强图像，{aug_fail} 失败")
        print(f"增强图像保存至：{augmented_image_dir}")

    # 2. 遍历标注数据（包含原始图像和增强图像）
    for _, row in tqdm(df.iterrows(), total=len(df)):
        img_id = str(row['id'])  # 确保是字符串，如 000001.jpg
        label = int(row['label'])

        # 构造原始图像路径
        img_path = os.path.abspath(os.path.join(image_dir, img_id))

        # 检查原始文件是否存在
        if not os.path.exists(img_path):
            continue

        # 3. 构造中文指令与输出
        # 修改后的提示词：强调忽略文字信息，专注于石头本身的视觉特征
        instruction = “请观察这张图片中石头的视觉特征（颜色、纹理、形状），判断其中是否有陨石？注意：忽略图片中的所有文字、标签、标尺等信息，仅根据石头本身的外观特征判断。回答时首先给出”是”或”否”，然后简要说明理由。”

        if label == 1:
            output = “是。图中石头展示了陨石特有的视觉特征：表面有深色熔壳或气印（类似手指按压的凹坑），颜色呈深褐色或金属灰色，纹理致密无明显晶体颗粒，形状经过大气层烧蚀呈流线型或圆润边缘。判断为陨石。”
        else:
            output = “否。图中石头不具备陨石的典型视觉特征：表面无熔壳，颜色为浅灰色/米黄色/白色等地球岩石常见色，纹理呈现明显的晶体颗粒结构（如花岗岩），或有地质脉体穿插。判断为普通石头或地球岩石。”

        # 4. 封装为博文要求的 JSONL 格式
        record = {
            "image_path": img_path,
            "instruction": instruction,
            "output": output
        }
        sft_records.append(record)

    # 添加增强图像的记录
    if augmented_image_dir and os.path.exists(augmented_image_dir):
        print(f"正在处理 {len(os.listdir(augmented_image_dir))} 张增强图像...")
        for aug_img_name in tqdm(os.listdir(augmented_image_dir), desc="处理增强图像"):
            if not aug_img_name.endswith(('.jpg', '.jpeg', '.png')):
                continue

            # 从增强文件名提取原始 ID (如 000001_aug0.jpg -> 000001.jpg)
            base_name = aug_img_name.replace('_aug0', '').replace('_aug1', '').replace('_aug2', '')
            img_id = base_name

            # 查找对应的 label
            label_row = df[df['id'].astype(str) == img_id]
            if label_row.empty:
                continue

            label = int(label_row.iloc[0]['label'])

            # 构造增强图像的绝对路径
            aug_img_path = os.path.abspath(os.path.join(augmented_image_dir, aug_img_name))

            # 构造指令和输出（与原始数据保持一致）
            instruction = “请观察这张图片中石头的视觉特征（颜色、纹理、形状），判断其中是否有陨石？注意：忽略图片中的所有文字、标签、标尺等信息，仅根据石头本身的外观特征判断。回答时首先给出”是”或”否”，然后简要说明理由。”

            if label == 1:
                output = “是。图中石头展示了陨石特有的视觉特征：表面有深色熔壳或气印（类似手指按压的凹坑），颜色呈深褐色或金属灰色，纹理致密无明显晶体颗粒，形状经过大气层烧蚀呈流线型或圆润边缘。判断为陨石。”
            else:
                output = “否。图中石头不具备陨石的典型视觉特征：表面无熔壳，颜色为浅灰色/米黄色/白色等地球岩石常见色，纹理呈现明显的晶体颗粒结构（如花岗岩），或有地质脉体穿插。判断为普通石头或地球岩石。”

            record = {
                "image_path": aug_img_path,
                "instruction": instruction,
                "output": output
            }
            sft_records.append(record)

    # 5. 保存为 JSONL 文件
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for record in sft_records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    print(f"成功生成 SFT 数据集：{output_jsonl}")
    print(f"有效样本数：{len(sft_records)}")
    if augment_factor > 0:
        original_count = len(df)
        augmented_count = len(sft_records) - original_count
        print(f"其中原始样本：{original_count}, 增强样本：{augmented_count}")


if __name__ == "__main__":
    # 配置项目路径
    # 假设当前脚本位于 meteorite-identification/src/fine-tuning_(training)/
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    print(f"当前脚本目录：{BASE_DIR}")
    DATA_ROOT = os.path.join(BASE_DIR, "../../data")
    print(f"数据根目录：{DATA_ROOT}")

    # 输入：原始 CSV 和 提取后的干净图片目录
    CSV_INPUT = os.path.join(DATA_ROOT, "train_labels.csv")
    print(f"CSV 输入路径：{CSV_INPUT}")
    IMAGE_INPUT = os.path.join(DATA_ROOT, "train_images")
    print(f"图片输入目录：{IMAGE_INPUT}")

    # 输出：供训练使用的 JSONL
    SFT_OUTPUT = os.path.join(DATA_ROOT, "qwen3_sft_train.jsonl")
    print(f"SFT 输出路径：{SFT_OUTPUT}")

    # 设置增强因子：2.0 表示生成 2 倍增强数据（总共 3 倍数据量）
    # 可调整范围：1.0-3.0
    AUGMENT_FACTOR = 2.0

    gen_sft_data(CSV_INPUT, IMAGE_INPUT, SFT_OUTPUT, augment_factor=AUGMENT_FACTOR)
