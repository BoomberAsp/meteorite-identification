import os
import random
from typing import List, Tuple, Optional
from PIL import Image, ImageEnhance
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ImageAugmentor:
    """
    图像数据增强器：通过缩放、旋转、翻转和颜色抖动生成新的训练数据

    增强策略：
    - 旋转：随机旋转 -30° 到 +30° 之间的角度
    - 缩放：随机缩放到原图的 0.8 到 1.2 倍
    - 翻转：按概率施加水平/垂直翻转
    - 颜色抖动：随机调整亮度、对比度、饱和度
    """

    def __init__(
        self,
        rotation_range: Tuple[float, float] = (-30, 30),
        scale_range: Tuple[float, float] = (0.8, 1.2),
        fill_color: Tuple[int, int, int] = (128, 128, 128),
        save_quality: int = 95,
        flip_horizontal_prob: float = 0.5,
        flip_vertical_prob: float = 0.0,
        color_jitter: Optional[Tuple[float, float, float]] = None,
    ):
        """
        Args:
            rotation_range: 旋转角度范围 (min, max)，单位度
            scale_range: 缩放比例范围 (min, max)，1.0 表示原大小
            fill_color: 旋转后填充的背景色，默认灰色
            save_quality: 保存质量 (1-100)
            flip_horizontal_prob: 水平翻转概率 (0.0=不翻转, 1.0=必翻转)
            flip_vertical_prob: 垂直翻转概率 (0.0=不翻转, 1.0=必翻转)
            color_jitter: (brightness_range, contrast_range, saturation_range)
                每个 range 为 (min, max) 因子，如 (0.7, 1.4)
                None 表示不施加颜色抖动
        """
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.fill_color = fill_color
        self.save_quality = save_quality
        self.flip_horizontal_prob = flip_horizontal_prob
        self.flip_vertical_prob = flip_vertical_prob
        self.color_jitter = color_jitter

    def augment(self, image: Image.Image) -> Image.Image:
        """
        对单张图像应用随机增强

        Args:
            image: PIL Image 对象

        Returns:
            增强后的 PIL Image 对象
        """
        # 随机旋转
        rotation_angle = random.uniform(*self.rotation_range)
        image = image.rotate(
            rotation_angle,
            resample=Image.BICUBIC,
            fillcolor=self.fill_color
        )

        # 随机缩放
        scale_factor = random.uniform(*self.scale_range)
        if scale_factor != 1.0:
            new_size = (
                int(image.width * scale_factor),
                int(image.height * scale_factor)
            )
            image = image.resize(new_size, resample=Image.BICUBIC)

        # 水平翻转
        if random.random() < self.flip_horizontal_prob:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        # 垂直翻转
        if random.random() < self.flip_vertical_prob:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)

        # 颜色抖动
        if self.color_jitter is not None:
            brightness_range, contrast_range, saturation_range = self.color_jitter
            n_ops = random.randint(1, 2)  # 每次随机选 1~2 种
            ops = []
            if brightness_range:
                ops.append(('brightness', ImageEnhance.Brightness(image)))
            if contrast_range:
                ops.append(('contrast', ImageEnhance.Contrast(image)))
            if saturation_range:
                ops.append(('saturation', ImageEnhance.Color(image)))
            if ops:
                chosen = random.sample(ops, min(n_ops, len(ops)))
                for name, enhancer in chosen:
                    if name == 'brightness':
                        factor = random.uniform(*brightness_range)
                    elif name == 'contrast':
                        factor = random.uniform(*contrast_range)
                    else:
                        factor = random.uniform(*saturation_range)
                    image = enhancer.enhance(factor)

        return image

    def augment_and_save(
        self,
        image_path: str,
        output_path: str,
        augment_count: int = 1
    ) -> List[str]:
        """
        读取图像，生成指定数量的增强版本并保存

        Args:
            image_path: 原始图像路径
            output_path: 输出图像路径（如果 augment_count > 1，会自动添加后缀）
            augment_count: 为这张图像生成的增强样本数量

        Returns:
            生成的所有增强图像的路径列表
        """
        try:
            image = Image.open(image_path).convert("RGB")
            generated_paths = []

            for i in range(augment_count):
                # 生成增强图像
                augmented = self.augment(image)

                # 构造输出路径
                if augment_count > 1:
                    base, ext = os.path.splitext(output_path)
                    final_path = f"{base}_aug{i}{ext}"
                else:
                    final_path = output_path

                # 保存
                augmented.save(final_path, quality=self.save_quality)
                generated_paths.append(final_path)

            return generated_paths

        except Exception as e:
            logger.error(f"Error augmenting {image_path}: {e}")
            return []

    def process_dataset(
        self,
        image_paths: List[str],
        output_dir: str,
        augmentation_factor: float = 2.0,
        skip_existing: bool = True
    ) -> Tuple[int, int]:
        """
        对整个数据集进行增强

        增强策略说明：
        - augmentation_factor=1.0: 每张原图生成 1 张增强图 (总量翻倍)
        - augmentation_factor=2.0: 每张原图生成 2 张增强图 (总量三倍)
        - augmentation_factor=3.0: 每张原图生成 3 张增强图 (总量四倍)

        Args:
            image_paths: 原始图像路径列表
            output_dir: 增强图像输出目录
            augmentation_factor: 增强因子，表示每张原图生成的增强图数量
            skip_existing: 是否跳过已存在的文件

        Returns:
            (成功数量，失败数量)
        """
        os.makedirs(output_dir, exist_ok=True)

        success_cnt = 0
        fail_cnt = 0

        # 确定每张图生成的增强样本数
        # augmentation_factor=1.0 -> 1 张，2.0 -> 2 张，3.0 -> 3 张
        base_aug_count = int(augmentation_factor)
        fractional_part = augmentation_factor - base_aug_count

        logger.info(f"Starting augmentation: {len(image_paths)} images, factor={augmentation_factor}")
        logger.info(f"Base augmentations per image: {base_aug_count}, fractional probability: {fractional_part}")

        for img_path in image_paths:
            img_name = os.path.basename(img_path)
            output_path = os.path.join(output_dir, img_name)

            # 检查是否需要跳过（仅针对原图路径，增强图总是新生成）
            # 这里我们总是生成新的增强图

            # 决定是否添加额外的一个增强样本（处理小数部分）
            aug_count = base_aug_count
            if fractional_part > 0 and random.random() < fractional_part:
                aug_count += 1

            # 确保至少生成 1 张增强图
            aug_count = max(1, aug_count)

            result = self.augment_and_save(img_path, output_path, augment_count=aug_count)

            if result:
                success_cnt += len(result)
            else:
                fail_cnt += 1

        logger.info(f"Augmentation completed: {success_cnt} generated, {fail_cnt} failed")
        return success_cnt, fail_cnt


def main():
    """
     standalone 测试/运行函数
    """
    # 配置路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(script_dir, "..", "..")
    data_root = os.path.join(project_root, "data")

    # 输入输出
    input_image_dir = os.path.join(data_root, "train_images_extracted")
    output_dir = os.path.join(data_root, "train_images_extracted_augmented")

    # 设备检查
    device = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
    logger.info(f"Using device: {device}")

    # 获取所有训练图片
    image_paths = [
        os.path.join(input_image_dir, f)
        for f in os.listdir(input_image_dir)
        if f.endswith(('.jpg', '.png', '.jpeg'))
    ]

    logger.info(f"Found {len(image_paths)} images to augment")

    # 创建增强器并处理
    augmentor = ImageAugmentor(
        rotation_range=(-30, 30),
        scale_range=(0.8, 1.2),
        fill_color=(128, 128, 128),
        save_quality=95,
    )

    # 增强因子：2.0 表示每张原图生成 2 张增强图（总共 3 倍数据量）
    success, fail = augmentor.process_dataset(
        image_paths=image_paths,
        output_dir=output_dir,
        augmentation_factor=2.0,
        skip_existing=False,
    )

    logger.info(f"Total: {success} augmented images generated, {fail} failures")


if __name__ == "__main__":
    main()
