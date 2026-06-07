"""
DomainDegradationAugmentor.py

对岩石图像施加域降质，模拟测试集的域偏移特征：
- 降分辨率、降亮度、降饱和度、高斯模糊、JPEG 压缩
每次随机选择 1~3 种操作组合应用。
"""

import os
import io
import random
from PIL import Image, ImageFilter, ImageEnhance
from tqdm import tqdm


class DomainDegradationAugmentor:
    """对岩石图像施加域降质，模拟测试集特征"""

    def __init__(self, save_quality=85):
        self.save_quality = save_quality

    def _degrade_resolution(self, img: Image.Image) -> Image.Image:
        """降分辨率：缩放到 256x256 再放大回原尺寸"""
        w, h = img.size
        small = img.resize((256, 256), Image.Resampling.BILINEAR)
        return small.resize((w, h), Image.Resampling.BILINEAR)

    def _degrade_brightness(self, img: Image.Image) -> Image.Image:
        """降低亮度：因子 0.5~0.9"""
        factor = random.uniform(0.5, 0.9)
        enhancer = ImageEnhance.Brightness(img)
        return enhancer.enhance(factor)

    def _degrade_saturation(self, img: Image.Image) -> Image.Image:
        """降低饱和度：因子 0.3~0.7"""
        factor = random.uniform(0.3, 0.7)
        enhancer = ImageEnhance.Color(img)
        return enhancer.enhance(factor)

    def _apply_blur(self, img: Image.Image) -> Image.Image:
        """高斯模糊：sigma = 0.5~2.0"""
        sigma = random.uniform(0.5, 2.0)
        return img.filter(ImageFilter.GaussianBlur(radius=sigma))

    def _degrade_jpeg(self, img: Image.Image) -> Image.Image:
        """随机 JPEG 压缩：quality 30~60"""
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=random.randint(30, 60))
        buf.seek(0)
        return Image.open(buf).convert("RGB")

    def apply(self, img: Image.Image) -> Image.Image:
        """随机选择 1~3 种降质操作组合应用"""
        ops = [
            self._degrade_resolution,
            self._degrade_brightness,
            self._degrade_saturation,
            self._apply_blur,
            self._degrade_jpeg,
        ]
        n_ops = random.randint(1, 3)
        chosen = random.sample(ops, n_ops)
        for op in chosen:
            img = op(img)
        return img

    def process_dataset(self, image_paths, output_dir,
                        augmentation_factor=1.0,
                        skip_existing=True):
        """
        遍历图像，为每张图生成 augmentation_factor 份降质副本。
        augmentation_factor 可以是浮点数（如 1.5 = 每张图至少 1 份，50% 图额外 1 份）。

        Returns:
            (成功数, 失败数)
        """
        os.makedirs(output_dir, exist_ok=True)

        success = 0
        fail = 0

        for img_path in tqdm(image_paths, desc="生成降质图像"):
            img_name = os.path.basename(img_path)
            base_name = os.path.splitext(img_name)[0]

            try:
                img = Image.open(img_path).convert("RGB")
            except Exception as e:
                fail += 1
                continue

            # 决定生成几份（支持浮点因子）
            n_copies = int(augmentation_factor)
            if augmentation_factor - n_copies > 0:
                if random.random() < (augmentation_factor - n_copies):
                    n_copies += 1

            for i in range(n_copies):
                out_name = f"{base_name}_deg{i}.jpg"
                out_path = os.path.join(output_dir, out_name)

                if skip_existing and os.path.exists(out_path):
                    success += 1
                    continue

                try:
                    degraded = self.apply(img)
                    degraded.save(out_path, format="JPEG", quality=self.save_quality)
                    success += 1
                except Exception as e:
                    fail += 1

        return success, fail
