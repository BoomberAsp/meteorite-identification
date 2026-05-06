"""
StoneExtractor: 使用 SAM2.1 提取石头主体区域，去除背景干扰。

功能：
1. 使用 SAM2.1 large model 以图像中心点为 prompt 进行分割
2. 根据分割掩码边界裁剪石头主体
3. 输出裁剪后的图片，大小为分割区域大小
4. 不改变图像分辨率
5. 仅对未增强的 train_images 和 test_images 进行处理

依赖：
    - sam2 (从 /root/Projects/sam2 安装)
    - torch, pillow, numpy, tqdm
"""

import os
import sys
import logging
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from typing import Optional, List, Tuple, Dict, Any

# 添加项目根目录到路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, "..", "..")
sys.path.insert(0, project_root)

# 常量路径
SAM2_ROOT = "/root/Projects/sam2"

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class StoneExtractor:
    """
    使用 SAM2.1 提取石头主体区域。

    工作流程：
    1. 使用图像中心点作为 SAM2 的 prompt
    2. SAM2 生成分割掩码
    3. 根据掩码边界裁剪石头主体区域
    4. 输出裁剪后的图片（大小为分割区域大小，不改变分辨率）
    """

    def __init__(
        self,
        sam2_predictor: Optional[Any] = None,
        device: str = "cuda",
        background_value: int = 128,
        save_quality: int = 95,
        padding: int = 10,
    ):
        """
        Args:
            sam2_predictor: SAM2 图像预测器
            device: 计算设备
            background_value: 背景填充灰度值 (0-255)
            save_quality: JPEG 保存质量
            padding: 裁剪边距 (像素)
        """
        self.sam2_predictor = sam2_predictor
        self.device = device
        self.background_value = background_value
        self.save_quality = save_quality
        self.padding = padding

    @classmethod
    def from_pretrained(
        cls,
        sam2_checkpoint: str = os.path.join(SAM2_ROOT, "checkpoints/sam2.1_hiera_large.pt"),
        sam2_config: str = "configs/sam2.1/sam2.1_hiera_l.yaml",
        device: str = "cuda",
        **kwargs,
    ):
        """
        从预训练模型加载 StoneExtractor

        Args:
            sam2_checkpoint: SAM2.1 large 权重路径
            sam2_config: SAM2.1 large 配置路径
            device: 计算设备
        """
        logger.info("=" * 50)
        logger.info("加载 SAM2.1 large 模型...")

        sam2_predictor = None

        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            logger.info(f"Checkpoint: {sam2_checkpoint}")
            logger.info(f"Config: {sam2_config}")

            # 切换到 sam2 目录加载配置
            original_dir = os.getcwd()
            os.chdir(SAM2_ROOT)

            sam2_model = build_sam2(sam2_config, sam2_checkpoint, device=device)
            sam2_predictor = SAM2ImagePredictor(sam2_model)

            os.chdir(original_dir)
            logger.info("SAM2.1 large 加载成功")

        except Exception as e:
            logger.error(f"SAM2 加载失败: {e}")
            raise

        logger.info("=" * 50)

        return cls(
            sam2_predictor=sam2_predictor,
            device=device,
            **kwargs,
        )

    def _get_center_point_mask(
        self,
        image_np: np.ndarray
    ) -> np.ndarray:
        """
        使用 SAM2 以图像中心点为 prompt 生成分割掩码

        Args:
            image_np: numpy 图像数组 (H, W, 3)

        Returns:
            bool 掩码数组 (H, W)
        """
        h, w = image_np.shape[:2]

        # 使用图像中心点作为 prompt
        center_x = w // 2
        center_y = h // 2

        logger.debug(f"中心点 prompt: ({center_x}, {center_y})")

        # 设置图像
        self.sam2_predictor.set_image(image_np)

        # 使用中心点进行分割
        masks, scores, _ = self.sam2_predictor.predict(
            point_coords=np.array([[center_x, center_y]]),
            point_labels=np.array([1]),  # 1 表示前景
            multimask_output=True
        )

        # 选择得分最高的掩码
        best_idx = scores.argmax()
        best_mask = masks[best_idx].astype(bool)
        best_score = scores[best_idx]

        logger.debug(f"SAM2 掩码得分: {scores.tolist()}, 最佳: {best_score:.3f}")

        return best_mask

    def _get_mask_bbox(
        self,
        mask: np.ndarray,
        padding: int = 0
    ) -> Tuple[int, int, int, int]:
        """
        从掩码获取边界框

        Args:
            mask: bool 掩码数组 (H, W)
            padding: 边距

        Returns:
            (x1, y1, x2, y2)
        """
        h, w = mask.shape

        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)

        y_indices = np.where(rows)[0]
        x_indices = np.where(cols)[0]

        if len(y_indices) == 0 or len(x_indices) == 0:
            # 没有有效掩码，返回整图
            return (0, 0, w, h)

        y1, y2 = y_indices[0], y_indices[-1] + 1
        x1, x2 = x_indices[0], x_indices[-1] + 1

        # 添加边距
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)

        return (int(x1), int(y1), int(x2), int(y2))

    def process_image(
        self,
        image_path: str,
        output_path: str
    ) -> bool:
        """
        处理单张图片：分割、裁剪

        Args:
            image_path: 输入图片路径
            output_path: 输出图片路径

        Returns:
            是否成功
        """
        try:
            # 读取图像
            image_pil = Image.open(image_path).convert("RGB")
            image_np = np.array(image_pil)
            w, h = image_pil.size

            # Step 1: 使用 SAM2 以中心点为 prompt 生成分割掩码
            mask = self._get_center_point_mask(image_np)

            # Step 2: 获取掩码边界
            bbox = self._get_mask_bbox(mask, padding=self.padding)
            x1, y1, x2, y2 = bbox

            # 检查裁剪区域有效性
            if x2 <= x1 or y2 <= y1:
                logger.warning(f"无效裁剪区域: [{x1},{y1},{x2},{y2}]，返回原图")
                image_pil.save(output_path, quality=self.save_quality)
                return True

            # Step 3: 裁剪掩码区域（不改变分辨率）
            cropped_np = image_np[y1:y2, x1:x2]
            mask_cropped = mask[y1:y2, x1:x2]

            # Step 4: 将背景填充为灰色
            if cropped_np.ndim == 3:
                cropped_np[~mask_cropped] = self.background_value
            else:
                cropped_np[~mask_cropped] = self.background_value

            # Step 5: 保存结果
            result_img = Image.fromarray(cropped_np)
            result_img.save(output_path, quality=self.save_quality)

            logger.debug(f"处理成功: {image_path} -> {output_path} (裁剪: {x2-x1}x{y2-y1})")
            return True

        except Exception as e:
            logger.error(f"处理失败 {image_path}: {e}")
            return False

    def process_batch(
        self,
        image_paths: List[str],
        output_dir: str,
        skip_existing: bool = True
    ) -> Tuple[int, int]:
        """
        批量处理图片

        Args:
            image_paths: 输入图片路径列表
            output_dir: 输出目录
            skip_existing: 是否跳过已存在的文件

        Returns:
            (成功数量, 失败数量)
        """
        os.makedirs(output_dir, exist_ok=True)
        success_cnt = 0
        fail_cnt = 0

        for i, path in enumerate(tqdm(image_paths, desc="Processing images")):
            img_name = os.path.basename(path)
            out_path = os.path.join(output_dir, img_name)

            # 跳过已处理的文件
            if skip_existing and os.path.exists(out_path):
                success_cnt += 1
                continue

            if self.process_image(path, out_path):
                success_cnt += 1
            else:
                fail_cnt += 1

            # 定期释放显存
            if (i + 1) % 50 == 0:
                torch.cuda.empty_cache()

        return success_cnt, fail_cnt

    def process_dataset(
        self,
        data_root: str,
        process_train: bool = True,
        process_test: bool = True,
        skip_existing: bool = True
    ) -> Dict[str, Tuple[int, int]]:
        """
        处理完整数据集（仅未增强的数据）

        Args:
            data_root: 数据根目录
            process_train: 是否处理 train_images
            process_test: 是否处理 test_images
            skip_existing: 是否跳过已存在的文件

        Returns:
            {'train': (success, fail), 'test': (success, fail)}
        """
        results = {}

        # 处理 train_images
        if process_train:
            train_input = os.path.join(data_root, "train_images")
            train_output = os.path.join(data_root, "train_images_extracted")

            if os.path.exists(train_input):
                logger.info(f"\n处理训练图片: {train_input}")

                image_paths = [
                    os.path.join(train_input, f)
                    for f in os.listdir(train_input)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
                ]

                logger.info(f"找到 {len(image_paths)} 张训练图片")
                results['train'] = self.process_batch(
                    image_paths, train_output, skip_existing
                )
            else:
                logger.warning(f"训练图片目录不存在: {train_input}")
                results['train'] = (0, 0)

        # 处理 test_images
        if process_test:
            test_input = os.path.join(data_root, "test_images")
            test_output = os.path.join(data_root, "test_images_extracted")

            if os.path.exists(test_input):
                logger.info(f"\n处理测试图片: {test_input}")

                image_paths = [
                    os.path.join(test_input, f)
                    for f in os.listdir(test_input)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
                ]

                logger.info(f"找到 {len(image_paths)} 张测试图片")
                results['test'] = self.process_batch(
                    image_paths, test_output, skip_existing
                )
            else:
                logger.warning(f"测试图片目录不存在: {test_input}")
                results['test'] = (0, 0)

        return results

    def release(self):
        """释放模型资源"""
        if self.sam2_predictor is not None:
            if hasattr(self.sam2_predictor, "model"):
                self.sam2_predictor.model.to("cpu")
                del self.sam2_predictor.model
            del self.sam2_predictor
            self.sam2_predictor = None
            logger.info("SAM2 模型已释放")

        torch.cuda.empty_cache()
        logger.info("显存已清理")


def main():
    """
    主函数：处理未增强的 train_images 和 test_images
    """
    print("=" * 60)
    print("StoneExtractor - 石头主体提取工具 (SAM2.1)")
    print("=" * 60)

    # 数据根目录
    data_root = os.path.join(project_root, "data")

    # 设备检查
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n使用设备: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # 加载模型
    print("\n加载 SAM2.1 large 模型...")
    try:
        extractor = StoneExtractor.from_pretrained(
            sam2_checkpoint=os.path.join(SAM2_ROOT, "checkpoints/sam2.1_hiera_large.pt"),
            sam2_config="configs/sam2.1/sam2.1_hiera_l.yaml",
            device=device,
            background_value=128,
            save_quality=95,
            padding=10,
        )
    except Exception as e:
        print(f"模型加载失败: {e}")
        print("请确保已安装 SAM2: pip install -e . (从 /root/Projects/sam2)")
        return

    # 处理数据集
    print("\n开始处理数据集...")
    results = extractor.process_dataset(
        data_root=data_root,
        process_train=True,
        process_test=True,
        skip_existing=True
    )

    # 输出结果
    print("\n" + "=" * 60)
    print("处理完成!")
    print("=" * 60)

    for name, (success, fail) in results.items():
        total = success + fail
        print(f"{name}: 成功 {success}/{total} 张, 失败 {fail} 张")

    # 输出目录
    print(f"\n输出目录:")
    print(f"  训练图片: {os.path.join(data_root, 'train_images_extracted')}")
    print(f"  测试图片: {os.path.join(data_root, 'test_images_extracted')}")

    # 释放资源
    print("\n释放模型资源...")
    extractor.release()
    print("完成!")


if __name__ == "__main__":
    main()