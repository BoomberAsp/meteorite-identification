"""
StoneExtractor: 使用 Grounding DINO + SAM2 提取石头主体区域，去除背景干扰。

功能：
1. 使用 Grounding DINO 检测石头区域
2. 使用 SAM2 生成精确分割掩码
3. 裁剪石头主体并填充背景
4. 批量处理数据集图片

依赖安装：
    pip install transformers torch pillow numpy tqdm
    pip install sam-2  # 或从源码安装：pip install -e . (SAM2 仓库)
"""

import os
import sys
import logging
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from typing import Optional, List, Tuple

# 添加项目根目录到路径，确保导入正确
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, "..", "..")
sys.path.insert(0, project_root)

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class StoneExtractor:
    """
    使用 Grounding DINO + SAM2 提取石头主体区域，去除背景干扰。

    工作流程：
    1. Grounding DINO 检测石头/岩石区域，返回边界框
    2. SAM2 根据边界框生成精确的分割掩码
    3. 使用掩码去除背景，填充为统一颜色
    4. 裁剪石头主体区域并保存
    """

    def __init__(
        self,
        dino_model: Optional[torch.nn.Module] = None,
        dino_processor: Optional[object] = None,
        sam2_predictor: Optional[object] = None,
        device: str = "cuda",
        background_value: int = 128,
        default_padding: int = 20,
        save_quality: int = 95,
        dino_box_threshold: float = 0.35,
        dino_text_threshold: float = 0.25,
    ):
        """
        Args:
            dino_model: Grounding DINO 模型
            dino_processor: Grounding DINO 处理器 (transformers 格式)
            sam2_predictor: SAM2 图像预测器
            device: 计算设备
            background_value: 背景填充值 (0-255)，128 为中性灰色
            default_padding: 裁剪时的额外边距 (像素)
            save_quality: JPEG 保存质量 (1-100)
            dino_box_threshold: DINO 检测框置信度阈值
            dino_text_threshold: DINO 文本相似度阈值
        """
        self.dino_model = dino_model
        self.dino_processor = dino_processor
        self.sam2_predictor = sam2_predictor
        self.device = device
        self.background_value = background_value
        self.default_padding = default_padding
        self.save_quality = save_quality
        self.dino_box_threshold = dino_box_threshold
        self.dino_text_threshold = dino_text_threshold

        if dino_model is not None:
            self.dino_model.to(device).eval()

    @classmethod
    def from_pretrained(
        cls,
        dino_model_name: str = "IDEA-Research/grounding-dino-base",
        sam2_model_cfg: str = "configs/sam2.1/sam2.1_hiera_l.yaml",
        sam2_checkpoint: str = "checkpoints/sam2.1_hiera_large.pt",
        device: str = "cuda",
        **kwargs,
    ):
        """
        从预训练模型加载 StoneExtractor

        Args:
            dino_model_name: Grounding DINO 模型名称或本地路径
            sam2_model_cfg: SAM2 模型配置文件路径
            sam2_checkpoint: SAM2 模型权重文件路径
            device: 计算设备
        """
        logger.info(f"Loading Grounding DINO from {dino_model_name}...")
        dino_model = None
        dino_processor = None

        try:
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
            dino_processor = AutoProcessor.from_pretrained(dino_model_name)
            dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
                dino_model_name,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            )
            logger.info("Grounding DINO 加载成功")
        except ImportError as e:
            logger.warning(f"transformers 库未安装或模型加载失败：{e}")
            logger.info("将继续加载 SAM2，但需要手动提供 DINO 模型或使用备用方案")
        except Exception as e:
            logger.warning(f"Grounding DINO 加载失败：{e}")

        logger.info(f"Loading SAM2 from {sam2_checkpoint}...")
        sam2_predictor = None

        try:
            # SAM2 导入 - 支持不同安装方式
            try:
                # 方式 1: pip 安装的 sam-2 包
                from sam2.build_sam import build_sam2
                from sam2.sam2_image_predictor import SAM2ImagePredictor
            except ImportError:
                # 方式 2: 从源码安装的 sam2 包
                from sam2.build_sam import build_sam2
                from sam2.sam2_image_predictor import SAM2ImagePredictor

            sam2_model = build_sam2(sam2_model_cfg, sam2_checkpoint, device=device)
            sam2_predictor = SAM2ImagePredictor(sam2_model)
            logger.info("SAM2 加载成功")
        except ImportError as e:
            logger.warning(f"SAM2 库未安装或加载失败：{e}")
            logger.info("请安装 SAM2: pip install -e . (从 SAM2 仓库) 或 pip install sam-2")
        except Exception as e:
            logger.warning(f"SAM2 加载失败：{e}")

        return cls(
            dino_model=dino_model,
            dino_processor=dino_processor,
            sam2_predictor=sam2_predictor,
            device=device,
            **kwargs,
        )

    def _get_dino_bbox(
        self,
        image_pil: Image.Image,
        prompt: str = "rock or stone"
    ) -> Optional[List[float]]:
        """
        使用 Grounding DINO 检测目标物体

        Args:
            image_pil: PIL Image 对象
            prompt: 文本提示，用于检测的目标

        Returns:
            边界框 [x1, y1, x2, y2] 或 None
        """
        if self.dino_model is None or self.dino_processor is None:
            logger.warning("DINO 模型未加载，无法检测边界框")
            return None

        try:
            # 准备输入
            inputs = self.dino_processor(
                images=image_pil,
                text=prompt,
                return_tensors="pt"
            ).to(self.device)

            # 推理
            with torch.no_grad():
                outputs = self.dino_model(**inputs)

            # 后处理
            results = self.dino_processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                box_threshold=self.dino_box_threshold,
                text_threshold=self.dino_text_threshold,
                target_sizes=[image_pil.size[::-1]]  # [height, width]
            )

            # 提取最高置信度的框
            if results and len(results[0]["boxes"]) > 0:
                boxes = results[0]["boxes"][0]  # [x1, y1, x2, y2]
                scores = results[0]["scores"][0]
                logger.debug(f"DINO 检测到目标：box={boxes}, score={scores:.3f}")
                return boxes.tolist()
            else:
                logger.debug("DINO 未检测到目标")
                return None

        except Exception as e:
            logger.warning(f"DINO 检测失败：{e}")
            return None

    def _get_sam2_mask(
        self,
        image_np: np.ndarray,
        bbox: List[float]
    ) -> np.ndarray:
        """
        使用 SAM2 生成分割掩码

        Args:
            image_np: numpy 图像数组 (H, W, 3)
            bbox: 边界框 [x1, y1, x2, y2]

        Returns:
            bool 类型的掩码数组 (H, W)
        """
        if self.sam2_predictor is None:
            logger.warning("SAM2 预测器未加载，使用边界框作为掩码")
            # 备用方案：直接用边界框作为掩码
            h, w = image_np.shape[:2]
            x1, y1, x2, y2 = map(int, bbox)
            mask = np.zeros((h, w), dtype=bool)
            mask[y1:y2, x1:x2] = True
            return mask

        try:
            # 设置图像
            self.sam2_predictor.set_image(image_np)

            # 使用边界框提示进行预测
            mask, score, _ = self.sam2_predictor.predict(
                box=np.array(bbox),
                multimask_output=False
            )

            # SAM2 返回的是 (H, W) 的 bool 数组或 (1, H, W)
            if mask.ndim == 3:
                mask = mask[0]

            logger.debug(f"SAM2 掩码生成成功，得分：{score:.3f}")
            return mask.astype(bool)

        except Exception as e:
            logger.warning(f"SAM2 掩码生成失败：{e}")
            # 备用方案：直接用边界框作为掩码
            h, w = image_np.shape[:2]
            x1, y1, x2, y2 = map(int, bbox)
            mask = np.zeros((h, w), dtype=bool)
            mask[y1:y2, x1:x2] = True
            return mask

    def _get_center_bbox(self, image_pil: Image.Image) -> List[float]:
        """
        获取图像中心区域的边界框（备用方案）

        当 DINO 检测失败时，使用图像中心 80% 区域作为默认框
        """
        w, h = image_pil.size
        margin_x = int(w * 0.1)
        margin_y = int(h * 0.1)
        return [margin_x, margin_y, w - margin_x, h - margin_y]

    def _crop_stone(
        self,
        image_np: np.ndarray,
        bbox: List[float],
        padding: int
    ) -> Image.Image:
        """
        根据边界框裁剪石头区域

        Args:
            image_np: numpy 图像数组
            bbox: 边界框 [x1, y1, x2, y2]
            padding: 额外边距

        Returns:
            裁剪后的 PIL Image
        """
        h, w = image_np.shape[:2]
        x1, y1, x2, y2 = map(int, bbox)

        # 加入边距并确保在图像范围内
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)

        # 检查裁剪区域有效性
        if x2 <= x1 or y2 <= y1:
            logger.warning(f"无效的裁剪区域：[{x1},{y1},{x2},{y2}]，返回原图")
            return Image.fromarray(image_np)

        cropped = image_np[y1:y2, x1:x2]
        return Image.fromarray(cropped)

    def process_and_save(self, image_path: str, output_path: str) -> bool:
        """
        处理单张图片并保存结果

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

            # 可选：如果图片顶部有文字标注，先裁剪掉顶部区域
            # 检测图片高度，如果很高可能包含标尺等
            h, w = image_np.shape[:2]
            # 如果图片上方有超过 10% 的空白或文字区域，可以裁剪
            # 这里暂时不处理，依赖 SAM2 的分割能力

            # Step 1: 使用 DINO 检测石头区域
            bbox = self._get_dino_bbox(image_pil, prompt="rock or stone or meteorite")

            if bbox is None:
                logger.debug(f"DINO 检测失败，使用中心区域作为备用：{image_path}")
                bbox = self._get_center_bbox(image_pil)

            # Step 2: 使用 SAM2 生成分割掩码
            mask = self._get_sam2_mask(image_np, bbox)

            # Step 3: 使用掩码去除背景，填充为灰色
            if image_np.ndim == 3:
                # RGB 图像
                image_np[~mask] = self.background_value
            else:
                # 灰度图像
                image_np[~mask] = self.background_value

            # Step 4: 裁剪石头主体
            result_img = self._crop_stone(image_np, bbox, padding=self.default_padding)

            # Step 5: 保存结果
            result_img.save(output_path, quality=self.save_quality)
            logger.debug(f"处理成功：{image_path} -> {output_path}")
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
            (成功数量，失败数量)
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

            if self.process_and_save(path, out_path):
                success_cnt += 1
            else:
                fail_cnt += 1

            # 定期释放显存
            if (i + 1) % 50 == 0:
                torch.cuda.empty_cache()

        return success_cnt, fail_cnt

    def release(self):
        """释放模型资源"""
        if self.dino_model is not None:
            self.dino_model.to("cpu")
            del self.dino_model
            self.dino_model = None
            logger.info("DINO 模型已释放到 CPU")

        if self.sam2_predictor is not None:
            if hasattr(self.sam2_predictor, "model"):
                self.sam2_predictor.model.to("cpu")
                del self.sam2_predictor.model
            del self.sam2_predictor
            self.sam2_predictor = None
            logger.info("SAM2 模型已释放到 CPU")

        torch.cuda.empty_cache()
        logger.info("显存已清理")


def main():
    """
    主函数：批量处理训练图片
    """
    print("=" * 60)
    print("🪨 StoneExtractor - 石头主体提取工具")
    print("=" * 60)

    # 配置路径
    data_root = os.path.join(project_root, "data")

    # 输入输出目录
    input_image_dir = os.path.join(data_root, "train_images")
    output_dir = os.path.join(data_root, "train_images_cleaned")

    # 检查输入目录
    if not os.path.exists(input_image_dir):
        print(f"❌ 输入目录不存在：{input_image_dir}")
        return

    # 获取所有图片
    image_extensions = ('.jpg', '.jpeg', '.png', '.webp')
    image_paths = [
        os.path.join(input_image_dir, f)
        for f in os.listdir(input_image_dir)
        if f.lower().endswith(image_extensions)
    ]

    print(f"\n📂 找到 {len(image_paths)} 张图片待处理")
    print(f"📁 输入目录：{input_image_dir}")
    print(f"📁 输出目录：{output_dir}")

    # 设备检查
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n💻 使用设备：{device}")
    if device == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   显存：{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # 加载模型
    print("\n🔧 加载模型...")
    try:
        extractor = StoneExtractor.from_pretrained(
            dino_model_name="IDEA-Research/grounding-dino-base",
            sam2_model_cfg="configs/sam2.1/sam2.1_hiera_l.yaml",
            sam2_checkpoint="checkpoints/sam2.1_hiera_large.pt",
            device=device,
            background_value=128,
            default_padding=20,
            save_quality=95,
        )
    except Exception as e:
        print(f"❌ 模型加载失败：{e}")
        print("请确保已安装必要的依赖:")
        print("   pip install transformers torch pillow numpy tqdm")
        print("   pip install -e . (从 SAM2 仓库)")
        return

    # 检查模型加载状态
    if extractor.dino_model is None:
        print("⚠️  DINO 模型未加载，将使用备用方案")
    if extractor.sam2_predictor is None:
        print("⚠️  SAM2 未加载，将使用边界框裁剪")

    # 批量处理
    print("\n🚀 开始处理图片...")
    success, fail = extractor.process_batch(
        image_paths=image_paths,
        output_dir=output_dir,
        skip_existing=True,
    )

    # 输出结果
    print("\n" + "=" * 60)
    print(f"✅ 处理完成!")
    print(f"   成功：{success} 张")
    print(f"   失败：{fail} 张")
    print(f"   输出目录：{output_dir}")
    print("=" * 60)

    # 释放资源
    print("\n🔄 释放模型资源...")
    extractor.release()
    print("👋 再见!")


if __name__ == "__main__":
    main()
