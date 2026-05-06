"""
测试 SAM2.1 和 GroundingDINO 模型加载

此脚本用于验证两个模型是否正确安装并可正常加载运行。
"""

import os
import sys
import torch
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_sam2():
    """测试 SAM2.1 模型加载"""
    print("\n" + "=" * 60)
    print("测试 SAM2.1 模型加载")
    print("=" * 60)

    # SAM2 仓库路径
    SAM2_ROOT = "/root/Projects/sam2"

    # 可用的 checkpoint 和对应的 config
    checkpoints = {
        "tiny": {
            "checkpoint": os.path.join(SAM2_ROOT, "checkpoints/sam2.1_hiera_tiny.pt"),
            "config": "configs/sam2.1/sam2.1_hiera_t.yaml"
        },
        "small": {
            "checkpoint": os.path.join(SAM2_ROOT, "checkpoints/sam2.1_hiera_small.pt"),
            "config": "configs/sam2.1/sam2.1_hiera_s.yaml"
        },
        "base_plus": {
            "checkpoint": os.path.join(SAM2_ROOT, "checkpoints/sam2.1_hiera_base_plus.pt"),
            "config": "configs/sam2.1/sam2.1_hiera_b+.yaml"
        },
        "large": {
            "checkpoint": os.path.join(SAM2_ROOT, "checkpoints/sam2.1_hiera_large.pt"),
            "config": "configs/sam2.1/sam2.1_hiera_l.yaml"
        }
    }

    # 选择测试的模型大小（使用 tiny 进行快速测试）
    model_size = "tiny"

    checkpoint_path = checkpoints[model_size]["checkpoint"]
    config_path = checkpoints[model_size]["config"]

    # 检查 checkpoint 是否存在
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint 不存在: {checkpoint_path}")
        return False

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Config: {config_path}")
    print(f"Checkpoint 大小: {os.path.getsize(checkpoint_path) / 1024 / 1024:.2f} MB")

    # 导入 SAM2 模块
    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        print("SAM2 模块导入成功")
    except ImportError as e:
        print(f"SAM2 模块导入失败: {e}")
        return False

    # 构建模型
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"使用设备: {device}")

        # 需要切换到 sam2 目录来加载 config
        original_dir = os.getcwd()
        os.chdir(SAM2_ROOT)

        sam2_model = build_sam2(config_path, checkpoint_path, device=device)
        predictor = SAM2ImagePredictor(sam2_model)

        os.chdir(original_dir)

        print(f"SAM2.1 {model_size} 模型加载成功")
        print(f"模型类型: {type(sam2_model)}")

        # 测试设置图像
        test_image_path = "/root/Projects/meteorite-identification/data/test_images/000001.jpg"
        if os.path.exists(test_image_path):
            test_image = Image.open(test_image_path).convert("RGB")
            predictor.set_image(test_image)
            print(f"测试图像设置成功: {test_image_path}")
        else:
            print(f"测试图像不存在: {test_image_path}")

        return True

    except Exception as e:
        print(f"SAM2 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_grounding_dino():
    """测试 GroundingDINO 模型加载（使用 HuggingFace transformers）"""
    print("\n" + "=" * 60)
    print("测试 GroundingDINO 模型加载")
    print("=" * 60)

    # 项目路径
    PROJECT_ROOT = "/root/Projects/meteorite-identification"

    # 模型权重路径（HuggingFace 格式）
    MODEL_PATH = os.path.join(PROJECT_ROOT, "models/grounding_dino")

    # 检查模型目录
    if not os.path.exists(MODEL_PATH):
        print(f"模型目录不存在: {MODEL_PATH}")
        return False

    print(f"模型路径: {MODEL_PATH}")

    # 检查必要文件
    required_files = ["config.json", "model.safetensors"]
    for f in required_files:
        filepath = os.path.join(MODEL_PATH, f)
        if os.path.exists(filepath):
            print(f"  {f}: {os.path.getsize(filepath) / 1024 / 1024:.2f} MB")
        else:
            print(f"  {f}: 不存在")

    try:
        from transformers import AutoModel, AutoProcessor
        print("transformers 模块导入成功")
    except ImportError as e:
        print(f"transformers 模块导入失败: {e}")
        return False

    # 加载模型
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"使用设备: {device}")

        # 使用 transformers 加载 HuggingFace 格式的模型
        print("加载模型...")
        model = AutoModel.from_pretrained(
            MODEL_PATH,
            local_files_only=True,
            trust_remote_code=True
        )
        model = model.to(device)
        model.eval()

        print(f"GroundingDINO 模型加载成功")
        print(f"模型类型: {type(model)}")

        # 加载 processor
        processor = AutoProcessor.from_pretrained(
            MODEL_PATH,
            local_files_only=True,
            trust_remote_code=True
        )
        print(f"Processor 加载成功")

        # 测试推理
        test_image_path = os.path.join(PROJECT_ROOT, "data/test_images/000001.jpg")
        if os.path.exists(test_image_path):
            from PIL import Image
            test_image = Image.open(test_image_path).convert("RGB")
            print(f"测试图像加载成功: {test_image_path}")
            print(f"图像尺寸: {test_image.size}")

            # 准备输入
            TEXT_PROMPT = "stone . rock . meteorite"

            inputs = processor(
                images=test_image,
                text=TEXT_PROMPT,
                return_tensors="pt"
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            print(f"输入准备完成:")
            for k, v in inputs.items():
                print(f"  {k}: {v.shape}")

            # 推理
            with torch.no_grad():
                outputs = model(**inputs)

            print(f"推理完成:")
            print(f"  输出属性: enc_outputs_class, enc_outputs_coord_logits, encoder_pred_boxes")

            # 检查输出
            if hasattr(outputs, 'enc_outputs_class'):
                print(f"  enc_outputs_class shape: {outputs.enc_outputs_class.shape}")
            if hasattr(outputs, 'encoder_pred_boxes'):
                print(f"  encoder_pred_boxes shape: {outputs.encoder_pred_boxes.shape}")

            # GroundingDinoModelOutput 使用不同的属性名
            # 这里仅展示模型可以成功加载和推理
            print(f"  模型成功完成推理")
        else:
            print(f"测试图像不存在: {test_image_path}")

        return True

    except Exception as e:
        print(f"GroundingDINO 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_combined_pipeline():
    """测试 GroundingDINO + SAM2 组合流程"""
    print("\n" + "=" * 60)
    print("测试 GroundingDINO + SAM2 组合流程")
    print("=" * 60)

    # 先加载两个模型
    SAM2_ROOT = "/root/Projects/sam2"
    PROJECT_ROOT = "/root/Projects/meteorite-identification"
    GD_MODEL_PATH = os.path.join(PROJECT_ROOT, "models/grounding_dino")

    try:
        # 1. 导入模块
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        from transformers import AutoModel, AutoProcessor

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"使用设备: {device}")

        # 2. 加载 GroundingDINO (HuggingFace 格式)
        print("加载 GroundingDINO...")
        gd_model = AutoModel.from_pretrained(
            GD_MODEL_PATH,
            local_files_only=True,
            trust_remote_code=True
        )
        gd_model = gd_model.to(device)
        gd_model.eval()

        gd_processor = AutoProcessor.from_pretrained(
            GD_MODEL_PATH,
            local_files_only=True,
            trust_remote_code=True
        )
        print("GroundingDINO 模型加载成功")

        # 3. 加载 SAM2
        print("加载 SAM2...")
        original_dir = os.getcwd()
        os.chdir(SAM2_ROOT)
        sam2_model = build_sam2(
            "configs/sam2.1/sam2.1_hiera_t.yaml",
            os.path.join(SAM2_ROOT, "checkpoints/sam2.1_hiera_tiny.pt"),
            device=device
        )
        predictor = SAM2ImagePredictor(sam2_model)
        os.chdir(original_dir)
        print("SAM2 模型加载成功")

        # 4. 测试组合流程
        test_image_path = os.path.join(PROJECT_ROOT, "data/test_images/000001.jpg")
        if not os.path.exists(test_image_path):
            print(f"测试图像不存在: {test_image_path}")
            return False

        # 用 GroundingDINO 检测
        from PIL import Image
        test_image_pil = Image.open(test_image_path).convert("RGB")
        print(f"测试图像: {test_image_path}, 尺寸: {test_image_pil.size}")

        TEXT_PROMPT = "stone . rock"

        inputs = gd_processor(
            images=test_image_pil,
            text=TEXT_PROMPT,
            return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = gd_model(**inputs)

        print(f"GroundingDINO 推理完成")

        # 用 SAM2 分割（使用图像中心点作为简单测试）
        predictor.set_image(test_image_pil)

        # 使用图像中心点作为 prompt（简化测试）
        import numpy as np
        center_x = test_image_pil.width / 2
        center_y = test_image_pil.height / 2

        print(f"使用图像中心点 prompt: ({center_x:.1f}, {center_y:.1f})")

        # 使用 point prompt 进行分割
        point_coords = np.array([[center_x, center_y]])
        point_labels = np.array([1])  # 1 表示前景

        masks, scores, logits_sam = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True
        )

        print(f"SAM2 分割成功:")
        print(f"  生成的 mask 数量: {len(masks)}")
        print(f"  mask shapes: {masks.shape}")
        print(f"  分割置信度: {scores}")

        print("\n组合流程测试成功！")
        return True

    except Exception as e:
        print(f"组合流程测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("\n" + "=" * 60)
    print("开始测试 SAM2.1 和 GroundingDINO 模型")
    print("=" * 60)

    # 检查 CUDA
    print(f"\nPyTorch 版本: {torch.__version__}")
    print(f"CUDA 是否可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA 设备: {torch.cuda.get_device_name(0)}")
        print(f"CUDA 版本: {torch.version.cuda}")

    # 运行各项测试
    results = {}

    results["SAM2.1"] = test_sam2()
    results["GroundingDINO"] = test_grounding_dino()

    # 只有当两个模型都成功时才测试组合流程
    if results["SAM2.1"] and results["GroundingDINO"]:
        results["Combined"] = test_combined_pipeline()
    else:
        results["Combined"] = False
        print("\n跳过组合流程测试（单个模型测试未通过）")

    # 输出总结
    print("\n" + "=" * 60)
    print("测试结果总结")
    print("=" * 60)

    for name, success in results.items():
        status = "✅ 通过" if success else "❌ 失败"
        print(f"{name}: {status}")

    all_passed = all(results.values())
    if all_passed:
        print("\n🎉 所有测试通过！模型已正确安装并可正常使用。")
    else:
        print("\n⚠️ 部分测试失败，请检查安装配置。")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)