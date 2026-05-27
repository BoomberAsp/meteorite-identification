import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import PeftModel
import os
from tqdm import tqdm

# 路径配置
def load_model_and_quick_test(lora_model_name="qwen3-vl-lora-meteorite/final_adapter"):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    print(f"当前脚本目录: {BASE_DIR}")
    DATA_ROOT = os.path.join(BASE_DIR, "../../data")
    print(f"数据根目录: {DATA_ROOT}")
    IMAGE_INPUT = os.path.join(DATA_ROOT, "test_images_extracted")
    MODEL_INPUT = os.path.join(BASE_DIR, "../../models")
    print(f"模型输入目录: {MODEL_INPUT}")
    base_model_id = os.path.join(MODEL_INPUT, "Qwen3-VL-8B-Instruct")  # 原始模型路径
    adapter_path = os.path.join(MODEL_INPUT, lora_model_name)  # 保存的LoRA权重路径
    device = "cuda"  

    # 1. 加载基础模型（与训练时相同配置，建议使用 4-bit 量化）
    from transformers import BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    base_model = AutoModelForImageTextToText.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    # 2. 加载 LoRA 适配器
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()  # 切换到推理模式

    # 3. 加载处理器（与训练时相同）
    processor = AutoProcessor.from_pretrained(base_model_id, trust_remote_code=True)
    # 4. 准备测试图像（与训练时使用相同的缩放策略）
    for i in range(1,11):
    
        test_image_path = [os.path.join(IMAGE_INPUT, f"{i:06d}.jpg"), 
                           os.path.join(IMAGE_INPUT, f"{i:06d}.jpeg"),
                           os.path.join(IMAGE_INPUT, f"{i:06d}.png")]  # 优先使用 jpg 格式
        for path in test_image_path:
            if os.path.exists(path):
                test_image_path = path
                break
            else:
                raise FileNotFoundError(f"测试图像 {i:06d} 在 {IMAGE_INPUT} 中未找到！请检查文件名和格式。")
        image = Image.open(test_image_path).convert("RGB")
        # 如果训练时使用了 thumbnail 缩放，这里也要保持一致
        image.thumbnail((720, 720), Image.Resampling.LANCZOS)  # 与训练时相同尺寸

        # 5. 构造指令（与训练时相同的模板）
        instruction = "请观察这张图片，它有80%可能为非陨石（或不是石头），有20%概率为陨石。请你判断它是否为陨石？回答\"是\"或\"否\"。然后简要给出理由。"

        # 6. 生成回答
        # 按照 Qwen3-VL 官方推荐，使用 apply_chat_template 构建对话
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": test_image_path},  # 路径仅用于占位
                    {"type": "text", "text": instruction}
                ]
            }
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # 处理输入
        inputs = processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding=True
        ).to(device)

        # 生成
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,          # 使用贪婪解码（确定性输出）
                temperature=None,
                top_p=None
            )

        # 解码并跳过输入部分
        generated_text = processor.decode(generated_ids[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        print(f"第{i}张图片的模型回答:", generated_text)


def load_model(lora_model_name="qwen3-vl-lora-meteorite_20260504_123315/final_adapter_20260504_123315"):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    print(f"当前脚本目录: {BASE_DIR}")
    DATA_ROOT = os.path.join(BASE_DIR, "../../data")
    print(f"数据根目录: {DATA_ROOT}")
    MODEL_INPUT = os.path.join(BASE_DIR, "../../models")
    print(f"模型输入目录: {MODEL_INPUT}")
    base_model_id = os.path.join(MODEL_INPUT, "Qwen3-VL-8B-Instruct")  # 原始模型路径
    adapter_path = os.path.join(MODEL_INPUT, lora_model_name)  # 保存的LoRA权重路径
    device = "cuda"  

    # 1. 加载基础模型（与训练时相同配置，建议使用 4-bit 量化）
    from transformers import BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    base_model = AutoModelForImageTextToText.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    # 2. 加载 LoRA 适配器
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()  # 切换到推理模式

    # 3. 加载处理器（与训练时相同）
    processor = AutoProcessor.from_pretrained(base_model_id, trust_remote_code=True)
    processor.tokenizer.padding_side = 'left'

    return model, processor, device


def predict(model, processor, test_loader, device):
    """
    对测试集进行预测
    :param model: 已加载 LoRA 的模型
    :param processor: 处理器
    :param test_loader: DataLoader，每个 batch 是列表，元素为 (filename, PIL.Image)
    :param device: 设备
    :return: dict {filename: label}
    """
    model.eval()
    id_to_pred = {}
    # instruction = "请观察这张图片，判断它是否为陨石？回答\"是\"或\"否\"。"
    instruction = "请观察这张图片，它有60%可能为非陨石（或不是石头），有40%概率为陨石。请你判断它是否为陨石？回答\"是\"或\"否\"。" # 这种给定先验概率的提示词效果更好

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting"):
            # 解包 batch: 每个元素是 (filename, image)
            filenames, images = zip(*batch)
            filenames = list(filenames)
            images = list(images)

            # 缩放图像（与训练时一致）
            scaled_images = []
            for img in images:
                img.thumbnail((720, 720), Image.Resampling.LANCZOS)
                scaled_images.append(img)

            # 为每个样本生成文本（使用相同的指令）
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": None},  # 占位，实际图像在 images 中
                        {"type": "text", "text": instruction}
                    ]
                }
            ]
            texts = [processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) for _ in range(len(images))]

            # 批量处理图像和文本
            inputs = processor(
                text=texts,
                images=scaled_images,
                return_tensors="pt",
                padding=True
            ).to(device)

            # 生成
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                temperature=None,
                top_p=None
            )

            # 解码并提取答案
            for filename, gen_id in zip(filenames, generated_ids):
                generated_text = processor.decode(gen_id[inputs['input_ids'].shape[1]:], skip_special_tokens=True)
                # 简单规则：如果回答中包含“是”，则预测为陨石（1），否则为非陨石（0）
                pred_label = 1 if "是" in generated_text else 0
                id_to_pred[filename] = pred_label

    return id_to_pred


if __name__ == "__main__":
    load_model_and_quick_test(lora_model_name="qwen3-vl-lora-meteorite_20260506_211417_biased_training/final_adapter_20260506_211417")