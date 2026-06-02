"""
train_lora_combo.py

联合增强训练脚本：
- Visual-Text Co-Cropping: 图像层裁剪 + instruction 改写（50% 概率）
- TokenMixUp: 嵌入层 pixel_values 混合（50% 概率）
- 多尺度训练: 随机 384/512/720
三者可独立开关，协同工作。
"""

import os, sys, json, random, re, torch, numpy as np
from datetime import datetime
from PIL import Image
from typing import Optional, List, Dict, Tuple
from datasets import load_dataset
from transformers import (
    AutoModelForImageTextToText, AutoProcessor, TrainingArguments, Trainer,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from transformers import BitsAndBytesConfig
from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState
from transformers.trainer_utils import EvalLoopOutput

# ================= 量化配置 =================
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)

# ================= 增强开关 =================
ENABLE_COCROP = True       # Visual-Text Co-Cropping
ENABLE_MIXUP = True        # TokenMixUp
ENABLE_MULTISCALE = True   # 多尺度

COCROP_PROB = 0.5
MIXUP_PROB = 0.5
MIXUP_ALPHA = 0.2
TRAIN_SCALES = (384, 512, 720)
EVAL_SCALE = 720

# ================= Co-Cropper 关键词 =================
KEY_VISUAL_TERMS = ["熔壳", "气印", "金属颗粒", "矿物", "纹理", "颜色", "形状"]

# ================= 回调 =================
class EpochLossCallback(TrainerCallback):
    def __init__(self):
        self.epoch_losses = []
    def on_epoch_end(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.epoch_losses.append(logs.get("loss"))
            print(f"\n{'='*50}\n✅ Epoch {int(state.epoch)} | loss={logs['loss']:.4f}\n{'='*50}")

class DebugTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_callback(EpochLossCallback())
    def evaluation_loop(self, dataloader, description, prediction_loss_only,
                        ignore_keys=None, metric_key_prefix="eval"):
        self.model.eval()
        total_loss, n = 0.0, 0
        for batch in dataloader:
            batch = {k: v.to(self.args.device) for k, v in batch.items()}
            with torch.inference_mode():
                outputs = self.model(**batch)
                total_loss += (outputs.loss if hasattr(outputs, 'loss') else outputs[0]).item()
                n += 1
            del batch, outputs
            torch.cuda.empty_cache()
        return EvalLoopOutput(predictions=None, label_ids=None,
                              metrics={f"{metric_key_prefix}_loss": total_loss/n}, num_samples=len(dataloader.dataset))

# ================= 配置 =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(BASE_DIR, "../../data")
MODEL_ROOT = os.path.join(BASE_DIR, "../../models")
MODEL_ID = os.path.join(MODEL_ROOT, "Qwen3-VL-8B-Instruct")
DATA_PATH = os.path.join(DATA_ROOT, "qwen3_sft_train_curated.jsonl")

# 从 Curated/VC2 继续微调
CHECKPOINT_PATH = os.path.join(MODEL_ROOT, "qwen3-vl-lora-meteorite_vc2_20260602_180603", "checkpoint-516")

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = os.path.join(MODEL_ROOT, f"qwen3-vl-lora-meteorite_combo_{TIMESTAMP}")
VAL_SPLIT_RATIO = 0.1


# ================= Co-Cropper（在线增强） =================
def cocrop_image_text(image: Image.Image, instruction: str, output: str) -> Tuple[Image.Image, str]:
    """图文联合裁剪：检测关键词区域 → 安全裁剪 → 改写 instruction"""
    matched = [t for t in KEY_VISUAL_TERMS if t in output]
    if not matched:
        return image, instruction

    w, h = image.size
    # 简化版显著性：基于边缘检测估算感兴趣区域
    gray = image.convert('L')
    edges = gray.filter(Image.Filter.FIND_EDGES)
    edge_arr = np.array(edges)
    # 找边缘密集区中心
    ys, xs = np.where(edge_arr > 30)
    if len(xs) < 100:
        return image, instruction

    x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
    area_ratio = (x2-x1)*(y2-y1) / (w*h)
    if area_ratio < 0.05 or area_ratio > 0.8:
        return image, instruction

    # 安全裁剪
    scale = np.random.uniform(0.8, 1.2)
    cx, cy = (x1+x2)//2, (y1+y2)//2
    crop_w = min(int((x2-x1)*scale), w)
    crop_h = min(int((y2-y1)*scale), h)
    left = max(0, min(cx - crop_w//2, w - crop_w))
    top = max(0, min(cy - crop_h//2, h - crop_h))
    cropped = image.crop((left, top, left+crop_w, top+crop_h))

    # 改写 instruction: 聚焦关键词
    focus = matched[0]
    pattern = r"观察这张图片中石头的视觉特征（[^）]*）"
    repl = f"重点观察图中【{focus}】的细节特征"
    new_inst = re.sub(pattern, repl, instruction)
    if new_inst == instruction:
        new_inst += f" 请特别注意图中的{focus}特征。"

    return cropped, new_inst


# ================= Collator（联合增强） =================
class ComboCollator:
    def __init__(self, processor, is_training=True):
        self.processor = processor
        self.is_training = is_training
        self.ast = processor.tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)

    def __call__(self, examples):
        import torch

        # Step 1: Co-Cropping (online, image-level)
        if self.is_training and ENABLE_COCROP:
            for i, ex in enumerate(examples):
                if np.random.random() < COCROP_PROB:
                    try:
                        raw_img = Image.open(ex["image_path"]).convert("RGB")
                        cropped_img, new_inst = cocrop_image_text(raw_img, ex["instruction"], ex["output"])
                        ex = dict(ex)  # shallow copy
                        ex["_cocrop_image"] = cropped_img
                        ex["instruction"] = new_inst
                        examples[i] = ex
                    except Exception:
                        pass

        # Step 2: Load images (with multi-scale)
        images = []
        for ex in examples:
            img = ex.get("_cocrop_image")
            if img is None:
                img = Image.open(ex["image_path"]).convert("RGB")
            scale = random.choice(TRAIN_SCALES) if (self.is_training and ENABLE_MULTISCALE) else EVAL_SCALE
            img.thumbnail((scale, scale), Image.Resampling.LANCZOS)
            images.append(img)

        # Step 3: Build texts
        texts = []
        for ex in examples:
            msgs = [
                {"role": "user", "content": [{"type": "image", "image": ex["image_path"]},
                                              {"type": "text", "text": ex["instruction"]}]},
                {"role": "assistant", "content": [{"type": "text", "text": ex["output"]}]},
            ]
            texts.append(self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False))

        # Step 4: Processor
        batch = self.processor(text=texts, images=images, return_tensors="pt", padding=True)

        # Step 5: Labels
        labels = []
        for i in range(len(batch["input_ids"])):
            seq = batch["input_ids"][i].tolist()
            lbl = batch["input_ids"][i].clone()
            pos = -1
            for j in range(len(seq) - len(self.ast) + 1):
                if seq[j:j+len(self.ast)] == self.ast:
                    pos = j + len(self.ast); break
            if pos != -1:
                lbl[:pos] = -100
                lbl[batch["attention_mask"][i] == 0] = -100
            else:
                lbl[:] = -100
            labels.append(lbl)
        batch["labels"] = torch.stack(labels)

        # Step 6: TokenMixUp (embedding-level)
        if self.is_training and ENABLE_MIXUP and np.random.random() < MIXUP_PROB and len(examples) > 1:
            grid = batch["image_grid_thw"]
            n_tokens = (grid[:, 1] * grid[:, 2]).tolist()
            offsets = [0]
            for c in n_tokens: offsets.append(offsets[-1] + c)

            lam = float(np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA))
            pv = batch["pixel_values"]

            groups = {}
            for i, tc in enumerate(n_tokens):
                groups.setdefault(tc, []).append(i)

            mixed = [None] * len(examples)
            for tc, indices in groups.items():
                if len(indices) >= 2:
                    perm = torch.randperm(len(indices)).tolist()
                    for pos_idx, idx in enumerate(indices):
                        p = indices[perm[pos_idx]]
                        a_s, a_e = offsets[idx], offsets[idx+1]
                        b_s, b_e = offsets[p], offsets[p+1]
                        mixed[idx] = lam * pv[a_s:a_e] + (1-lam) * pv[b_s:b_e]
                else:
                    idx = indices[0]
                    mixed[idx] = pv[offsets[idx]:offsets[idx+1]]
            batch["pixel_values"] = torch.cat(mixed, dim=0)

        return batch


# ================= 训练 =================
def train():
    print("="*60)
    print("🔧 联合增强：Co-Cropping + TokenMixUp + 多尺度")
    print(f"   Co-Crop prob={COCROP_PROB} | MixUp prob={MIXUP_PROB} alpha={MIXUP_ALPHA}")
    print(f"   多尺度={TRAIN_SCALES}")
    print(f"   数据: {os.path.basename(DATA_PATH)}")
    print(f"   起始权重: {os.path.basename(CHECKPOINT_PATH)}")
    print("="*60)

    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    processor.tokenizer.padding_side = 'left'

    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, quantization_config=bnb_config, device_map="auto", trust_remote_code=True)
    model = PeftModel.from_pretrained(model, CHECKPOINT_PATH, is_trainable=True)
    print("✅ 从 checkpoint 加载 LoRA 权重")
    model.print_trainable_parameters()

    raw = load_dataset("json", data_files=DATA_PATH, split="train")
    val_size = int(len(raw) * VAL_SPLIT_RATIO)
    split = raw.train_test_split(test_size=val_size, seed=42)
    train_ds, val_ds = split["train"], split["test"]
    eval_sub = val_ds.shuffle(seed=42).select(range(min(100, len(val_ds))))
    print(f"\n 训练: {len(train_ds)} | 验证: {len(val_ds)} | 评估: {len(eval_sub)}")

    steps_per_epoch = len(train_ds) // 8
    eval_steps = max(1, steps_per_epoch // 10)

    tb_dir = os.path.join(DATA_ROOT, "logs", "tensorboard"); os.makedirs(tb_dir, exist_ok=True)

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2, gradient_accumulation_steps=4,
        learning_rate=5e-5, num_train_epochs=1, bf16=True,
        logging_steps=10, save_total_limit=3,
        lr_scheduler_type="cosine", warmup_ratio=0.1, weight_decay=0.05,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False, optim="adamw_torch",
        dataloader_num_workers=2, dataloader_pin_memory=True,
        per_device_eval_batch_size=1, fp16_full_eval=False,
        eval_strategy="steps", eval_steps=eval_steps,
        save_strategy="steps", save_steps=eval_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss", greater_is_better=False,
        logging_dir=tb_dir, report_to="tensorboard",
    )

    trainer = DebugTrainer(
        model=model, args=args, train_dataset=train_ds, eval_dataset=eval_sub,
        data_collator=ComboCollator(processor, is_training=True),
    )

    print(f"\n🚀 训练: {len(train_ds)} 条 | Epochs={args.num_train_epochs} | lr={args.learning_rate}")
    trainer.train()

    print("\n🎉 完成！")
    torch.cuda.empty_cache()
    trainer.eval_dataset = val_ds
    for k, v in trainer.evaluate().items():
        print(f"  {k}: {v:.4f}")

    adapter = os.path.join(OUTPUT_DIR, f"final_adapter_{TIMESTAMP}")
    trainer.save_model(adapter); processor.save_pretrained(adapter)
    print(f"\n✅ {adapter}")


if __name__ == "__main__":
    train()
