"""
train_lora_augmented.py

增强版训练脚本，集成以下正则化方法：
- TokenMixUp: batch 内随机配对，对 pixel_values 做线性插值
- 多尺度训练: 随机选择 384/512/720 尺寸加载图像
- 域降质数据: 从 qwen3_sft_train_resampled_aug.jsonl 读取（含离线降质增强）

用于与 train_lora_baseline.py 的结果对照消融实验。
"""

import os
import sys
import json
import random
import torch
import numpy as np
from datetime import datetime
from PIL import Image
from datasets import load_dataset
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, TaskType
from transformers import BitsAndBytesConfig
from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState
from transformers.trainer_utils import EvalLoopOutput
from transformers.training_args import TrainingArguments

# ================= 4-bit 量化配置 =================
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# ================= TokenMixUp 超参数 =================
MIXUP_PROB = 0.5      # 每个 batch 触发混合的概率
MIXUP_ALPHA = 0.2     # Beta(α,α) 分布，值小 → λ 倾向 0 或 1

# ================= 多尺度训练配置 =================
TRAIN_SCALES = (384, 512, 720)  # 训练时随机选择
EVAL_SCALE = 720               # 评估时固定尺寸


# ================= 自定义回调 =================
class EpochLossCallback(TrainerCallback):
    def __init__(self):
        self.epoch_losses = []
        self.step_losses = []

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.step_losses.append(logs["loss"])

    def on_epoch_end(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if logs and "loss" in logs:
            epoch_loss = logs.get("loss")
            self.epoch_losses.append(epoch_loss)
            print(f"\n{'='*50}")
            print(f"✅ Epoch {int(state.epoch)} 完成  |  Epoch loss: {epoch_loss:.4f}")
            if len(self.epoch_losses) >= 2:
                change = epoch_loss - self.epoch_losses[-2]
                direction = "↓" if change < 0 else "↑"
                print(f"   {direction} {abs(change):.4f}")
            print(f"{'='*50}")


class DebugTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epoch_callback = EpochLossCallback()
        self.add_callback(self.epoch_callback)

    def evaluation_loop(self, dataloader, description, prediction_loss_only,
                        ignore_keys=None, metric_key_prefix="eval"):
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        for batch in dataloader:
            batch = {k: v.to(self.args.device) for k, v in batch.items()}
            with torch.inference_mode():
                outputs = self.model(**batch)
                loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
                total_loss += loss.item()
                num_batches += 1
            del batch, outputs
            torch.cuda.empty_cache()
        avg_loss = total_loss / num_batches
        metrics = {f"{metric_key_prefix}_loss": avg_loss}
        return EvalLoopOutput(predictions=None, label_ids=None, metrics=metrics,
                              num_samples=len(dataloader.dataset))


# ================= 1. 配置参数 =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(BASE_DIR, "../../data")
MODEL_ROOT = os.path.join(BASE_DIR, "../../models")
MODEL_ID = os.path.join(MODEL_ROOT, "Qwen3-VL-8B-Instruct")

# 增强数据路径（含离线域降质）
AUG_DATA_PATH = os.path.join(DATA_ROOT, "qwen3_sft_train_resampled_aug.jsonl")
# 如果增强数据不存在则 fallback 到基线数据
if not os.path.exists(AUG_DATA_PATH):
    DATA_PATH = os.path.join(DATA_ROOT, "qwen3_sft_train_baseline.jsonl")
    print("⚠️  未找到增强数据，使用基线数据")
else:
    DATA_PATH = AUG_DATA_PATH

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = os.path.join(MODEL_ROOT, f"qwen3-vl-lora-meteorite_aug_{TIMESTAMP}")

VAL_SPLIT_RATIO = 0.1


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    return {"eval_loss": float(np.mean(predictions))}


def train():
    print("=" * 60)
    print("🔧 增强训练：TokenMixUp + 多尺度 + 域降质数据")
    print(f"   MixUp: prob={MIXUP_PROB}, alpha={MIXUP_ALPHA}")
    print(f"   多尺度: {TRAIN_SCALES} (评估固定 {EVAL_SCALE})")
    print(f"   数据: {os.path.basename(DATA_PATH)}")
    print("=" * 60)

    # ================= 2. 加载模型与处理器 =================
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    processor.tokenizer.padding_side = 'left'

    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    # ================= 3. 配置 LoRA =================
    lora_config = LoraConfig(
        r=64,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ================= 4. 加载数据集 =================
    raw_dataset = load_dataset("json", data_files=DATA_PATH, split="train")
    val_size = int(len(raw_dataset) * VAL_SPLIT_RATIO)
    split = raw_dataset.train_test_split(test_size=val_size, seed=42)
    train_dataset = split["train"]
    val_dataset = split["test"]

    EVAL_SUBSET_SIZE = 100
    eval_subset = val_dataset.shuffle(seed=42).select(
        range(min(EVAL_SUBSET_SIZE, len(val_dataset))))
    full_val_dataset = val_dataset

    print(f"\n 训练集: {len(train_dataset)} | 验证集: {len(val_dataset)} | 评估子集: {len(eval_subset)}")

    # ================= 5. 定义 Collator =================
    class QwenVLDataCollator:
        def __init__(self, processor, mixup_prob=MIXUP_PROB, mixup_alpha=MIXUP_ALPHA,
                     scales=TRAIN_SCALES, is_training=True):
            self.processor = processor
            self.assistant_start_tokens = processor.tokenizer.encode(
                "<|im_start|>assistant\n", add_special_tokens=False
            )
            self.im_end_id = processor.tokenizer.encode(
                "<|im_end|>", add_special_tokens=False)[0]
            self.mixup_prob = mixup_prob
            self.mixup_alpha = mixup_alpha
            self.scales = scales
            self.is_training = is_training

        def __call__(self, examples):
            import torch

            images = []
            for ex in examples:
                img = Image.open(ex["image_path"]).convert("RGB")
                if self.is_training and self.scales:
                    scale = random.choice(self.scales)
                else:
                    scale = EVAL_SCALE
                img.thumbnail((scale, scale), Image.Resampling.LANCZOS)
                images.append(img)

            texts = []
            for ex in examples:
                messages = [
                    {"role": "user", "content": [
                        {"type": "image", "image": ex["image_path"]},
                        {"type": "text", "text": ex["instruction"]}
                    ]},
                    {"role": "assistant", "content": [{"type": "text", "text": ex["output"]}]}
                ]
                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False)
                texts.append(text)

            batch = self.processor(
                text=texts, images=images, return_tensors="pt", padding=True)

            # 构建 labels
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = []
            for i in range(len(input_ids)):
                seq = input_ids[i].tolist()
                labels_i = input_ids[i].clone()
                assistant_start_pos = -1
                ast_tokens = self.assistant_start_tokens
                for j in range(len(seq) - len(ast_tokens) + 1):
                    if seq[j:j + len(ast_tokens)] == ast_tokens:
                        assistant_start_pos = j + len(ast_tokens)
                        break
                if assistant_start_pos != -1:
                    labels_i[:assistant_start_pos] = -100
                    labels_i[attention_mask[i] == 0] = -100
                else:
                    labels_i[:] = -100
                labels.append(labels_i)

            batch["labels"] = torch.stack(labels)

            # ===== TokenMixUp =====
            if self.is_training and np.random.random() < self.mixup_prob and len(examples) > 1:
                perm = torch.randperm(len(examples))
                lam = float(np.random.beta(self.mixup_alpha, self.mixup_alpha))
                pv = batch["pixel_values"]
                batch["pixel_values"] = lam * pv + (1 - lam) * pv[perm]

            return batch

    # ================= 6. 训练参数 =================
    PER_DEVICE_BATCH = 2
    GRADIENT_ACCUMULATION = 4
    effective_batch_size = PER_DEVICE_BATCH * GRADIENT_ACCUMULATION
    steps_per_epoch = len(train_dataset) // effective_batch_size
    eval_steps = max(1, steps_per_epoch // 10)

    print(f"\n 有效 batch size: {effective_batch_size}")
    print(f" 每 epoch 步数: {steps_per_epoch}")
    print(f" 评估间隔: 每 {eval_steps} 步")

    TENSORBOARD_LOG_DIR = os.path.join(DATA_ROOT, "logs", "tensorboard")
    os.makedirs(TENSORBOARD_LOG_DIR, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=PER_DEVICE_BATCH,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=1e-4,
        num_train_epochs=3,
        bf16=True,
        logging_steps=10,
        save_total_limit=3,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        weight_decay=0.05,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
        optim="adamw_torch",
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
        per_device_eval_batch_size=1,
        fp16_full_eval=False,
        eval_strategy="steps",
        eval_steps=eval_steps,
        eval_accumulation_steps=1,
        save_strategy="steps",
        save_steps=eval_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_dir=TENSORBOARD_LOG_DIR,
        report_to="tensorboard",
    )

    # ================= 7. 创建 Trainer =================
    trainer = DebugTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_subset,
        data_collator=QwenVLDataCollator(processor, is_training=True),
        compute_metrics=compute_metrics,
    )

    print(f"\n🚀 开始训练...")
    print(f"  数据量: {len(train_dataset)} | Epochs: {training_args.num_train_epochs}")
    print(f"  学习率: {training_args.learning_rate} | 输出: {OUTPUT_DIR}\n")

    trainer.train()

    # ================= 8. 训练汇总 =================
    print("\n" + "=" * 60)
    print("🎉 训练完成！")
    print("=" * 60)

    print("\n📊 完整验证集最终评估...")
    torch.cuda.empty_cache()
    trainer.eval_dataset = full_val_dataset
    eval_results = trainer.evaluate()
    for k, v in eval_results.items():
        print(f"  {k}: {v:.4f}")

    if hasattr(trainer, 'epoch_callback'):
        epoch_losses = trainer.epoch_callback.epoch_losses
        if epoch_losses:
            print(f"\n📈 Epoch Loss 历史:")
            for i, loss in enumerate(epoch_losses, 1):
                print(f"  Epoch {i}: {loss:.4f}")
            if len(epoch_losses) >= 3:
                initial = np.mean(epoch_losses[:3])
                final = np.mean(epoch_losses[-3:])
                print(f"\n📈 Loss: {initial:.4f} → {final:.4f} ({(final-initial)/initial*100:+.1f}%)")

    final_adapter_dir = os.path.join(OUTPUT_DIR, f"final_adapter_{TIMESTAMP}")
    trainer.save_model(final_adapter_dir)
    processor.save_pretrained(final_adapter_dir)
    print(f"\n✅ 权重已保存至 {final_adapter_dir}")


if __name__ == "__main__":
    train()
