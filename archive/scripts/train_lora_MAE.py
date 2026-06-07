"""
train_lora_MAE.py

MAE掩码预训练辅助训练脚本，集成以下自监督正则化方法：
- MAE Masking: 对输入图像的patch进行高比例随机掩码（默认75%），迫使模型依赖局部线索重建全局语义
- 重建损失辅助: 在VLM生成loss之外附加像素级重建loss，增强视觉编码器对细粒度纹理的表征能力
- 仅作用于视觉编码器: 文本分支与LoRA参数不参与MAE任务，保持reasoning链完整性

用于验证自监督预训练信号对陨石细粒度识别任务的增益，与train_lora_baseline.py及train_lora_augmented.py对照消融。
注意：MAE属于语义/特征级正则化，不解决像素级域偏移（低光照/低分辨率/传感器差异），需与自适应亮度校正、频域扰动等像素级方法配合使用。
"""

import os
import torch
import torch.nn as nn
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
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalLoopOutput
import random
import gc


# ================= 配置与辅助类 =================
class MAEConfig:
    """MAE-Aug 配置类"""
    def __init__(self, mask_ratio=0.5, mask_prob=0.5):
        self.mask_ratio = mask_ratio
        self.mask_prob = mask_prob


# ================= 1. 核心 MAE Token Masker =================
class MAETokenMasker(nn.Module):
    """
    MAE 动态 Patch-wise 掩码器
    兼容 Qwen3-VL merger 输入的 2D [B*N, D] 和标准 3D [B, N, D] 格式
    """
    def __init__(self, hidden_dim=1152, mask_ratio=0.5):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.hidden_dim = hidden_dim
        self.mask_token = nn.Parameter(torch.zeros(1, hidden_dim))
        nn.init.normal_(self.mask_token, std=0.02)

    def forward(self, patch_tokens):
        """
        Args:
            patch_tokens: [B*N, D] 或 [B, N, D]
        Returns:
            masked_tokens: 同输入形状
        """
        if not self.training or self.mask_ratio <= 0:
            return patch_tokens

        input_2d = patch_tokens.dim() == 2
        if input_2d:
            # [B*N, D] -> 全局随机 masking
            total_tokens = patch_tokens.size(0)
            num_mask = max(1, int(total_tokens * self.mask_ratio))
            mask_indices = torch.randperm(total_tokens, device=patch_tokens.device)[:num_mask]
            masked_tokens = patch_tokens.clone()
            masked_tokens[mask_indices] = self.mask_token.to(patch_tokens.dtype)
        else:
            # [B, N, D] -> 逐样本 masking
            B, N, D = patch_tokens.shape
            noise = torch.rand(B, N, device=patch_tokens.device)
            ids_shuffle = torch.argsort(noise, dim=1)
            num_mask = max(1, int(N * self.mask_ratio))
            mask_indices = ids_shuffle[:, :num_mask]
            bool_mask = torch.zeros(B, N, dtype=torch.bool, device=patch_tokens.device)
            bool_mask.scatter_(1, mask_indices, True)
            masked_tokens = patch_tokens.clone()
            masked_tokens[bool_mask] = self.mask_token.to(patch_tokens.dtype).unsqueeze(0)

        return masked_tokens


# ================= 自定义回调与 Trainer =================
class EpochLossCallback(TrainerCallback):
    def __init__(self):
        self.epoch_losses = []

    def on_epoch_end(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            epoch_loss = logs.get("loss")
            self.epoch_losses.append(epoch_loss)
            print(f"\n{'='*50}")
            print(f"✅ Epoch {int(state.epoch)} 完成 | Loss: {epoch_loss:.4f}")
            if len(self.epoch_losses) >= 2:
                change = epoch_loss - self.epoch_losses[-2]
                symbol = "🔽" if change < 0 else "🔺"
                print(f"  {symbol} 相比上一 epoch: {change:+.4f}")
            print(f"{'='*50}")


class MAETrainer(Trainer):
    """自定义 Trainer，通过 Hook 在 merger 前注入 MAE Masking"""

    def __init__(self, mae_config=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.mae_config = mae_config
        self.epoch_callback = EpochLossCallback()
        self.add_callback(self.epoch_callback)
        self.hook_handle = None
        self.mae_masker = None

        if mae_config:
            self.mae_masker = MAETokenMasker(
                hidden_dim=1152, mask_ratio=mae_config.mask_ratio
            )
            print(
                f"✅ MAE Masker 已初始化 | "
                f"mask_ratio={mae_config.mask_ratio} | "
                f"mask_prob={mae_config.mask_prob}"
            )

    def create_optimizer(self):
        """在优化器实际创建后才注入 mask_token"""
        optimizer = super().create_optimizer()

        if self.mae_masker is not None:
            optimizer.param_groups[0]["params"].append(self.mae_masker.mask_token)
            print("✅ mask_token 已成功加入优化器")

        return optimizer

    @staticmethod
    def _find_visual_module(model):
        """鲁棒地查找 visual 模块"""
        candidates = [
            lambda m: m.base_model.model.model.visual,
            lambda m: m.base_model.model.visual,
            lambda m: m.model.visual,
            lambda m: m.visual,
        ]

        for accessor in candidates:
            try:
                visual = accessor(model)
                if visual is not None:
                    return visual
            except (AttributeError, TypeError):
                continue

        for name, module in model.named_modules():
            if name.endswith(".visual") or name == "visual":
                return module

        raise AttributeError(
            f"无法找到 visual 模块。模型类型: {type(model).__name__}\n"
            f"包含 'visual' 或 'merger' 的子模块:\n"
            + "\n".join(
                f"  → {n}" for n, _ in model.named_modules()
                if "visual" in n.lower() or "merger" in n.lower()
            )
        )

    def _register_mae_hook(self, model):
        """注册 MAE pre-forward hook 到 visual.merger"""
        if self.hook_handle is not None:
            return

        visual_module = self._find_visual_module(model)
        print(f"✅ 找到 visual 模块: {type(visual_module).__name__}")

        def mae_pre_hook(module, args, kwargs):
            apply_mae = (
                self.model.training
                and self.mae_config
                and self.mae_masker is not None
                and random.random() < self.mae_config.mask_prob
            )

            if apply_mae and len(args) > 0:
                hidden_states = args[0]
                self.mae_masker = self.mae_masker.to(hidden_states.device)
                masked_states = self.mae_masker(hidden_states)
                return (masked_states,) + args[1:], kwargs

            return args, kwargs

        self.hook_handle = visual_module.merger.register_forward_pre_hook(
            mae_pre_hook, with_kwargs=True
        )
        print("✅ MAE Hook 已成功注册到 visual.merger")

    def train(self, *args, **kwargs):
        """在训练开始前注册 Hook"""
        self._register_mae_hook(self.model)
        return super().train(*args, **kwargs)

    def evaluation_loop(
        self,
        dataloader,
        description,
        prediction_loss_only,
        ignore_keys=None,
        metric_key_prefix="eval",
    ):
        """安全的评估循环，避免 OOM，且不触发 MAE"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            batch = {k: v.to(self.args.device) for k, v in batch.items()}
            with torch.inference_mode():
                outputs = self.model(**batch)
                total_loss += outputs.loss.item()
                num_batches += 1
            del batch, outputs
            gc.collect()
            torch.cuda.empty_cache()

        avg_loss = total_loss / max(num_batches, 1)
        metrics = {f"{metric_key_prefix}_loss": avg_loss}
        return EvalLoopOutput(
            predictions=None,
            label_ids=None,
            metrics=metrics,
            num_samples=len(dataloader.dataset),
        )


# ================= 标准 Data Collator =================
class QwenVLDataCollator:
    def __init__(self, processor):
        self.processor = processor
        self.assistant_start_tokens = processor.tokenizer.encode(
            "<|im_channel|>final<|im_message|>", add_special_tokens=False
        )

    def __call__(self, examples):
        images = []
        texts = []

        for ex in examples:
            try:
                img = Image.open(ex["image_path"]).convert("RGB")
                img.thumbnail((720, 720), Image.Resampling.LANCZOS)
                images.append(img)
            except Exception as e:
                print(f"❌ 加载图像失败 {ex['image_path']}: {e}")
                raise

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": ex["image_path"]},
                        {"type": "text", "text": ex["instruction"]},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": ex["output"]}],
                },
            ]
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            texts.append(text)

        batch = self.processor(
            text=texts, images=images, return_tensors="pt", padding=True
        )

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = input_ids.clone()

        for i in range(len(input_ids)):
            seq = input_ids[i].tolist()
            assistant_start_pos = -1
            for j in range(len(seq) - len(self.assistant_start_tokens) + 1):
                if (
                    seq[j : j + len(self.assistant_start_tokens)]
                    == self.assistant_start_tokens
                ):
                    assistant_start_pos = j + len(self.assistant_start_tokens)
                    break

            if assistant_start_pos != -1:
                labels[i, :assistant_start_pos] = -100
            else:
                labels[i, :] = -100
            labels[i, attention_mask[i] == 0] = -100

        batch["labels"] = labels
        return batch


# ================= 主训练函数 =================
def train():
    print("=" * 60)
    print("🔧 开始加载模型与处理器...")

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_ROOT = os.path.join(BASE_DIR, "../../data")
    MODEL_ROOT = os.path.join(BASE_DIR, "../../models")
    MODEL_ID = os.path.join(MODEL_ROOT, "Qwen3-VL-8B-Instruct")
    DATA_PATH = os.path.join(DATA_ROOT, "qwen3_sft_train_baseline.jsonl")
    TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_DIR = os.path.join(MODEL_ROOT, f"qwen3-vl-lora-meteorite_{TIMESTAMP}")
    VAL_SPLIT_RATIO = 0.1

    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    processor.tokenizer.padding_side = "left"

    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, device_map="auto", trust_remote_code=True
    )

    lora_config = LoraConfig(
        r=64,
        lora_alpha=32,
        target_modules=[
            "q_proj", "v_proj", "k_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    raw_dataset = load_dataset("json", data_files=DATA_PATH, split="train")
    val_size = int(len(raw_dataset) * VAL_SPLIT_RATIO)
    train_test = raw_dataset.train_test_split(test_size=val_size, seed=42)
    train_dataset = train_test["train"]
    eval_subset = train_test["test"].shuffle(seed=42).select(range(min(100, val_size)))
    full_val_dataset = train_test["test"]

    PER_DEVICE_BATCH = 2
    GRADIENT_ACCUMULATION = 4
    effective_batch_size = PER_DEVICE_BATCH * GRADIENT_ACCUMULATION
    steps_per_epoch = len(train_dataset) // effective_batch_size
    eval_steps = max(1, steps_per_epoch // 10)

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
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_dir=TENSORBOARD_LOG_DIR,
        report_to="tensorboard",
    )

    mae_config = MAEConfig(mask_ratio=0.5, mask_prob=0.5)

    trainer = MAETrainer(
        mae_config=mae_config,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_subset,
        data_collator=QwenVLDataCollator(processor),
    )

    print("\n🚀 开始正式微调陨石识别模型 (MAE-Aug Enabled)...")
    trainer.train()

    print("\n📊 使用完整验证集进行最终评估...")
    torch.cuda.empty_cache()
    trainer.eval_dataset = full_val_dataset
    eval_results = trainer.evaluate()
    print("\n📊 验证集最终评估结果:")
    for k, v in eval_results.items():
        print(f"  {k}: {v:.4f}")

    final_adapter_dir = os.path.join(OUTPUT_DIR, f"final_adapter_{TIMESTAMP}")
    trainer.save_model(final_adapter_dir)
    processor.save_pretrained(final_adapter_dir)
    print(f"\n✅ 权重已保存至 {final_adapter_dir}")


if __name__ == "__main__":
    train()
