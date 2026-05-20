import os
import sys
import json
import torch
from datetime import datetime
from PIL import Image
from datasets import load_dataset, Dataset
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
import numpy as np

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)


# ================= 自定义回调：记录并打印每个 epoch 的 loss =================
class EpochLossCallback(TrainerCallback):
    """
    自定义回调，记录每个 epoch 的平均 loss，并在 epoch 结束时打印汇总。
    """
    def __init__(self):
        self.epoch_losses = []
        self.step_losses = []

    def on_epoch_begin(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        pass

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.step_losses.append(logs["loss"])

    def on_epoch_end(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if logs and "loss" in logs:
            epoch_loss = logs.get("loss")
            self.epoch_losses.append(epoch_loss)

            print(f"\n{'='*50}")
            print(f"✅ Epoch {int(state.epoch)} 完成")
            print(f"   Epoch 平均 loss: {epoch_loss:.4f}")

            if len(self.epoch_losses) >= 2:
                prev_loss = self.epoch_losses[-2]
                change = epoch_loss - prev_loss
                if change < 0:
                    print(f"    相比上一 epoch 下降：{abs(change):.4f}")
                else:
                    print(f"    相比上一 epoch 上升：{change:.4f}")

            if len(self.epoch_losses) % 5 == 0 or len(self.epoch_losses) == args.num_train_epochs:
                print(f"\n Epoch Loss 历史:")
                for i, loss in enumerate(self.epoch_losses, 1):
                    print(f"   Epoch {i}: {loss:.4f}")
        print(f"{'='*50}")


# ================= 自定义 Trainer：添加更详细的日志 =================
class DebugTrainer(Trainer):
    """
    扩展 Trainer 以添加更详细的调试信息和显存管理
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epoch_callback = EpochLossCallback()
        self.add_callback(self.epoch_callback)

    def get_train_dataloader(self):
        dataloader = super().get_train_dataloader()
        print(f"\n📦 训练数据加载器:")
        print(f"   总 batch 数：{len(dataloader)}")
        print(f"   每 batch 样本数：{self.args.per_device_train_batch_size}")
        return dataloader
    
    def evaluation_loop(self, dataloader, description, prediction_loss_only, ignore_keys=None, metric_key_prefix="eval"):
        """
        重写 evaluation_loop，手动控制每个 batch 并释放显存，
        返回 EvalLoopOutput 对象以避免 AttributeError。
        """
        import torch
        import gc

        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for step, batch in enumerate(dataloader):
            batch = {k: v.to(self.args.device) for k, v in batch.items()}
            with torch.inference_mode():
                outputs = self.model(**batch)
                loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
                total_loss += loss.item()
                num_batches += 1

            # 删除 batch 和 outputs 引用，释放显存
            del batch, outputs
            gc.collect()
            torch.cuda.empty_cache()

            # 可选：打印显存使用情况（调试用）
            # if (step + 1) % 10 == 0:
            #     allocated = torch.cuda.memory_allocated() / 1e9
            #     print(f"Step {step+1}, GPU memory allocated: {allocated:.2f} GB")

        avg_loss = total_loss / num_batches

        # 构建 EvalLoopOutput 对象
        metrics = {f"{metric_key_prefix}_loss": avg_loss}
        return EvalLoopOutput(predictions=None, label_ids=None, metrics=metrics, num_samples=len(dataloader.dataset))


# ================= 1. 配置参数 =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(BASE_DIR, "../../data")
MODEL_ROOT = os.path.join(BASE_DIR, "../../models")
MODEL_ID = os.path.join(MODEL_ROOT, "Qwen3-VL-8B-Instruct")
# DATA_PATH = os.path.join(DATA_ROOT, "qwen3_sft_train_resampled.jsonl")  # 简化output版本
DATA_PATH = os.path.join(DATA_ROOT, "qwen3_sft_train_resampled_without_extracted.jsonl")  # 简化output版本
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = os.path.join(MODEL_ROOT, f"qwen3-vl-lora-meteorite_{TIMESTAMP}")

# 验证集比例：0.1 表示 90% 训练，10% 验证
VAL_SPLIT_RATIO = 0.1


def compute_metrics(eval_pred):
    """
    简单的验证集评估指标（这里只返回 loss）
    对于 VLM 任务，准确的指标需要在推理阶段计算
    """
    predictions, labels = eval_pred
    return {"eval_loss": float(np.mean(predictions))}


def train():
    print("="*60)
    print("🔧 开始加载模型与处理器...")

    # ================= 2. 加载模型与处理器 =================
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    processor.tokenizer.padding_side = 'left'
    print(f"✅ 处理器加载完成，类型：{type(processor)}")
    print(f"   Tokenizer 词汇表大小：{len(processor.tokenizer)}")

    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    print(f"✅ 模型加载完成，类型：{type(model)}")
    print(f"   模型主设备：{next(model.parameters()).device}")

    # ================= 3. 配置 LoRA =================
    print("\n🔧 配置 LoRA...")
    lora_config = LoraConfig(
        r=64,  # 简化output，降低rank防止过拟合
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, lora_config)
    print("✅ LoRA 配置完成")
    model.print_trainable_parameters()

    # ================= 4. 加载原始数据集并划分训练/验证集 =================
    print("\n📂 加载原始数据集...")
    raw_dataset = load_dataset("json", data_files=DATA_PATH, split="train")
    print(f"✅ 数据集加载完成，共 {len(raw_dataset)} 条记录")

    # 按 90/10 划分训练集和验证集
    val_size = int(len(raw_dataset) * VAL_SPLIT_RATIO)
    train_test = raw_dataset.train_test_split(test_size=val_size, seed=42)
    train_dataset = train_test["train"]
    val_dataset = train_test["test"]

    print(f"\n 数据集划分完成:")
    print(f"   训练集：{len(train_dataset)} 条 ({len(train_dataset)/len(raw_dataset)*100:.1f}%)")
    print(f"   验证集：{len(val_dataset)} 条 ({len(val_dataset)/len(raw_dataset)*100:.1f}%)")
    
    # 训练时评估用的子集（减少显存占用，避免OOM）
    EVAL_SUBSET_SIZE = 100  # 训练过程中评估的样本数
    eval_subset = val_dataset.shuffle(seed=42).select(range(min(EVAL_SUBSET_SIZE, len(val_dataset))))
    print(f"   评估子集：{len(eval_subset)} 条 (用于训练过程中快速评估，避免OOM)")
    
    # 保留完整验证集用于最终评估
    full_val_dataset = val_dataset

    # 打印样本
    print("\n 训练数据样例（前 2 条）:")
    for i in range(min(2, len(train_dataset))):
        print(f"  样本 {i+1}: image_path={train_dataset[i]['image_path'][:50]}...")

    # ================= 5. 定义动态 Collator =================
    class QwenVLDataCollator:
        def __init__(self, processor):
            self.processor = processor
            # Qwen3-VL 的 assistant 起始 tokens: <|im_start|>assistant\n
            # 编码后是 [151644, 77091, 198] 三个 tokens
            self.assistant_start_tokens = processor.tokenizer.encode(
                "<|im_start|>assistant\n", add_special_tokens=False
            )
            # <|im_end|> token id
            self.im_end_id = processor.tokenizer.encode("<|im_end|>", add_special_tokens=False)[0]

        def __call__(self, examples):
            # 在 worker 进程中需要重新导入 torch
            import torch

            images = []
            for ex in examples:
                try:
                    img = Image.open(ex["image_path"]).convert("RGB")
                    img.thumbnail((720, 720), Image.Resampling.LANCZOS)
                    images.append(img)
                except Exception as e:
                    print(f"❌ 加载图像失败 {ex['image_path']}: {e}")
                    raise

            texts = []
            for ex in examples:
                messages = [
                    {"role": "user", "content": [
                        {"type": "image", "image": ex["image_path"]},
                        {"type": "text", "text": ex["instruction"]}
                    ]},
                    {"role": "assistant", "content": [{"type": "text", "text": ex["output"]}]}
                ]
                text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                texts.append(text)

            batch = self.processor(text=texts, images=images, return_tensors="pt", padding=True)

            required_keys = ["input_ids", "attention_mask", "pixel_values", "image_grid_thw"]
            missing = [k for k in required_keys if k not in batch]
            if missing:
                raise KeyError(f"Missing keys: {missing}")

            # 正确构建 labels：只计算 assistant 回复部分的 loss
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = []

            for i in range(len(input_ids)):
                seq = input_ids[i].tolist()
                labels_i = input_ids[i].clone()

                # 查找 assistant 起始位置: <|im_start|>assistant\n
                # 即查找 self.assistant_start_tokens 的连续序列
                assistant_start_pos = -1
                for j in range(len(seq) - len(self.assistant_start_tokens) + 1):
                    if seq[j:j + len(self.assistant_start_tokens)] == self.assistant_start_tokens:
                        assistant_start_pos = j + len(self.assistant_start_tokens)
                        break

                if assistant_start_pos != -1:
                    # Mask掉 user 部分（包括 assistant 起始标记本身）
                    # 只保留 assistant 实际回复内容参与 loss 计算
                    labels_i[:assistant_start_pos] = -100
                    # Mask掉 padding tokens
                    labels_i[attention_mask[i] == 0] = -100
                else:
                    # 未找到 assistant 起始位置，安全处理：全部 mask
                    print(f"⚠️ 未找到 assistant 起始位置，样本 {i} 将被跳过")
                    labels_i[:] = -100

                labels.append(labels_i)

            batch["labels"] = torch.stack(labels)
            return batch

    # ================= 6. 训练参数 =================
    print("\n⚙️ 配置训练参数...")

    # 计算每 0.1 个 epoch 的步数
    PER_DEVICE_BATCH = 2
    GRADIENT_ACCUMULATION = 4
    effective_batch_size = PER_DEVICE_BATCH * GRADIENT_ACCUMULATION
    steps_per_epoch = len(train_dataset) // effective_batch_size
    eval_steps = max(1, steps_per_epoch // 10)  # 每 0.1 个 epoch 评估一次

    print(f"   总样本数：{len(train_dataset)}")
    print(f"   有效 batch size: {effective_batch_size}")
    print(f"   每 epoch 步数：{steps_per_epoch}")
    print(f"   评估间隔：每 {eval_steps} 步 (约 0.1 个 epoch)")

    # TensorBoard 日志目录（固定目录，方便 TensorBoard 监控）
    TENSORBOARD_LOG_DIR = os.path.join(DATA_ROOT, "logs", "tensorboard")
    os.makedirs(TENSORBOARD_LOG_DIR, exist_ok=True)
    print(f"   TensorBoard 日志目录: {TENSORBOARD_LOG_DIR}")

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=PER_DEVICE_BATCH,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=1e-4,
        num_train_epochs=3,  # 增加epochs充分学习
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
        fp16_full_eval=False,                  # 评估时保持 bf16 精度
        eval_strategy="steps",              # 按步数评估
        eval_steps=eval_steps,              # 每 0.1 个 epoch 评估一次
        eval_accumulation_steps=1,
        save_strategy="steps",              # 按步数保存（与 eval_strategy 一致)
        save_steps=eval_steps,              # 保存间隔与评估间隔一致
        load_best_model_at_end=True,        # 训练结束时加载最佳模型
        metric_for_best_model="eval_loss",  # 以验证集 loss 为标准
        greater_is_better=False,            # loss 越小越好
        # TensorBoard 配置
        logging_dir=TENSORBOARD_LOG_DIR,    # TensorBoard 日志目录
        report_to="tensorboard",            # 启用 TensorBoard 报告
    )
    print("✅ 训练参数配置完成")
    print(f"   有效 batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")

    # ================= 7. 创建 Debug Trainer =================
    print("\n🔧 创建 DebugTrainer...")
    trainer = DebugTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_subset,         # 训练时用小规模评估集避免OOM
        data_collator=QwenVLDataCollator(processor),
        compute_metrics=compute_metrics,
    )
    print("✅ DebugTrainer 创建完成，已注册 EpochLossCallback")

    print("\n🚀 开始正式微调陨石识别模型...")
    print(f"   训练数据量：{len(train_dataset)}")
    print(f"   验证数据量：{len(val_dataset)}")
    print(f"   训练轮次：{training_args.num_train_epochs}")
    print(f"   学习率：{training_args.learning_rate}")
    print(f"   输出目录：{OUTPUT_DIR}")
    print()

    trainer.train()

    # ================= 8. 打印训练汇总 =================
    print("\n" + "="*60)
    print("🎉 训练完成！")
    print("="*60)

    # 用完整验证集进行最终评估
    print("\n📊 使用完整验证集进行最终评估...")
    import torch
    torch.cuda.empty_cache()  # 清理显存后再评估
    trainer.eval_dataset = full_val_dataset
    eval_results = trainer.evaluate()
    print(f"\n📊 验证集最终评估结果:")
    for k, v in eval_results.items():
        print(f"   {k}: {v:.4f}")

    if hasattr(trainer, 'epoch_callback'):
        epoch_losses = trainer.epoch_callback.epoch_losses
        if epoch_losses:
            print(f"\n📈 Epoch Loss 历史:")
            for i, loss in enumerate(epoch_losses, 1):
                print(f"   Epoch {i}: {loss:.4f}")

            final_avg = np.mean(epoch_losses[-3:]) if len(epoch_losses) >= 3 else np.mean(epoch_losses)
            initial_avg = np.mean(epoch_losses[:3]) if len(epoch_losses) >= 3 else epoch_losses[0]
            print(f"\n📈 Loss 变化：{initial_avg:.4f} → {final_avg:.4f} ({(final_avg-initial_avg)/initial_avg*100:+.1f}%)")

    # 保存最终结果
    final_adapter_dir = os.path.join(OUTPUT_DIR, f"final_adapter_{TIMESTAMP}")
    trainer.save_model(final_adapter_dir)
    processor.save_pretrained(final_adapter_dir)
    print(f"\n✅ 权重已保存至 {final_adapter_dir}")


if __name__ == "__main__":
    train()
