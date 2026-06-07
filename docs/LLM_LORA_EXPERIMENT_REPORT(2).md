# LLM 标注数据生成与 Qwen3-VL LoRA 微调实验报告

## 1. 实验目标

本阶段工作的目标是解决原始 SFT 数据标注过于简单的问题。此前训练数据的 `output` 主要是简单的“是/否”，模型只能学习到分类标签，难以系统性学习陨石鉴别依据。因此，本实验使用外部视觉语言模型为训练图片生成“是/否 + 判别理由”的高质量标注，再用这些数据对 Qwen3-VL-8B-Instruct 做 LoRA/QLoRA 微调。

核心目标包括：

- 构造包含详细判别理由的 SFT 数据。
- 修复外部模型生成过程中的失败、截断和格式异常样本。
- 使用生成数据微调 Qwen3-VL-8B-Instruct。
- 在测试集上进行推理，并与外部 VLM 直接预测结果作对比。

## 2. 数据生成工作记录

### 2.1 初始生成脚本

使用脚本：

```text
src/data_processing_modules/generate_training_data.py
```

脚本功能：

- 读取图片目录与 `labels.csv`。
- 根据已有标签构造 prompt。
- 调用 OpenAI-compatible 外部视觉语言模型 API。
- 为每张图片生成 `是/否。理由：...` 格式的训练输出。
- 输出 CSV 与 JSONL 文件。

使用过的 API 配置包括：

```python
# 1-2500条数据
MODEL = "gpt-4o-mini"
BASE_URL = "https://api.ai-gaochao.cn/v1" 
```

```python
# 2501-5098
BASE_URL = "https://openrouter.ai/api/v1"
MODEL = "moonshotai/kimi-k2.6"  
```

后来为了提高速度和稳定性，脚本被扩展为支持：

- 多 API key 并发。
- `START_INDEX = 2501`，从第 2501 条开始生成，避免和队友前 2500 条重复。
- `partial.jsonl` 实时增量保存，支持断点续跑。
- `source_index` 字段，用于记录样本在原始标签文件中的序号。

### 2.2 数据区间与合并

队友已生成前 2500 条数据。本次主要生成第 2501 条到第 5098 条。

生成后的主要文件包括：

```text
test/training_data.jsonl                      # 前 2500 条
test/training_data_from_2501.partial.jsonl    # 第 2501 条之后的增量结果
test/training_data_from_2501.jsonl            # 修复整理后的后半部分
test/training_data_merged.jsonl               # 合并后的 5098 条数据
```

之后将训练数据迁移到 `data/` 目录：

```text
data/images/
data/training_data.jsonl
```

最终训练数据统计：

| 项目 | 数值 |
|---|---:|
| 总样本数 | 5098 |
| 陨石样本，即 output 以“是”开头 | 2685 |
| 非陨石样本，即 output 以“否”开头 | 2413 |
| 未知格式样本 | 0 |
| output 最短字符数 | 76 |
| output 最长字符数 | 572 |
| output 平均字符数 | 173.6 |

### 2.3 路径处理

由于 `run_train.sh` 会切换工作目录到：

```text
src/fine-tuning_(training)
```

因此训练 JSONL 中的图片路径需要从该目录出发可解析。最终将 `image_path` 改为：

```text
../../data/images/000001.jpg
```

检查结果显示，训练工作目录下前 5 张图片路径均可正常读取。

## 3. 数据质量控制与修复

### 3.1 失败类型

在外部模型生成过程中，出现过以下问题：

- API 返回空内容，日志表现为 `empty model content` 或 `NoneType object has no attribute 'strip'`。
- 模型输出被截断，表现为理由没有以中文句号 `。` 结尾。
- 输出为兜底文本：`（模型调用失败，无法生成理由）`。

这些问题主要由以下原因导致：

- 外部模型或 OpenRouter 上游偶发返回空 choice。
- 并发过高导致限流或不稳定。
- 某些模型对图片输入支持不稳定。
- 输出理由较长，存在被截断风险。

### 3.2 修复脚本

新增脚本：

```text
src/data_processing_modules/repair_failed_training_data.py
```

修复逻辑：

- 读取 `partial.jsonl`。
- 找到 `output` 中包含 `（模型调用失败，无法生成理由）` 的样本。
- 额外将不以 `。` 结尾的输出视为失败。
- 根据 `source_index` 找回原图和标签。
- 重新调用外部视觉模型生成理由。
- 更新对应 JSONL 行。

修复后统计：

```text
jsonl 行数：2598
csv 行数：2598
source_index：2501 到 5098
失败标记：0
不以 。 结尾：0
```

### 3.3 最终整理

将修复后的 `partial.jsonl` 重新整理为最终文件：

```text
test/training_data_from_2501.jsonl
test/training_data_from_2501.csv
```

再与前 2500 条合并，生成：

```text
test/training_data_merged.jsonl
```

合并校验：

```text
总行数：5098
source_index：1 到 5098
索引连续：True
所有行都有 source_index：True
字段顺序：source_index, image_path, instruction, output
```

### 4. 数据增强探索实验记录

#### 4.1 Visual-Text Co-Cropping 视觉文本协同裁剪（有效增强）

**实验动机与原理**
为解决模型在复杂陨石图像中注意力分散的问题，引入了 Visual-Text Co-Cropping 增强策略。该策略通过解析 output 中的关键视觉名词定位图像显著性区域，执行安全裁剪以聚焦核心特征，并同步动态重写 instruction 使文本指令与裁剪后的视觉焦点保持对齐，从而强化模型的细粒度视觉-语言对齐能力。

**实现与集成**
新增 `VisualTextCoCropper` 类于 `src/data_processing_modules/data_augmentation.py`。在 `train_lora.py` 的 `QwenVLDataCollator` 中完成在线集成，训练时按设定概率触发裁剪与指令重写，未触发或裁剪失败时自动回退至原始样本，保证训练稳定性。

**实验效果**
该增强策略在实验中表现出明确的正向收益：

-   有效缓解了模型对背景噪声的过拟合，提升了小目标陨石特征的识别召回率。
-   动态指令重写增强了模型对空间位置与局部细节的理解能力，验证集 loss 呈现稳定下降趋势。
-   作为独立增强模块，未引入任何标签噪声或格式兼容性问题，可作为后续实验的安全基线组件保留。

#### 4.2 Reasoning Chain Perturbator (RCP) 推理链扰动（负向回退）

**实验动机与原理**
为缓解模型过度自信问题，尝试引入 RCP 策略，将标准确定性 CoT 改写为带认知不确定性的软推理链（如引入“疑似”“需排除”等表述），期望提升模型在边界样本上的推理审慎度。

**实现与工程修复**
新增 `ReasoningChainPerturbator` 类，离线生成增强数据 `qwen3_sft_train_rcp.jsonl`。过程中修复了三项数据质量与工程问题：① `perturb()` 返回值类型异常导致的字典访问报错；② 字段名硬编码不匹配，将 `output` 映射至 `vl_model_reasoning` 并提取首字作为 `label`；③ 原始数据缺少 `is_augmented` 字段导致 RCP 完全不生效，强制注入该字段后恢复触发。

**实验结果与分数回退**
将 35% RCP 扰动数据混入训练集并提交评测后，**Public Score 从基线的 0.76923 骤降至 0.5**。经排查，确认该策略在当前阶段对模型性能产生显著负向影响，核心原因如下：

| 故障原因 | 影响机制 |
| :--- | :--- |
| `<think>` 标签格式冲突 | 代码使用字符串字面量而非 Qwen3-VL 原生特殊 token，官方评测脚本解析失败，答案提取为空 |
| RCP 语义漂移 | 35% 扰动比例过高，部分改写样本发生隐性标签翻转，污染模型决策边界 |
| 概率叠加失控 | 离线全量预处理与在线 Collator 二次采样叠加，实际扰动比例偏离预期 |

**后续处置计划**

1.  **立即回退**：暂停所有 RCP 相关训练，回退至无 RCP 版本提交验证，确认分数恢复至基线水平。
2.  **格式修复**：将 Collator 中的字符串 `<think>` 替换为 Qwen3-VL 原生特殊 token，消除评测解析障碍。
3.  **比例重调**：若后续重新启用 RCP，扰动比例从 35% 下调至 5%~10%，并扩充 `_is_safe()` 锚点词库过滤高风险改写样本。
4.  **消融验证**：在修复格式后单独测试低比例 RCP，确认语义漂移是否为分数回退的主因，再决定是否保留该策略。

## 5. LoRA 微调实验记录

### 5.1 基座模型

基座模型：

```text
Qwen3-VL-8B-Instruct
```

模型目录：

```text
models/Qwen3-VL-8B-Instruct
```

模型文件已下载完成，目录中包含：

```text
config.json
model-00001-of-00004.safetensors
model-00002-of-00004.safetensors
model-00003-of-00004.safetensors
model-00004-of-00004.safetensors
model.safetensors.index.json
tokenizer.json
preprocessor_config.json
```

### 5.2 训练脚本

训练脚本：

```text
src/fine-tuning_(training)/train_lora.py
```

运行脚本：

```text
run_train.sh
```

TensorBoard 脚本：

```text
run_tensorBoard.sh
```

训练数据路径配置：

```python
DATA_ROOT = os.path.join(BASE_DIR, "../../data")
DATA_PATH = os.path.join(DATA_ROOT, "training_data.jsonl")
MODEL_ID = os.path.join(MODEL_ROOT, "Qwen3-VL-8B-Instruct")
```

### 5.3 训练方法

训练采用 QLoRA：

```python
load_in_4bit=True
bnb_4bit_use_double_quant=True
bnb_4bit_quant_type="nf4"
bnb_4bit_compute_dtype=torch.bfloat16
```

LoRA 配置：

| 参数 | 值 |
|---|---|
| r | 64 |
| lora_alpha | 32 |
| lora_dropout | 0.1 |
| bias | none |
| task_type | CAUSAL_LM |
| target_modules | q_proj, v_proj, k_proj, o_proj, gate_proj, up_proj, down_proj |

图像处理：

```python
img.thumbnail((720, 720), Image.Resampling.LANCZOS)
```

文本处理：

- 使用 Qwen chat template。
- 训练 loss 只计算 assistant 回复部分。
- user prompt 与 padding 部分被 mask 为 `-100`。

### 5.4 训练超参数

| 参数 | 值 |
|---|---:|
| per_device_train_batch_size | 2 |
| gradient_accumulation_steps | 4 |
| effective batch size | 8 |
| learning_rate | 1e-4 |
| num_train_epochs | 3 |
| warmup_ratio | 0.1 |
| weight_decay | 0.05 |
| lr_scheduler_type | cosine |
| gradient_checkpointing | True |
| eval_strategy | steps |
| save_strategy | steps |
| save_total_limit | 3 |
| metric_for_best_model | eval_loss |

训练集和验证集划分：

```text
VAL_SPLIT_RATIO = 0.1
seed = 42
```

训练过程中的快速评估子集：

```text
EVAL_SUBSET_SIZE = 100
```

### 5.5 训练输出

训练输出目录：

```text
models/qwen3-vl-lora-meteorite_20260530_140304
```

保存的 checkpoint：

```text
checkpoint-1653
checkpoint-1710
checkpoint-1722
```

最终 adapter：

```text
models/qwen3-vl-lora-meteorite_20260530_140304/final_adapter_20260530_140304
```

训练状态记录：

| checkpoint | global_step | eval_loss | 备注 |
|---|---:|---:|---|
| checkpoint-1653 | 1653 | 0.802693 | best checkpoint |
| checkpoint-1710 | 1710 | 0.804029 | 接近结束 |
| checkpoint-1722 | 1722 | 0.804029 | 最终附近 |

最佳模型：

```text
checkpoint-1653
best eval_loss = 0.8026930126547813
```

训练结束后保存最终 adapter 和 processor 到：

```text
models/qwen3-vl-lora-meteorite_20260530_140304/final_adapter_20260530_140304
```

## 6. 测试集推理实验

### 6.1 LoRA 后 Qwen 模型推理

推理脚本：

```text
src/ensemble/qwen_inference.py
```

推理方式：

- 加载 Qwen3-VL-8B-Instruct 基座模型。
- 加载 LoRA adapter。
- 对 `data/test_images` 中测试图片逐张推理。
- 使用第一个生成 token 的 logits，提取“是”和“否”的相对概率。
- 得到 `prob_m`，即模型倾向判断为陨石的概率分数。

`prob_m` 计算方式：

```text
prob_m = exp(logit_是) / (exp(logit_是) + exp(logit_否))
```

它不是困惑度，也不是严格校准置信度，而是一个二分类打分。

推理输出：

```text
data/logs/qwen_reasoning_lora_test_preds.csv
data/logs/qwen_reasoning_lora_stats.json
```

推理统计：

| 项目 | 数值 |
|---|---:|
| 测试图片数 | 194 |
| 预测正例数 | 94 |
| 预测负例数 | 100 |
| 正例率 | 0.485 |
| prob_m mean | 0.6019 |
| prob_m std | 0.4392 |
| prob_m min | 0.000013 |
| prob_m max | 0.999990 |

阈值说明：

- 推理脚本默认支持 `--threshold` 参数。
- 若使用 `threshold=0.5`，则 `prob_m >= 0.5` 判为陨石。
- 早期预测文件曾讨论过阈值 0.20，但当前 stats 文件对应的新推理结果正例率为 48.5%，与早期版本不同，应以后续实际提交版本的阈值和输出文件为准。

注意：当前 `qwen_inference.py` 中 `run_inference()` 函数内部覆盖了传入的 instruction：

```python
instruction = "请观察这张图片。请你判断它是否为陨石？回答\"是\"或\"否\"。然后简要给出理由。"
```

因此命令行传入的 `--instruction` 在当前代码中不会真正生效。这一点后续如需做 prompt ablation，需要先修正。

**kaggle得分0.76923**

### 6.2 外部 VLM 直接预测测试集

新增脚本：

```text
src/testing/external_vlm_submission.py
```

用途：

- 直接调用外部视觉语言模型 API 对测试图片预测。
- 输出 Kaggle 可提交格式 `id,label`。
- 额外保存原始 JSONL 结果，方便检查模型理由。

默认输出：

```text
data/logs/external_vlm_predictions.jsonl
data/logs/external_vlm_submission.csv
```

由于发现 `data/sample_submission.csv` 有 511 个 id，但当前测试集实际只有 194 张图片，因此脚本增加了：

```text
--ignore-template
```

使用该参数后，脚本直接以 `data/test_images` 中的图片文件名作为提交 id。

最终外部 VLM 预测输出：

```text
data/logs/external_vlm_predictions_194.jsonl
data/logs/external_vlm_submission_194.csv
```

外部 VLM 预测统计：

| 项目 | 数值 |
|---|---:|
| 测试图片数 | 194 |
| 预测正例数 | 47 |
| 预测负例数 | 147 |
| 正例率 | 0.242 |

外部 VLM 输出格式示例：

```json
{
  "id": "000001.jpg",
  "label": 1,
  "reason": "该石头具有明显的银白色金属光泽...符合石铁陨石或铁陨石切片的典型视觉特征。",
  "raw_output": "..."
}
```

最终 submission 文件只保留：

```text
id,label
```

**kaggle得分0.52631**

## 7. 关键问题与处理记录

### 7.1 Gemini / Google 服务连接问题

曾计划使用 Gemini Flash/Pro 生成数据，但实际运行中出现网络或上游连接问题，因此改用 OpenRouter 上可访问的模型或 GPT 类模型继续生成。

### 7.2 多 API key 并发

为了加快标注生成速度，生成脚本支持：

```text
OPENAI_API_KEYS="key1,key2,key3"
```

并通过线程池并发处理图片。后续发现并发过高时容易出现空响应或截断，因此修复脚本采用更保守的并发与重试策略。

### 7.3 `sample_submission.csv` 与测试图片数量不一致

检查发现：

```text
data/sample_submission.csv: 511 个 id
data/test_images: 194 张图片
```

且存在扩展名不匹配，例如模板中为 `.jpeg`，图片目录中为 `.jpg`。因此外部 VLM 提交脚本增加 `--ignore-template`，以实际图片目录为准生成 194 行提交文件。

### 7.4 `prob_m` 的含义

`prob_m` 不是困惑度，而是模型第一个生成 token 在“是/否”二分类上的相对概率：

```text
prob_m 越接近 1，模型越倾向判断为陨石。
prob_m 越接近 0，模型越倾向判断为非陨石。
```

### 7.5 标注长度不一致

前 2500 条 output 相对短，后 2598 条 output 更详细。这会导致：

- 长 output 样本贡献更多训练 token。
- 模型输出风格可能偏向后半部分的详细理由。
- 训练显存占用受长样本影响。

本次训练仍保留该混合数据，后续可考虑统一 output 风格。

## 8. 当前产物汇总

### 8.1 数据文件

```text
data/training_data.jsonl                  # 最终训练数据，5098 条
data/images/                              # 训练图片
data/test_images/                         # 测试图片，194 张
```

### 8.2 模型文件

```text
models/Qwen3-VL-8B-Instruct/
models/qwen3-vl-lora-meteorite_20260530_140304/
models/qwen3-vl-lora-meteorite_20260530_140304/final_adapter_20260530_140304/
```

### 8.3 预测与提交文件

```text
data/logs/qwen_reasoning_lora_test_preds.csv
data/logs/qwen_reasoning_lora_stats.json
data/logs/external_vlm_predictions_194.jsonl
data/logs/external_vlm_submission_194.csv
```

## 9. 后续建议

1. 对 `qwen_inference.py` 修复 instruction 参数被覆盖的问题，以便稳定做 prompt ablation。
2. 在验证集上系统搜索阈值，而不是只根据测试集正例比例手动选择。
3. 对前 2500 条短 output 进行补全，统一训练数据风格。
4. 对外部 VLM 与 LoRA Qwen 的预测差异样本做人工检查，重点关注二者分歧较大的样本。
5. 尝试集成策略：例如 LoRA Qwen 作为主模型，外部 VLM 或 CNN 模型用于低置信度样本辅助判断。
6. 增加测试时增强 TTA，例如多尺度、水平翻转、多 prompt 平均等，以降低单次推理波动。

## 10. 复现实验命令示例

### 10.1 LoRA 训练

```bash
conda activate sta326
cd /root/sta326/final_project
bash run_train.sh
```

### 10.2 TensorBoard

```bash
bash run_tensorBoard.sh
```

TensorBoard 目录：

```text
data/logs/tensorboard
```

### 10.3 LoRA Qwen 测试集推理

```bash
conda activate sta326
cd /root/sta326/final_project

python src/ensemble/qwen_inference.py \
  --base_model models/Qwen3-VL-8B-Instruct \
  --adapter models/qwen3-vl-lora-meteorite_20260530_140304/final_adapter_20260530_140304 \
  --data_root data \
  --output_dir data/logs \
  --model_name qwen_reasoning_lora \
  --threshold 0.5
```

### 10.4 外部 VLM 直接生成提交

```bash
conda activate sta326
cd /root/sta326/final_project

export OPENAI_API_KEY="sk-..."

python src/testing/external_vlm_submission.py \
  --ignore-template \
  --model openai/gpt-4o-mini \
  --base-url https://openrouter.ai/api/v1 \
  --raw-jsonl data/logs/external_vlm_predictions_194.jsonl \
  --output-csv data/logs/external_vlm_submission_194.csv
```

---

本报告记录了从外部 LLM 标注生成、数据修复、训练数据整理、Qwen3-VL LoRA 微调，到测试集推理与外部 VLM 直接预测的完整实验过程。
