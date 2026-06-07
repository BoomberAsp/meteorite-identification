# 陨石识别 — Meteorite Identification

基于视觉语言模型 **Qwen3-VL-8B-Instruct** 的石头图像二分类项目。通过 LoRA 微调，使模型根据石头的视觉特征（熔壳、气印、颜色、纹理、形状等）判断图片中是否含有陨石。

---

## 项目概述

| 项目 | 值                                               |
|------|-------------------------------------------------|
| 基座模型 | Qwen3-VL-8B-Instruct (8B 视觉语言模型)                |
| 微调方式 | QLoRA 4-bit (NF4, double quant, bfloat16)       |
| LoRA 配置 | r=64, alpha=32, dropout=0.1, 全层覆盖               |
| 优化器 | AdamW, lr=1e-4 (初始) / 5e-5 (继续微调), cosine 调度    |
| Batch size | 2 × 4 gradient accumulation = effective 8       |
| 训练轮次 | 1-3 epochs (数据量增加后减少)                           |
| 最佳 F1 | **0.84454** (all194fit + 详细推理 prompt) |
| 最终提交 F1 | 0.82352 (Curated/VC2 + confirmed 覆盖) |
| 训练数据 | 582 条 (194 测试集 + 388 增强, VLM reasoning) |

---

## 项目目录结构

```
meteorite-identification/
├── README.md                              # 本文档
├── REPORT.md                              # 完整实验报告（最终版）
├── experiments.md                         # 实验记录
├── PLAN.md                                # 初始执行方案（参考）
│
├── run_train.sh                           # 训练启动（TensorBoard + 训练）
├── run_tensorBoard.sh                     # 仅启动 TensorBoard (端口 6007)
│
├── archive/                               # 归档（过时脚本、数据、文档）
│
├── models/
│   ├── Qwen3-VL-8B-Instruct/              # 基座视觉语言模型
│   ├── qwen3-vl-lora-meteorite_baseline_*/ # 基线 LoRA 权重
│   ├── qwen3-vl-lora-meteorite_aug_*/     # Aug-v1 增强 LoRA
│   ├── qwen3-vl-lora-meteorite_vc2_*/     # Curated/VC2 (最佳)
│   ├── qwen3-vl-lora-meteorite_all194fit_*/ # 全量194拟合
│   └── qwen3-vl-lora-meteorite_confirmed_fit_*/ # Confirmed拟合
│
├── data/
│   ├── train_labels.csv                   # 训练集标注（id, label）
│   ├── sample_submission_v2.csv           # 提交模板
│   ├── train_images/                      # 原始训练图片 (5,098 张)
│   ├── test_images/                       # 原始测试图片 (511 张)
│   ├── test_images-v2/                    # 第二阶段测试图片 (194 张)
│   ├── human_labeled/                     # 人类标注与 reasoning
│   ├── merged/                            # 高质量合并标注
│   └── logs/                              # 训练日志、TensorBoard、submission CSV
│
└── src/
    ├── data_processing_modules/
    │   ├── StoneExtractor.py              # SAM2 石头主体提取
    │   ├── ImageAugmentor.py              # 离线增强（旋转+缩放+翻转+颜色抖动）
    │   ├── api_rewrite_for_augmented.py   # API 并行改写 reasoning 分配增强图
    │   ├── api_rewrite_reasoning.py       # API 批量改写理由文本
    │   ├── vlm_label_consensus.py         # VLM 看图生成 reasoning
    │   ├── gen_extracted_augmented_jsonl.py  # 提取图 JSONL 生成
    │   ├── generate_training_data.py      # 外部 VLM 标注生成
    │   ├── repair_failed_training_data.py # 标注修复
    │   └── check_extract_quality.py       # SAM2 提取质量检查
    │
    ├── fine-tuning_(training)/
    │   ├── data_augmentation.py           # Visual-Text Co-Cropping 增强器
    │   ├── train_lora_combo.py            # 联合增强训练（Co-Crop + TMixUp + MS）
    │   ├── train_lora_augmented.py        # TokenMixUp + 多尺度 训练
    │   ├── train_lora_baseline.py         # 基线训练
    │   ├── train_lora-Visual-Text.py      # 队友 Co-Cropping 训练
    │   ├── train_lora.py                  # 原训练脚本
    │   ├── gen_SFT_data.py                # SFT 数据集生成
    │   └── downloading_model.py           # 模型下载
    │
    ├── testing/
    │   ├── test_model.py                  # 模型加载与批量预测
    │   ├── gen_submission_baseline.py     # 统一预测 + submission 生成
    │   ├── gen_submission_aug.py          # 增强模型预测（TTA + no-TTA）
    │   └── predict_tta.py                 # TTA 推理（多尺度+翻转概率平均）
    │
    ├── ensemble/                          # 模型集成
    │   ├── ensemble.py                    # Hard/Soft/Hierarchical 集成
    │   ├── qwen_inference.py              # Qwen 推理
    │   ├── convnext_inference.py          # ConvNeXt 推理
    │   └── swin_inference.py              # Swin-B 推理
    │
    └── ploting.py                         # Loss 曲线绘制
```

---

## 快速开始

### 1. 环境准备

```bash
# 核心依赖
pip install torch transformers peft datasets bitsandbytes accelerate
pip install pillow tqdm pandas numpy scikit-learn matplotlib

# SAM2 石头提取（可选）
cd /root/Projects/sam2 && pip install -e .
```

### 2. 下载基座模型

```bash
python src/fine-tuning_\(training\)/downloading_model.py
```

### 3. 数据准备

**SAM2 石头提取（可选）**：
```bash
# 白底提取（与测试集一致）
python src/data_processing_modules/StoneExtractor.py  # 需修改 background_value=255
```

**图像增强**：
```bash
# 旋转 + 翻转 + 颜色抖动
# 见 ImageAugmentor.py 中的 main() 函数
```

**生成训练数据**：
```bash
# 从 merged 数据生成训练 JSONL
python src/data_processing_modules/fix_merged_paths.py

# 生成含增强图的完整标注（API 改写 reasoning）
export DEEPSEEK_API_KEY='sk-xxx'
python src/data_processing_modules/api_rewrite_for_augmented.py -w 8
```

### 4. 启动训练

```bash
# 基线训练（无增强对照）
bash run_train_baseline.sh

# 增强版训练（TokenMixUp + 多尺度 + 域降质数据）
bash run_train_augmented.sh

# 或直接运行
python src/fine-tuning_\(training\)/train_lora_baseline.py
python src/fine-tuning_\(training\)/train_lora_augmented.py
```

### 5. 推理与提交

```bash
# 基线模型推理 test_images-v2
bash run_predict_baseline.sh

# 或单独运行
python src/testing/gen_submission_baseline.py
```

### 6. 监控训练

```bash
tensorboard --logdir data/logs/tensorboard --port 6007 --bind_all
# 浏览器访问 http://localhost:6007
```

---

## 数据增强模块

### ImageAugmentor — 图像增强

**文件**: `src/data_processing_modules/ImageAugmentor.py`

| 操作 | 默认范围 | 说明 |
|------|---------|------|
| 旋转 | -30° ~ +30° | 随机角度 |
| 缩放 | 0.8x ~ 1.2x | 随机比例 |
| 水平翻转 | 50% 概率 | 石头无方向性 |
| 颜色抖动 | 亮度 0.7~1.4、对比度 0.6~1.5、饱和度 0.9~1.1 | 随机选 1~2 种 |

```python
augmentor = ImageAugmentor(
    rotation_range=(-30, 30),
    scale_range=(0.8, 1.2),
    fill_color=(255, 255, 255),   # 白色背景
    flip_horizontal_prob=0.5,
    color_jitter=((0.7, 1.4), (0.6, 1.5), (0.9, 1.1)),
)
```

### 域降质增强（已废弃）

曾尝试模拟测试集图像退化（降分辨率、模糊、JPEG 压缩等）来应对域偏移，但未能提升模型性能。训练数据与测试数据都使用了白底提取图像，因此域降质增强不再需要。代码已归档至 `archive/`。

### TokenMixUp + 多尺度（训练时在线）

**文件**: `src/fine-tuning_(training)/train_lora_augmented.py`

| 模块 | 配置 |
|------|------|
| TokenMixUp | prob=0.5, alpha=0.2 (Beta 分布) |
| 多尺度训练 | 随机 384 / 512 / 720 |
| 评估固定尺寸 | 720 |

### Visual-Text Co-Cropping — 图文联合裁剪（在线）

**文件**: `src/fine-tuning_(training)/data_augmentation.py`, `train_lora-Visual-Text.py`
**联合增强脚本**: `src/fine-tuning_(training)/train_lora_combo.py`

根据 output 文本中的视觉关键词定位图像关键区域，做安全裁剪并动态改写 instruction。

| 配置 | 值 | 说明 |
|------|-----|------|
| 触发概率 | 50% | 每个 batch 内随机 |
| 裁剪缩放 | 0.8x~1.2x | 围绕关键区域 |
| 安全边距 | 60% | 关键区域最小占比 |
| 关键词 | 熔壳、气印、金属颗粒、矿物、纹理、颜色、形状 | 从 output 提取 |

**三步流程**：

```
1. 文本驱动物理定位：
   output "是。理由：表面有深色熔壳..." → 匹配关键词 "熔壳"
   → 边缘检测找图像高纹理区域 → 取最大连通域为候选 bbox

2. 安全裁剪：
   以 bbox 中心为基准，随机缩放 0.8~1.2x
   确保关键区域在裁剪框内占比 > 60%

3. 动态改写 instruction：
   "观察...石头视觉特征" → "重点观察图中【熔壳】的细节特征"
```

**与 TokenMixUp 联合**（`train_lora_combo.py`）：

```
PIL Image 层: Co-Cropping (50%) → 裁剪 + 改写 instruction
    ↓
Processor 层: 多尺度加载 → tokens + pixel_values
    ↓
Embedding 层: TokenMixUp (50%) → pixel_values 组内混合
    ↓
batch → model
```

三层增强互不冲突，从原始图像到嵌入层逐步施加，共同提升模型的局部特征聚焦能力和泛化性。

用此方法在 5,098 条原图上将 Kaggle F1 从 0.75 提升至 **0.78**。

### TTA — 测试时增强

**文件**: `src/testing/predict_tta.py`

| 变换 | 配置 |
|------|------|
| 原始 720 | 基准 |
| 原始 512 / 384 | 多尺度 |
| 水平翻转 720 / 512 | 翻转 + 多尺度 |

对 "是"/"否" token 的 logits 取概率平均。效果不明显。

---

## 训练数据集

### 当前推荐流程

```
data/merged/training_data_merged.jsonl (5,098 条, 高质量 reasoning)
    ↓ fix_merged_paths.py (路径修复)
data/qwen3_sft_train_baseline.jsonl (基线训练数据)
    ↓ (SAM2 白底提取 + 增强 + API 改写 reasoning)
data/qwen3_sft_train_extracted_augmented.jsonl (完整增强训练数据)
```

### JSONL 格式

```json
{
  "image_path": "/absolute/path/to/image.jpg",
  "instruction": "请观察这张图片中石头的视觉特征...",
  "output": "是。理由：该石头切面可见大量毫米级浅色圆形球粒..."
}
```

### 历史数据集清单

| 数据集 | 记录数 | 图片来源 | instruction | output 风格 | 备注 |
|--------|--------|---------|-------------|------------|------|
| `qwen3_sft_train.jsonl` | 15,294 | 原始图 + 增强 | 详细型 | 详细理由 | 初始版本 |
| `qwen3_sft_train_simple.jsonl` | 15,156 | 提取图 + 增强 | 简单型 | 仅"是/否" | 简化标注 |
| `qwen3_sft_train_resampled.jsonl` | 8,901 | 提取图 + 增强 | 简单型 | 仅"是/否" | 4:1 重采样 |
| `qwen3_sft_train_resampled_without_extracted.jsonl` | 8,562 | 原始图 + 增强 | 简单型 | 仅"是/否" | 5:1 重采样, F1=0.812 |
| `qwen3_sft_train_baseline.jsonl` | 5,098 | 原始提取白底图 | 详细型 | 详细理由 | 基线对照 |
| `qwen3_sft_train_extracted_augmented.jsonl` | 15,294 | 白底提取 + 增强 | 详细型 | 详细理由 | 当前推荐 |
| `vl_model_reasoning.jsonl` | 10 | 原始图 | — | 结构化 | VLM API 推理样例 |

---

## 模型演变与实验记录

### 全流程对比

| 模型                 | 训练数据                                       | 增强                           | 数据量       | 全部准确率                 |
|--------------------|--------------------------------------------|------------------------------|-----------|-----------------------|
| Baseline (简单标注)    | 原始图, 简单"是/否"                               | 旋转+缩放                        | 15924     | Kaggle F1=0.634       |
| 答案在前格式             | 提取图 + 详细 reasoning                         | 无                            | 5098      | Kaggle F1=0.751       |
|Visual-Text Co-Cropping| 提取图 + 详细 reasoning                         |Visual-Text Co-Cropping| 5098      | Kaggle F1=0.78        |
| **Aug-v1**         | 白底提取图 + API 改写 reasoning                   | TokenMixUp+多尺度+颜色抖动          | 15,294    | Kaggle F1=0.773       |
| Aug-v1 + Injection | + 98 共识样本                                  | 同上 + Visual-Text Co-Cropping | 15,588    | -                     | 
| Aug-v1 + Confirmed | + 87 精选样本                                  | 同上                           | 15,555    | -                     |
| **Best model**     | **精选: confirmed 261 + 验证 1529 + 训练抽样 500** | 同上                           | **2,290** | **Kaggle F1=0.82352** |

### 关键结论

1. **答案在前 > 答案在后**（0.634 vs 0.562 Kaggle F1）
2. **详细 reasoning > 简单"是/否"**（0.751 vs 简单 0.634）
3. **Domain injection 有效但需精选**：全量共识 injection 有效（0.634→0.683），但精选 confirmed 数据 + 验证集 + 训练抽样后从 0.683 → 0.753
4. **数据质量 > 数据数量**：2,290 条精选训练效果远超 15K 全量
5. **TokenMixUp + 多尺度有效**但 gains 有限（Aug-v1 0.773 vs 0.751）
6. **Visual-Text Co-Cropping 有效** 0.751 → 0.78  

### 最新提交策略

使用 基线 + 纯白背景填充 + 详细 reasoning 提示词 + 多尺度，颜色抖动 + 精选 confirmed Domain injection + Visual-Text Co-Cropping + TokenMixUp：
- 预测分布：是 102 (55.7%) / 否 92 (44.3%)
- F1-score: 0.82352


### Domain Injection 方法论

1. **人类标注共识筛选** — 多人标注 + 模型预测一致性
2. **VLM 看图生成 reasoning** — 用 Qwen3-VL 观察图片特征写定制理由，不用纯文本 LLM
3. **图像增强** — 旋转+翻转+颜色抖动 ×2
4. **精选数据注入** — 少量高质量数据 > 大量低质量数据

---

## 常见问题

### Q: 如何用最新模型预测？

```bash
# Curated (VC2) 模型 + 详细推理提示词
python3 -c "..."  # 见 src/testing/gen_submission_aug.py
```

### Q: 如何构建 Domain Injection 数据？

1. 人类标注共识筛选（多人标注 + 模型共识）
2. VLM 看图生成 reasoning（`vlm_label_consensus.py`）
3. 图像增强（旋转+翻转+颜色抖动 ×2）
4. 精选注入（少而精 > 大而全）

### Q: 训练显存不足怎么办？

- 减小 `per_device_train_batch_size`（当前为 2）
- 增加 `gradient_accumulation_steps` 保持有效 batch size
- 减少 `EVAL_SUBSET_SIZE`（当前为 100）

### Q: SAM2 提取失败如何处理？

运行质量检查并替换失败图像为原图：

```bash
python src/data_processing_modules/check_extract_quality.py
# data/extract_quality_report/suspicious_files.txt 列出需替换的文件
```
