# 模型集成模块

模型无关的异构集成管线，支持硬投票、软投票、分层集成三种策略。

## 文件说明

| 文件 | 用途 |
|------|------|
| `ensemble.py` | 集成引擎，读多个模型 CSV → 三种策略 → Kaggle 提交文件 |
| `swin_inference.py` | Swin-B 单模型推理 |
| `qwen_inference.py` | Qwen3-VL LoRA 推理，通过 first-token logits 提取陨石概率 |
| `convnext_inference.py` | ConvNeXt 单模型推理（模板，待队友提供权重） |
| `val_eval_full.py` | 本地验证集评估 + 阈值/权重搜索 |

## 用法

### 第一步：单模型推理

```bash
# Swin-B
python swin_inference.py \
  --model /path/to/best_model.pth \
  --data_root /path/to/stage2 \
  --output_dir data/baselines \
  --threshold 0.5

# Qwen3-VL
python qwen_inference.py \
  --base_model /path/to/Qwen3-VL-8B-Instruct \
  --adapter /path/to/final_adapter \
  --data_root /path/to/stage2 \
  --output_dir data/baselines \
  --threshold 0.5

# ConvNeXt（同上模板）
python convnext_inference.py \
  --model /path/to/convnext.pth \
  --data_root /path/to/stage2 \
  --output_dir data/baselines \
  --threshold 0.5
```

每个脚本输出统一格式 CSV：`id,label,prob_m`

### 第二步：集成

```bash
# 异构集成（推荐：3 模型硬投票不再有平局问题）
python ensemble.py \
  --pred_files data/baselines/qwen3vl_test_preds.csv \
               data/baselines/swinb_test_preds.csv \
               data/baselines/convnext_test_preds.csv \
  --names qwen3vl swinb convnext \
  --output_dir data/ensemble_results \
  --template_csv sample_submission.csv \
  --primary_idx 0 \
  --hierarchical_low 0.3 --hierarchical_high 0.7 \
  --threshold 0.5
```

### 第三步：本地验证（Kaggle 提交前）

```bash
# 快速测试（200 样本）
python val_eval_full.py \
  --swin_model /path/to/swinb.pth \
  --qwen_base /path/to/Qwen3-VL-8B-Instruct \
  --qwen_adapter /path/to/adapter \
  --data_root /path/to/stage2 \
  --output_dir data/ensemble_baselines \
  --quick
```

## 三种集成策略

| 策略 | 原理 | 适用场景 |
|------|------|----------|
| **硬投票** | 每个模型投 0/1，多数胜出；平局取置信度高者 | 模型数量 ≥ 3 时效果最好 |
| **软投票** | 概率加权平均，再阈值化 | 模型概率校准良好时 |
| **分层集成** | 主模型高置信直接决策，不确定时交辅助模型投票 | 主模型明显强于辅助模型时 |

## 接入新模型

`ensemble.py` 是模型无关的，只读 CSV。接入新模型只需：

1. 按模板写推理脚本（参考 `swin_inference.py`）
2. 输出格式：`id,label,prob_m`
3. 作为额外 `--pred_files` 参数传入
