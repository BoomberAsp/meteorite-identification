# 模型集成模块

模型无关的异构集成管线。输入多个单模型预测 CSV，严格按
`sample_submission.csv` 的 `id` 顺序对齐，输出可提交 CSV、debug 明细和 JSON 报告。

## 文件说明

| 文件 | 用途 |
|------|------|
| `ensemble.py` | 严格集成引擎：校验 CSV → ID 对齐 → hard/soft/hierarchical/single 输出 |
| `swin_inference.py` | Swin-B 单模型推理，输出 `id,label,prob_m` |
| `qwen_inference.py` | Qwen3-VL LoRA 推理，默认使用 60:40 先验 prompt，并通过 first-token logits 提取 `prob_m` |
| `convnext_inference.py` | ConvNeXt 单模型推理，输出 `id,label,prob_m` |

## 输入格式

每个模型预测文件必须至少包含：

```csv
id,label,prob_m
000001.jpg,0,0.12
000002.jpg,1,0.83
```

要求：

- `id` 必须和 `sample_submission.csv` 完全一致，不能缺失、不能多出、不能重复。
- `label` 必须是 `0` 或 `1`。
- `prob_m` 是预测为陨石的概率，范围 `[0, 1]`。如果缺失，会退化为 `label` 的 float 值。

## 单模型推理

```bash
# Qwen3-VL
python qwen_inference.py \
  --base_model /path/to/Qwen3-VL-8B-Instruct \
  --adapter /path/to/final_adapter \
  --data_root /path/to/stage2 \
  --output_dir data/baselines \
  --threshold 0.5

# Qwen 默认 prompt 与 src/testing/test_model.py 保持一致：
# 请观察这张图片，它有60%可能为非陨石（或不是石头），有40%概率为陨石。请你判断它是否为陨石？回答"是"或"否"。
# 如需做 prompt 消融，可通过 --instruction 覆盖。

# Swin-B
python swin_inference.py \
  --model /path/to/best_model.pth \
  --data_root /path/to/stage2 \
  --output_dir data/baselines \
  --threshold 0.5

# ConvNeXt
python convnext_inference.py \
  --model /path/to/convnext.pth \
  --data_root /path/to/stage2 \
  --output_dir data/baselines \
  --threshold 0.5
```

## 集成

```bash
python ensemble.py \
  --pred_files data/baselines/qwen3vl_test_preds.csv \
               data/baselines/swinb_test_preds.csv \
               data/baselines/convnext_test_preds.csv \
  --names qwen3vl swinb convnext \
  --template_csv data/sample_submission.csv \
  --output_dir data/ensemble_results \
  --strategies hard soft hierarchical \
  --threshold 0.5 \
  --hard_tie_break soft \
  --primary_idx 0 \
  --hierarchical_low 0.3 \
  --hierarchical_high 0.7
```

## 输出文件

每次运行会带时间戳生成：

- `ensemble_hard_*.csv`、`ensemble_soft_*.csv`、`ensemble_hierarchical_*.csv`：提交文件。
- `ensemble_*_debug_*.csv`：每张图的最终结果、来源、各模型 label/prob。
- `ensemble_aligned_inputs_*.csv`：按模板顺序对齐后的所有模型输入。
- `ensemble_report_*.json`：模型分歧率、各策略正例数量、概率统计、输出路径。

## 策略说明

| 策略 | 逻辑 |
|------|------|
| `hard` | 多数投票；偶数模型平局时默认用 soft 平均打破，也可选 `highest_confidence` / `label_0` / `label_1` |
| `soft` | 对 `prob_m` 加权平均，再按 `--threshold` 阈值化 |
| `hierarchical` | 主模型概率小于 `low` 或大于 `high` 时直接采用主模型，否则对辅助模型做 soft voting |
| `single` | 可选控制组：手动指定 `--strategies single` 时才为单模型生成提交 |

## 安全原则

集成脚本不会填默认值、不会只取共同 ID、不会静默丢弃样本。任何 ID 不一致都会直接报错，避免生成表面格式正确但实际错位或缺失的提交文件。
