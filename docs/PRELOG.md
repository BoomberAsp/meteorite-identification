# 模型集成部分工作记录

> 用途：给队友同步模型集成部分目前做了什么、产出了哪些文件、实验结果如何，以及明天 pre / PPT 可以怎么组织。  
> 这不是最终报告，重点是交代工作过程和实验日志，方便团队内部复盘与做展示材料。

## 0. 给队友的快速摘要

这一部分可以在 PPT 中概括为：

- 完成了模型集成模块重构，把 Qwen / Swin-B / ConvNeXt 的输出统一成 `id,label,prob_m`。
- 新版 ensemble 严格按 `sample_submission.csv` 对齐，缺失、重复、多余 id 都会直接报错，避免生成错误提交。
- 在服务器上跑完三路模型预测，并生成 hard voting、soft voting、hierarchical ensemble 三类提交。
- 实验结果显示 naive ensemble 没有超过 Qwen 单模：Qwen 单模 0.769，hierarchical 0.756，hard 0.740。
- 主要分析结论：CNN 分支更保守，平权投票会把很多 Qwen 正例改成负例，导致集成降分。
- 后续方向：不做简单平均，改做 Qwen-dominant ensemble，只在 Qwen 不确定或辅助模型强一致时小范围修正。

## 1. 负责范围

本模块负责 `teamcode/src/ensemble` 下的模型集成与推理对齐，核心目标是把不同模型的预测结果统一成同一种 CSV 格式，再生成最终可提交的 Kaggle submission。

目前涉及的模型有三类：

| 模型 | 文件 | 作用 |
|---|---|---|
| Qwen3-VL LoRA | `qwen_inference.py` | 主力视觉语言模型，输出 `id,label,prob_m` |
| Swin-B | `swin_inference.py` | CNN/Transformer 图像分类支路，输出 `id,label,prob_m` |
| ConvNeXt-Large | `convnext_inference.py` | CNN 图像分类支路，输出 `id,label,prob_m` |
| Ensemble | `ensemble.py` | 读取多个模型 CSV，执行 hard / soft / hierarchical 集成 |

本轮主要完成了三件事：

1. 重写 `ensemble.py`，把原来不够稳的集成脚本改成严格校验版本。
2. 对齐 Qwen 推理配置，保证进入 ensemble 的 Qwen 预测 CSV 和队友已有推理方式一致。
3. 在服务器上跑完 Qwen / Swin-B / ConvNeXt 三路预测和多种集成实验，并记录结果。

## 2. 为什么重写 ensemble

原来的 ensemble 脚本是用 DeepSeek 快速生成的，能跑基本流程，但有几个严重风险：

1. **只计算 common ids，但后续没有真正按 common ids 过滤**
   如果某个模型缺少部分测试图片，脚本可能在后续步骤报错，或者生成和 submission 模板不一致的结果。

2. **submission 缺失检查顺序不安全**
   原逻辑里可能先做 `astype(int)`，如果有缺失预测，会在 pandas 类型转换阶段报错，而不是清楚告诉我们缺哪些图片。

3. **模型输出顺序和 sample_submission 顺序没有被作为第一优先级**
   Kaggle 提交文件最重要的是必须严格按照 `sample_submission.csv` 的 id 顺序生成。

4. **hard / soft / hierarchical 的 debug 信息不够**
   出现集成变差时，很难回答“到底是哪几张图被哪个模型改掉了”。

所以现在的 ensemble 改成了严格版本：

- 所有预测 CSV 必须包含 `id,label`，最好包含 `prob_m`。
- `id` 必须和 `sample_submission.csv` 完全一致，不能缺失、不能多出、不能重复。
- 所有模型预测先按 sample_submission 的 id 顺序对齐。
- 每种集成策略都输出：
  - submission CSV
  - debug CSV
  - JSON report
- 默认只生成集成结果，不再单独生成单模型 submission。

## 3. Qwen 推理配置对齐

为了让集成输入和队友已有推理方法保持一致，`qwen_inference.py` 的默认 instruction 已对齐到 `src/testing/test_model.py` 中使用的版本：

```text
请观察这张图片，它有60%可能为非陨石（或不是石头），有40%概率为陨石。请你判断它是否为陨石？回答"是"或"否"。
```

这部分不是模型集成的主要创新点，更多是为了保证 Qwen 单模型 CSV 的生成方式和已有实验口径一致，避免不同脚本之间的推理配置不统一。

修改后：

- `qwen_inference.py` 默认使用与队友测试脚本一致的 instruction。
- 支持通过 `--instruction` 覆盖 prompt，方便后续做配置对比。
- stats JSON 中记录实际使用的 instruction，避免以后分不清某个 CSV 是用哪个 prompt 生成的。

## 4. 当前 ensemble pipeline

现在完整流程是：

```bash
# 1. Qwen 推理
python src/ensemble/qwen_inference.py \
  --base_model models/Qwen3-VL-8B-Instruct \
  --adapter models/final_adapter_20260530_140304 \
  --data_root data \
  --output_dir data/ensemble_baselines \
  --model_name qwen3vl_6030 \
  --threshold 0.5

# 2. Swin-B 推理
python src/ensemble/swin_inference.py \
  --model models/swinb/best_model_20260506_040216.pth \
  --data_root data \
  --output_dir data/ensemble_baselines \
  --model_name swinb \
  --threshold 0.5

# 3. ConvNeXt 推理
python src/ensemble/convnext_inference.py \
  --model models/convnext/best_model.pth \
  --data_root data \
  --output_dir data/ensemble_baselines \
  --model_name convnext \
  --threshold 0.5

# 4. 三模型集成
python src/ensemble/ensemble.py \
  --pred_files data/ensemble_baselines/qwen3vl_6030_test_preds.csv \
               data/ensemble_baselines/swinb_test_preds.csv \
               data/ensemble_baselines/convnext_test_preds.csv \
  --names qwen3vl_6030 swinb convnext \
  --template_csv data/sample_submission.csv \
  --output_dir data/ensemble_results \
  --primary_idx 0 \
  --hierarchical_low 0.3 \
  --hierarchical_high 0.7 \
  --threshold 0.5
```

其中 `data/sample_submission.csv` 已经修正为本轮真实测试集对应的 194 行：

```text
000001.jpg ~ 000194.jpg
```

## 5. 已完成实验记录

```

### 5.1 单模型输出统计

| 模型 | 输出文件 | 测试数 | 正例数 | 概率均值 |
|---|---|---:|---:|---:|
| Qwen3-VL | `qwen3vl_6030_test_preds.csv` | 194 | 117 | 0.5661 |
| Swin-B | `swinb_test_preds.csv` | 194 | 80 | 0.4016 |
| ConvNeXt | `convnext_test_preds.csv` | 194 | 81 | 0.4332 |

这里能明显看到 Qwen 的预测更偏正例，Swin/ConvNeXt 更保守。

### 5.2 初始三模型集成结果

输出文件：

```text
data/ensemble_results/ensemble_hard_20260601_181729.csv
data/ensemble_results/ensemble_soft_20260601_181729.csv
data/ensemble_results/ensemble_hierarchical_20260601_181729.csv
```

| 策略 | 正例数 | 负例数 | Kaggle 分数 |
|---|---:|---:|---:|
| Qwen 单模型 | 117 | 77 | 0.769 |
| Hard voting | 88 | 106 | 0.740 |
| Soft voting | 87 | 107 | 未提交/低优先级 |
| Hierarchical | 99 | 95 | 0.756 |

结论：初始集成没有提升，反而低于 Qwen 单模型。

### 5.3 为什么集成变差

对比 ensemble 和 Qwen 单模的差异：

| 策略 | 与 Qwen 不同的样本数 | Qwen=1 被改成 0 | Qwen=0 被改成 1 |
|---|---:|---:|---:|
| Hard | 35 | 32 | 3 |
| Soft | 40 | 35 | 5 |
| Hierarchical | 24 | 21 | 3 |

这说明当前集成主要做了一件坏事：**把 Qwen 判断为陨石的样本大量压成非陨石**。

也就是说，Swin-B 和 ConvNeXt 没有提供足够互补的错误修正，反而因为更保守，把强模型 Qwen 的正例判断拉低了。

这个现象解释了为什么：

- Qwen 单模型：0.769
- hierarchical：0.756
- hard：0.740

当前测试集上，Qwen 是主力，CNN 分支不能和 Qwen 平权投票。

## 6. 当前得到的经验

### 6.1 不能用 naive hard voting

Hard voting 默认每个模型一票。对于三模型：

```text
Qwen = 1, Swin = 0, ConvNeXt = 0  => ensemble = 0
```

如果 Qwen 是明显更强的模型，这种多数投票会系统性削弱 Qwen。

### 6.2 Soft voting 也不一定安全

虽然 soft voting 看起来比 hard voting 平滑，但它隐含一个前提：

> 不同模型的 `prob_m` 是同一尺度、同样校准过的概率。

实际不是这样。

Qwen 的概率来自 first-token logits，Swin/ConvNeXt 的概率来自分类 softmax。它们的概率尺度不一致，直接平均可能会破坏 Qwen 的判断。

### 6.3 Hierarchical 的阈值不能太宽

初始 hierarchical 设置：

```text
low = 0.3
high = 0.7
```

这导致 Qwen 有 47 张图被视为“不确定”，交给 Swin/ConvNeXt。最后其中 24 张和 Qwen 预测不同，降分。

下一步如果继续 hierarchical，应该把不确定区间缩窄，比如：

```text
low = 0.45
high = 0.55
```

只在 Qwen 真正接近 0.5 时才让辅助模型介入。

```

比较推荐的候选：

| 候选 | 改动逻辑 | 正例数 | 与 Qwen 不同数 |
|---|---|---:|---:|
| `qwen_boost_bothcnn1_qabove45.csv` | 仅当 Qwen=0 且两个 CNN 都=1 且 Qwen prob > 0.45 时翻成 1 | 118 | 1 |
| `qwen_boost_bothcnn1_qabove30.csv` | 类似上面，但阈值更宽 | 119 | 2 |
| `soft_q90_th45.csv` | Qwen 权重 0.9，Swin/ConvNeXt 总权重 0.1，阈值 0.45 | 120 | 3 |
| `qwen_union_6030_teammate.csv` | 两个 Qwen 方法取并集 | 121 | 4 |
| `qwen_teammate_only_quncertain_45_55.csv` | 只在 Qwen 概率 0.45-0.55 时参考 teammate Qwen | 118 | 5 |

这些候选的共同思想是：

> 不要让 CNN 大规模否决 Qwen，只让它们在极少数情况下补充 Qwen 可能漏掉的正例。

## 8. PPT / Pre 可用讲法

队友做 PPT 或 pre 时，可以按下面逻辑组织这一部分：

1. 模型集成模块负责把不同模型输出统一为 `id,label,prob_m`，并生成最终 submission。
2. 这部分重写了 ensemble engine，重点解决提交安全性：
   - 严格按 sample submission 对齐
   - 缺失/重复/多余 id 直接报错
   - 输出 debug 和 report
3. Qwen inference 的默认配置与队友测试脚本对齐，保证进入 ensemble 的 Qwen CSV 口径一致。
4. 完成了三模型集成实验：
   - Qwen
   - Swin-B
   - ConvNeXt
5. 实验发现 naive ensemble 降分：
   - Qwen 单模 0.769
   - hierarchical 0.756
   - hard 0.740
6. 原因分析：
   - CNN 分支更保守
   - hard/soft/hierarchical 把很多 Qwen 正例改成负例
   - 三模型不是互补关系，而是弱模型拉低强模型
7. 下一步策略：
   - 做 Qwen-dominant ensemble
   - 缩窄 hierarchical 不确定区间
   - 多 Qwen 推理配置 / TTA 比 CNN 平权投票更有潜力
   - 如果有验证集，做概率校准和 stacking

## 9. 下一步计划


### 9.2 做 Qwen-family ensemble

比起让 CNN 平权投票，更有希望的是多个 Qwen 推理配置 / Qwen inference method 集成：

- 带先验的 instruction
- 不带先验的 instruction
- 强调忽略文字、标签、标尺
- 强调先判断石头主体，再判断陨石特征
- first-token logits
- generate 文本解析
- 多尺度 / TTA

这些模型的能力来源一致，但推理配置和输出解析方式不同，可能比 CNN 更互补。

### 9.3 做概率校准

目前 `prob_m` 没有校准：

- Qwen: first-token logits
- Swin: classification softmax
- ConvNeXt: classification softmax

后续可以在 holdout 上做：

- temperature scaling
- logistic regression stacking
- isotonic regression

然后再 soft voting，而不是直接平均原始概率。

### 9.4 人工检查不确定样本

测试集只有 194 张，可以重点看：

```text
Qwen prob in [0.4, 0.6]
```

这些样本是最可能通过人工规则或推理配置改进提升分数的部分。

## 10. 当前结论

目前最重要的结论是：

> 集成不是简单投票。当前最强模型是 Qwen，Swin/ConvNeXt 不能和它平权。后续集成应该以 Qwen 为主，只在 Qwen 不确定或两个辅助模型强一致时做小范围修正。

短期最稳的提交仍然是 Qwen 单模；如果必须尝试 ensemble，应优先尝试只改动 1-3 个样本的 Qwen-dominant candidates，而不是 hard/soft/hierarchical 这种会大规模改变 Qwen 判断的版本。
