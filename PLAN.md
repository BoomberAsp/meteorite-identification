## 未来执行方案

基于中期报告识别出的核心问题——**训练-测试域偏移严重**、**标注信息过于简陋**——制定以下三个方向的执行方案。

### 方向一：增强标注质量，引导模型学习正确判据

**问题**：当前 SFT 数据仅包含”是/否”回答，缺乏判断理由，模型依赖预训练中的通用视觉逻辑，而非系统性地学习陨石判别规则（熔壳、气印、磁性等特征）。

**执行步骤**：

1. **生成高质量判别理由**
   - 利用强视觉语言模型 API（qwen3.6-plus / GPT-4o / Gemini3.1-pro）为已有标签的训练样本生成详细判别理由
   - prompt 设计：要求模型按”观察 → 推理 → 结论”链输出，覆盖熔壳、气印、颜色、纹理、形状五大维度
   - 已知标签模式：让模型解释”为什么是/不是陨石”，避免误判
   - 已有基础设施：`src/data_processing_modules/test_gen_labels.py` 可直接复用，当前已生成 10 条样例（`vl_model_reasoning.jsonl`，5 正 5 负，均成功）

2. **质量控制与过滤**
   - 人工抽检：随机抽取 50-100 条，验证理由的准确性和相关性
   - 置信度过滤：仅保留 API 返回 `confidence >= 0.8` 的样本
   - 一致性检查：对同一图片多次调用，选择理由最一致的版本

3. **构造 ChatML 格式数据**
   - 格式：`{“image_path”: “...”, “instruction”: “请观察这张图片...说明理由”, “output”: “是/否。理由：...”}`
   - 复用 `gen_SFT_data.py` 的生成管线，替换 output 字段为高质量理由

4. **训练验证**
   - 对比实验：简单标注 vs 详细理由标注，在相同训练设置下评估 Kaggle F1
   - 预期：模型从”黑盒分类”转向”依规则推理”，提升泛化能力

### 方向二：面向域偏移的数据增强与多尺度策略

**问题**：测试集图片普遍尺寸更小、亮度更低、色彩更单调，当前训练未有效模拟这些降质，模型对低质量图像鲁棒性不足。

#### 2.1 TokenMixUp（embedding 空间混合）

**原理**：在 `QwenVLDataCollator` 中，每 batch 内随机配对样本，对它们的 `pixel_values` 做线性插值后送入模型 forward，loss 按混合比例 λ 加权。只对 image embedding 混合，text 保持各样本自身完整内容。

**执行步骤**：

1. **修改 `train_lora.py` 的 `QwenVLDataCollator`**
   - 在 `__call__` 末尾，以概率 `mixup_prob=0.5` 触发混合
   - 对 batch 内样本做随机置换 `perm`，取 `λ ~ Beta(α, α)`（α=0.2，倾向极端值使混合不过于平滑）
   - 对 `pixel_values` 执行：`pixel_values = λ * pixel_values + (1-λ) * pixel_values[perm]`
   - 对 `labels` 做对应加权：
     - 原始 labels 中 `-100`（mask 位）保持不变
     - 有效 token 位置：`labels = λ * labels_A + (1-λ) * labels_B`
     - 实际实现：不直接修改 labels 数值（VLM loss 依赖离散 token），而是**在 loss 层面加权**——将配对样本的 labels 分别保留，loss = λ * loss_A + (1-λ) * loss_B
   - 简化实现：不修改 labels，只混合 `pixel_values`。模型对混合图片的 forward 天然产生"中间特征"响应，这本身就是一种 strong regularization，无需 label 加权

2. **关键设计决策**
   - **只对 `pixel_values` 混合**，不混合 `image_grid_thw`（保持 grid 结构不变）
   - **不修改 instruction/output**——每个样本的 text 保持完整 reasoning 链
   - **不修改 labels 数值**——VLM 是 token generation 任务，label smoothing 无意义；pixel 混合产生的特征扰动已足够作为正则化

3. **超参数**
   - `mixup_prob`: 0.5（50% 的 batch 触发混合）
   - `mixup_alpha`: 0.2（Beta(0.2, 0.2) 倾向 λ 接近 0 或 1，保留更多原始信息）
   - 可通过实验调整：观察 train loss 收敛速度与 val loss 的 gap

#### 2.2 离线域降质增强（生成到 `train_images_augmented/`）

**原理**：模拟测试集的域偏移特征，离线生成降质图像，与现有旋转/缩放增强合并使用。

**执行步骤**：

1. **新增 `DomainDegradationAugmentor` 模块**（`src/data_processing_modules/DomainDegradationAugmentor.py`）
   - **降分辨率**：随机缩小至 256x256 → 双线性插值回原尺寸（模拟低分辨率）
   - **降低亮度**：随机因子 0.5~0.9（模拟测试集暗色图像）
   - **降低饱和度**：随机因子 0.3~0.7（模拟测试集单调色彩）
   - **高斯模糊**：σ = 0.5~2.0（模拟运动模糊/失焦）
   - **随机 JPEG 压缩**：quality 30~60（模拟低质量传输）
   - 每次从上述 5 种操作中随机选择 1~3 种组合应用（非全部串联，避免过度降质）
   - 输出到 `train_images_degraded/` 目录，命名格式 `000001_deg0.jpg`

2. **合并到训练数据管线**
   - 修改 `resample_without_extracted.py`，增加扫描 `train_images_degraded/` 目录
   - 保持 5:1 采样比例

3. **注意**：避免大角度旋转和仿射变换破坏陨石关键鉴别特征（熔壳、气印）

#### 2.3 多尺度训练

**执行步骤**：

1. **修改 `train_lora.py` 中 `QwenVLDataCollator` 的 `img.thumbnail()`**
   - 替换固定尺寸 720 为随机选择：`random.choice([384, 512, 720])`
   - 每个样本在训练时随机切换尺寸，不同尺寸共享同一模型权重

#### 2.4 测试时增强（TTA）

**执行步骤**：

1. **在 `test_model.py` 的 `predict()` 中实现**：
   - 水平翻转
   - 多尺度推理（720、512、384）
   - 对各版本的 softmax 概率取平均，得到最终预测
   - 预期稳定预测结果，降低单次推理的方差

#### 2.5 验证方案

1. **消融实验**：依次叠加各策略，记录 Kaggle F1
   - Baseline → +TokenMixUp → +域降质 → +多尺度 → +TTA
2. **组合最优策略进行最终训练**

### 方向三：模型集成

**问题**：单一架构模型的错误模式固定，预测方差较高。

**执行步骤**：

1. **异构模型选择**
   - ViT 路线：Qwen3-VL-8B-Instruct（当前最优，F1 > 0.812）
   - CNN 路线：ConvNeXt-Large（F1 = 0.759）、Swin-B（F1 = 0.700）
   - 注意：同源模型集成（如 ResNet18 + ResNet50）因错误模式高度重叠，未能带来明显增益

2. **增加模型多样性**
   - 对 Qwen3-VL 使用不同随机种子训练多组 LoRA 权重
   - 对 CNN 模型使用不同增强策略（强/轻增强、SAM2 提取/不提取）

3. **集成策略**
   - 软投票：各模型输出 softmax 概率加权平均
   - 硬投票：各模型预测标签取多数
   - 分层集成：先用 Qwen3-VL 做主模型，对置信度低的样本（概率接近 0.5）引入 CNN 模型辅助判断

4. **验证**
   - 集成后 Kaggle F1 应高于任一单模型
   - 分析集成在哪些样本上改进了预测，验证异构集成的有效性

### 预期效果

通过高质量标注数据使 Qwen 模型学会系统性的陨石鉴别逻辑，配合域适应增强和多尺度 TTA 对抗域偏移，探索更精准的目标区域聚焦方法，最终提升模型在测试集上的泛化能力。

