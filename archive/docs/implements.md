# 数据增强模块实现计划

基于 `data/merged/training_data_merged_reformatted.jsonl` 实现四种增强方法，
目标：抑制 8B 模型在有限 SFT 数据上的过拟合，对抗 train-test 域偏移。

---

## 模块一：TokenMixUp（Embedding 空间混合）

### 文件：修改 `train_lora.py` 中 `QwenVLDataCollator`

### 原理

在每个 batch 内，以概率 `mixup_prob` 对样本做随机配对，对 `pixel_values`
做线性插值 `pixel_values' = λ·A + (1-λ)·B`。模型对混合图片 forward 时，
视觉编码器产生"中间特征"响应，天然形成正则化。**不修改 text、不修改 labels
数值**（VLM 是 token generation，label 是离散 token ID，数值插值无意义）。

### 修改点

**1. `QwenVLDataCollator.__init__` 添加参数**

```python
def __init__(self, processor, mixup_prob=0.5, mixup_alpha=0.2):
    ...
    self.mixup_prob = mixup_prob
    self.mixup_alpha = mixup_alpha
```

**2. `QwenVLDataCollator.__call__` 末尾（`batch["labels"] = torch.stack(labels)` 之后、`return batch` 之前）添加混合逻辑**

```python
import torch
import numpy as np

# TokenMixUp：对 pixel_values 做 batch 内随机配对混合
if np.random.random() < self.mixup_prob and len(examples) > 1:
    batch_size = len(examples)
    perm = torch.randperm(batch_size)
    lam = float(np.random.beta(self.mixup_alpha, self.mixup_alpha))

    # pixel_values 形状: [B, C, H, W] 或 [B, N, D] 取决于 Qwen3-VL 内部处理
    # 对第一个维度做插值
    pv = batch["pixel_values"]
    mixed_pv = lam * pv + (1 - lam) * pv[perm]
    batch["pixel_values"] = mixed_pv

    # image_grid_thw 保持不变（每个样本的 grid 结构不受影响）
    # labels 保持不变（离散 token ID，不做数值插值）
```

### 超参数（可通过 `train_lora.py` 顶层变量控制）

| 参数 | 默认值 | 说明 |
|---|---|---|
| `mixup_prob` | 0.5 | 每个 batch 触发混合的概率 |
| `mixup_alpha` | 0.2 | Beta(α,α) 分布参数，值小 → λ 倾向 0 或 1，保留更多信息 |

### 验证方式

- 训练时观察 train loss 收敛速度应稍慢于 baseline（正则化效果）
- 对比 train/val loss gap 是否缩小

---

## 模块二：离线域降质增强

### 文件一：新建 `src/data_processing_modules/DomainDegradationAugmentor.py`

### 原理

模拟测试集的域偏移特征（更低分辨率、更暗、更单调色彩、更模糊），离线生成
降质图像，写入磁盘后通过 `resample_without_extracted.py` 合并到训练数据。

### 实现细节

```python
class DomainDegradationAugmentor:
    """对岩石图像施加域降质，模拟测试集特征"""

    def __init__(self, save_quality=85):
        self.save_quality = save_quality

    def _degrade_resolution(self, img: Image.Image) -> Image.Image:
        """降分辨率：缩放到 256x256 再放大回原尺寸"""
        w, h = img.size
        small = img.resize((256, 256), Image.Resampling.BILINEAR)
        return small.resize((w, h), Image.Resampling.BILINEAR)

    def _degrade_brightness(self, img: Image.Image) -> Image.Image:
        """降低亮度：因子 0.5~0.9"""
        factor = random.uniform(0.5, 0.9)
        enhancer = ImageEnhance.Brightness(img)
        return enhancer.enhance(factor)

    def _degrade_saturation(self, img: Image.Image) -> Image.Image:
        """降低饱和度：因子 0.3~0.7"""
        factor = random.uniform(0.3, 0.7)
        enhancer = ImageEnhance.Color(img)
        return enhancer.enhance(factor)

    def _apply_blur(self, img: Image.Image) -> Image.Image:
        """高斯模糊：σ = 0.5~2.0"""
        sigma = random.uniform(0.5, 2.0)
        return img.filter(ImageFilter.GaussianBlur(radius=sigma))

    def _degrade_jpeg(self, img: Image.Image) -> Image.Image:
        """随机 JPEG 压缩：quality 30~60"""
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=random.randint(30, 60))
        buf.seek(0)
        return Image.open(buf)

    def apply(self, img: Image.Image) -> Image.Image:
        """随机选择 1~3 种降质操作组合应用"""
        ops = [
            self._degrade_resolution,
            self._degrade_brightness,
            self._degrade_saturation,
            self._apply_blur,
            self._degrade_jpeg,
        ]
        n_ops = random.randint(1, 3)
        chosen = random.sample(ops, n_ops)
        for op in chosen:
            img = op(img)
        return img

    def process_dataset(self, image_paths, output_dir,
                        augmentation_factor=1.0,
                        skip_existing=True):
        """
        遍历图像目录，为每张图生成 augmentation_factor 份降质副本。
        返回 (成功数, 失败数)。
        命名格式：000001_deg0.jpg, 000001_deg1.jpg, ...
        """
        ...
```

### 文件二：修改 `resample_without_extracted.py`

**新增扫描 `train_images_degraded/` 目录**（与现有 `train_images_augmented/` 逻辑一致）：

```python
DEGRADED_IMG_DIR = os.path.join(DATA_ROOT, "train_images_degraded")

# 在 main() 中新增：
# 3. 处理降质图片目录
if os.path.exists(DEGRADED_IMG_DIR):
    degraded_images = [f for f in os.listdir(DEGRADED_IMG_DIR)
                       if f.endswith(('.jpg', '.jpeg', '.png'))]
    for img_name in tqdm(degraded_images, desc="降质图片"):
        # 从文件名提取原始 ID（如 000001_deg0.jpg -> 000001.jpg）
        base_num = img_name.split('_deg')[0]
        img_id_str = f"{base_num}.jpg"
        label = label_dict.get(img_id_str)
        if label is None:
            continue
        output = "是" if label == 1 else "否"
        data_entry = {
            "image_path": os.path.join(DEGRADED_IMG_DIR, img_name),
            "instruction": SIMPLE_INSTRUCTION,
            "output": output,
        }
        if label == 1:
            meteorite_data.append(data_entry)
        else:
            non_meteorite_data.append(data_entry)
```

### 文件三：新建执行脚本

`src/data_processing_modules/gen_degraded_data.py`

```python
"""生成域降质增强图像，输出到 data/train_images_degraded/"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from src.data_processing_modules.DomainDegradationAugmentor import DomainDegradationAugmentor

DATA_ROOT = os.path.join(os.path.dirname(__file__), "../../data")
IMG_DIR = os.path.join(DATA_ROOT, "train_images")
OUTPUT_DIR = os.path.join(DATA_ROOT, "train_images_degraded")

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    augmentor = DomainDegradationAugmentor()
    image_paths = [os.path.join(IMG_DIR, f) for f in os.listdir(IMG_DIR)
                   if f.endswith(('.jpg', '.jpeg', '.png'))]
    count, fail = augmentor.process_dataset(
        image_paths, OUTPUT_DIR, augmentation_factor=2.0, skip_existing=False)
    print(f"降质增强完成: {count} 成功, {fail} 失败")
```

---

## 模块三：多尺度训练

### 文件：修改 `train_lora.py` 中 `QwenVLDataCollator.__call__`

### 原理

训练时随机选择不同尺寸加载图像，使模型适应不同分辨率下的纹理表达。

### 修改点

**将 `QwenVLDataCollator.__init__` 中添加 `scales` 参数**：

```python
def __init__(self, processor, scales=(384, 512, 720), ...):
    ...
    self.scales = scales
```

**将图像加载处的固定 720 改为随机选择**：

原代码（第 227 行）：
```python
img.thumbnail((720, 720), Image.Resampling.LANCZOS)
```

替换为：
```python
scale = random.choice(self.scales)
img.thumbnail((scale, scale), Image.Resampling.LANCZOS)
```

### 超参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `scales` | (384, 512, 720) | 候选尺寸列表 |

### 注意

- 评估时仍使用固定 720 尺寸（与测试时一致），避免评估波动
- 可在 collator 中通过 `training=True/False` 参数区分，或直接
  在 `__call__` 中检查 `self.processor.tokenizer.padding_side`
  （训练时 right，评估时 left）

---

## 模块四：测试时增强（TTA）

### 文件：修改 `src/testing/test_model.py` 的 `predict()` 函数

### 原理

对每张测试图做多种变换后分别推理，对 softmax 概率取平均，
降低单次推理的方差。

### 变换列表

| 变换 | 说明 |
|---|---|
| 原始 720 | 基准 |
| 原始 512 | 多尺度 |
| 原始 384 | 多尺度 |
| 水平翻转 720 | 视角变化 |
| 水平翻转 512 | 多尺度 + 翻转 |

### 修改后的 `predict` 函数签名

```python
TTA_CONFIGS = [
    {"scale": 720, "flip_h": False},
    {"scale": 512, "flip_h": False},
    {"scale": 384, "flip_h": False},
    {"scale": 720, "flip_h": True},
    {"scale": 512, "flip_h": True},
]

def predict_with_tta(model, processor, test_loader, device,
                     tta_configs=TTA_CONFIGS):
    """
    带 TTA 的预测：对每张图的多种变换版本分别推理，取概率平均。
    返回 dict {filename: label}
    """
    model.eval()
    id_to_logits = {}  # filename -> 累计 logit

    with torch.no_grad():
        for config in tta_configs:
            for batch in tqdm(test_loader, desc=f"TTA scale={config['scale']} flip={config['flip_h']}"):
                filenames, images = zip(*batch)
                filenames = list(filenames)
                scaled_images = []
                for img in images:
                    img = img.copy()
                    if config["flip_h"]:
                        img = img.transpose(Image.FLIP_LEFT_RIGHT)
                    img.thumbnail((config["scale"], config["scale"]), Image.Resampling.LANCZOS)
                    scaled_images.append(img)

                messages = [...]  # 与现有相同
                texts = [...]
                inputs = processor(text=texts, images=scaled_images, return_tensors="pt", padding=True).to(device)

                # 获取 logits 而非生成结果，用于概率平均
                outputs = model(**inputs)
                logits = outputs.logits  # [B, seq_len, vocab]

                for i, fn in enumerate(filenames):
                    if fn not in id_to_logits:
                        id_to_logits[fn] = 0.0
                    # 取 "是"/"否" 对应 token 的 logit 差值作为分类得分
                    # 需要在 vocab 中找到这两个 token 的 id
                    ...

    # 平均后判定
    id_to_pred = {}
    for fn, logit in id_to_logits.items():
        id_to_pred[fn] = 1 if logit > 0 else 0

    return id_to_pred
```

### 关键实现细节

1. **"是"/"否" token 定位**：需要在 Qwen3-VL tokenizer 中找到 "是" 和 "否" 对应的 token ID：
   ```python
   YES_TOKEN = processor.tokenizer.encode("是", add_special_tokens=False)[0]
   NO_TOKEN = processor.tokenizer.encode("否", add_special_tokens=False)[0]
   ```

2. **概率平均方式**：对最后一个生成位置的 logits 取 softmax，然后对概率平均：
   ```python
   # 找到 assistant 回复中第一个 token 的位置
   # 取该位置 logits 的 YES_TOKEN 和 NO_TOKEN 的 softmax 概率
   # 对多个 TTA 版本的概率取平均
   ```

3. **显存管理**：TTA 会多次遍历测试集，确保每个 config 之间清理显存：
   ```python
   torch.cuda.empty_cache()
   ```

4. **TTA 开关**：通过参数 `use_tta=True/False` 控制，默认开启。
   `gen_submission_v2.py` 中调用时传入 `use_tta=True`。

---

## 实施顺序

1. **TokenMixUp** → 最快实现（只改 collator，不改数据），立即可训练验证
2. **离线域降质** → 生成数据 + 更新 resample 脚本
3. **多尺度训练** → 在 collator 中一行改动，与 TokenMixUp 合并
4. **TTA** → 推理端改动，需配合 gen_submission_v2.py 使用

## 消融实验计划

依次叠加各策略，记录 Kaggle F1：

| 实验 | 策略 | 预期 F1 |
|---|---|---|
| Baseline | 无增强 | ~0.812 |
| Exp-1 | +TokenMixUp |  |
| Exp-2 | +TokenMixUp+域降质 |  |
| Exp-3 | +TokenMixUp+域降质+多尺度 |  |
| Exp-4 | 全量（含TTA推理） |  |
