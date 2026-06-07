# Meteorite Identification — Project Documentation

## 项目概述

基于视觉语言大模型 Qwen3-VL-8B-Instruct 的陨石图像二分类项目。项目使用 LoRA 对 Qwen3-VL-8B-Instruct 进行微调，使其能够根据石头图片的视觉特征（熔壳、气印、颜色、纹理、形状等）判断该石头是否为陨石（`是` / `否`）。整个管线涵盖数据预处理（SAM2 石头提取、数据增强）、SFT 数据集构建、LoRA 微调训练、模型评估和推理提交。

---

## 项目运行模块图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           DATA PREPARATION                              │
│                                                                         │
│  train_labels.csv + train_images ──────────────────┐                    │
│  test_images           ──────────────────────────┐ │                    │
│                                                   ▼ ▼                    │
│  ┌──────────────────────────────────────────────────────────────┐       │
│  │ StoneExtractor.py  (SAM2.1 分割提取石头主体)                   │       │
│  │   → train_images_extracted / test_images_extracted           │       │
│  └──────────────────────────────────────────────────────────────┘       │
│                                                   │                     │
│  ┌────────────────────────────────────────────────┘                     │
│  │ ImageAugmentor.py  (旋转+缩放数据增强)                                │
│  │   → train_images_extracted_augmented                                 │
│  └────────────────────────────────────────────────┘                     │
│                                                   │                     │
│  ┌────────────────────────────────────────────────┘                     │
│  │ gen_SFT_data.py  (构建 SFT JSONL 训练数据)                           │
│  │   → qwen3_sft_train.jsonl                                            │
│  └────────────────────────────────────────────────┘                     │
│                                                   │                     │
│  ┌────────────────────────────────────────────────┘                     │
│  │ update_train_data_path.py  (更新 JSONL 中的图片路径)                 │
│  │   → qwen3_sft_train_extracted.jsonl                                  │
│  └────────────────────────────────────────────────┘                     │
│                                                   │                     │
│  ┌────────────────────────────────────────────────┘                     │
│  │ update_train_data_augmented.py (合并增强数据)                        │
│  │   → qwen3_sft_train_extracted_augmented.jsonl                        │
│  └────────────────────────────────────────────────┘                     │
└─────────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         TRAINING (LoRA Fine-tuning)                      │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────┐       │
│  │ downloading_model.py (从 ModelScope 下载基座模型)              │       │
│  │   → models/Qwen3-VL-8B-Instruct/                              │       │
│  └──────────────────────────────────────────────────────────────┘       │
│                                                   │                     │
│  ┌────────────────────────────────────────────────┘                     │
│  │ train_lora.py  (LoRA 微调训练 + 验证)                                │
│  │   → models/qwen3-vl-lora-meteorite_YYYYMMDD_HHMMSS/                 │
│  │   → models/qwen3-vl-lora-meteorite_YYYYMMDD_HHMMSS/final_adapter_*/ │
│  └────────────────────────────────────────────────┘                     │
└─────────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       EVALUATION & TESTING                               │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────┐       │
│  │ eval_train_val.py  (训练集 vs 验证集评估，诊断过拟合)           │       │
│  │   → data/logs/eval_train_val_YYYYMMDD_HHMMSS.json            │       │
│  └──────────────────────────────────────────────────────────────┘       │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────┐       │
│  │ test_model.py  (加载 LoRA 模型，批量推理)                      │       │
│  │   load_model() → predict() → {filename: label}               │       │
│  └──────────────────────────────────────────────────────────────┘       │
│                                                   │                     │
│  ┌────────────────────────────────────────────────┘                     │
│  │ gen_submission_files.py (生成 Kaggle submission.csv)                 │
│  │   → data/logs/submission_*.csv                                       │
│  └────────────────────────────────────────────────┘                     │
└─────────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      VISUALIZATION & MONITORING                          │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────┐       │
│  │ test_gd_sam2_visualization.py (GD + SAM2 检测/分割可视化)      │       │
│  │   → data/visualization_results/*.jpg / *.png                 │       │
│  └──────────────────────────────────────────────────────────────┘       │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────┐       │
│  │ ploting.py  (训练 loss 曲线绘制)                               │       │
│  │   → data/logs/loss_curves.png                                │       │
│  └──────────────────────────────────────────────────────────────┘       │
│                                                                         │
│  run_train.sh  → 启动 TensorBoard + train_lora.py                       │
│  run_tensorBoard.sh → 仅启动 TensorBoard (端口 6007)                    │
│                                                                         │
│  test_gen_labels.py → 调用视觉大模型 API (qwen3.5-plus) 生成推理理由     │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 项目目录结构

```
meteorite-identification/
├── readme.md                          # 原始说明文档（submission 生成教程）
├── ReadMe.md                          # 本文档：完整项目文档
├── Warning.md                         # Git 提交注意事项
├── run_train.sh                       # 训练启动脚本（TensorBoard + 训练）
├── run_tensorBoard.sh                 # TensorBoard 启动脚本
│
├── models/
│   ├── Qwen3-VL-8B-Instruct/          # 基座视觉语言模型
│   ├── qwen3-vl-lora-meteorite/       # LoRA 权重（多组实验 checkpoint）
│   ├── Qwem2-VL-7B/                   # 备用模型
│   └── grounding_dino/               # GroundingDINO 目标检测模型
│
├── data/
│   ├── dataset.py                     # PyTorch Dataset 类
│   ├── train_labels.csv               # 训练集标注（id, label）
│   ├── sample_submission.csv          # 提交模板
│   ├── train_images/                  # 原始训练图片
│   ├── test_images/                   # 原始测试图片
│   ├── train_images_extracted/       # SAM2 提取后的训练图片
│   ├── test_images_extracted/        # SAM2 提取后的测试图片
│   ├── train_images_extracted_augmented/  # 增强后的训练图片
│   ├── qwen3_sft_train*.jsonl        # 各阶段 SFT 训练数据
│   └── logs/                          # 评估日志、TensorBoard 日志
│
└── src/
    ├── data_processing_modules/
    │   ├── StoneExtractor.py          # SAM2 石头主体提取
    │   ├── ImageAugmentor.py          # 图像数据增强
    │   ├── update_train_data_path.py  # JSONL 路径更新（extracted）
    │   ├── update_train_data_augmented.py  # JSONL 合并增强数据
    │   └── test_gen_labels.py         # 视觉大模型 API 推理
    │
    ├── fine-tuning_(training)/
    │   ├── downloading_model.py       # 模型下载
    │   ├── gen_SFT_data.py            # SFT 数据集生成
    │   └── train_lora.py             # LoRA 微调训练
    │
    ├── testing/
    │   ├── test_model.py             # 模型加载与批量预测
    │   ├── gen_submission_files.py   # 生成 submission.csv
    │   ├── eval_train_val.py         # 训练集/验证集评估
    │   ├── test_dir.py               # 模型快速测试
    │   ├── test_gd_sam2_visualization.py  # GD+SAM2 可视化
    │   └── test_model.py (predict / make_submission)
    │
    └── ploting.py                     # Loss 曲线绘制
```

---

## 核心类与函数详解

### 1. StoneExtractor — SAM2 石头主体提取

**文件**: `src/data_processing_modules/StoneExtractor.py`

使用 SAM2.1 分割模型从石头图片中提取主体区域，去除背景干扰。

#### `class StoneExtractor`

```python
class StoneExtractor:
    def __init__(
        self,
        sam2_predictor: Optional[Any] = None,
        device: str = "cuda",
        background_value: int = 128,
        save_quality: int = 95,
        padding: int = 10,
    )
```

**参数**:
- `sam2_predictor`: SAM2 图像预测器实例
- `device`: 计算设备 (`"cuda"` / `"cpu"`)
- `background_value`: 背景填充灰度值 (0-255)，默认 128（灰色）
- `save_quality`: JPEG 保存质量 (1-100)，默认 95
- `padding`: 裁剪边距（像素），默认 10

**工作流程**: 以图像中心点为 prompt → SAM2 生成分割掩码 → 根据掩码边界裁剪石头主体 → 背景填充灰色 → 输出裁剪后图片（不改变分辨率）。

##### `classmethod from_pretrained()`

```python
@classmethod
def from_pretrained(
    cls,
    sam2_checkpoint: str = ".../sam2.1_hiera_large.pt",
    sam2_config: str = "configs/sam2.1/sam2.1_hiera_l.yaml",
    device: str = "cuda",
    **kwargs,
) -> StoneExtractor
```

从预训练模型加载 SAM2.1 large，构建并返回 `StoneExtractor` 实例。加载时自动切换到 SAM2 仓库目录以正确解析配置文件。

##### `_get_center_point_mask()`

```python
def _get_center_point_mask(self, image_np: np.ndarray) -> np.ndarray
```

以图像中心点 `(w//2, h//2)` 为前景 prompt 调用 SAM2，返回得分最高的布尔掩码 `(H, W)`。

##### `_get_mask_bbox()`

```python
def _get_mask_bbox(
    self,
    mask: np.ndarray,
    padding: int = 0
) -> Tuple[int, int, int, int]
```

从布尔掩码中提取边界框 `(x1, y1, x2, y2)`，支持 padding 扩展。若掩码无效则返回整图边界。

##### `process_image()`

```python
def process_image(
    self,
    image_path: str,
    output_path: str
) -> bool
```

处理单张图片：读取 → SAM2 分割 → 裁剪掩码区域 → 背景填充 → 保存。返回成功/失败状态。

##### `process_batch()`

```python
def process_batch(
    self,
    image_paths: List[str],
    output_dir: str,
    skip_existing: bool = True
) -> Tuple[int, int]
```

批量处理图片列表，返回 `(成功数量, 失败数量)`。每 50 张自动清理显存。

##### `process_dataset()`

```python
def process_dataset(
    self,
    data_root: str,
    process_train: bool = True,
    process_test: bool = True,
    skip_existing: bool = True
) -> Dict[str, Tuple[int, int]]
```

处理完整数据集，分别处理 `train_images/` 和 `test_images/`，输出到 `train_images_extracted/` 和 `test_images_extracted/`。返回 `{'train': (success, fail), 'test': (success, fail)}`。

##### `release()`

```python
def release(self)
```

释放 SAM2 模型资源和 GPU 显存。

---

### 2. ImageAugmentor — 图像数据增强

**文件**: `src/data_processing_modules/ImageAugmentor.py`

通过随机旋转和缩放生成新的训练样本。

#### `class ImageAugmentor`

```python
class ImageAugmentor:
    def __init__(
        self,
        rotation_range: Tuple[float, float] = (-30, 30),
        scale_range: Tuple[float, float] = (0.8, 1.2),
        fill_color: Tuple[int, int, int] = (128, 128, 128),
        save_quality: int = 95,
    )
```

**参数**:
- `rotation_range`: 随机旋转角度范围（度），默认 ±30°
- `scale_range`: 随机缩放比例范围，默认 0.8x ~ 1.2x
- `fill_color`: 旋转后背景填充色，默认灰色 (128, 128, 128)
- `save_quality`: JPEG 保存质量

##### `augment()`

```python
def augment(self, image: Image.Image) -> Image.Image
```

对单张 PIL 图像应用随机旋转 + 缩放，返回增强后的 PIL Image。

##### `augment_and_save()`

```python
def augment_and_save(
    self,
    image_path: str,
    output_path: str,
    augment_count: int = 1
) -> List[str]
```

读取图像，生成 `augment_count` 张增强版本并保存。当 `augment_count > 1` 时自动添加 `_aug0`, `_aug1` 等后缀。返回生成的文件路径列表。

##### `process_dataset()`

```python
def process_dataset(
    self,
    image_paths: List[str],
    output_dir: str,
    augmentation_factor: float = 2.0,
    skip_existing: bool = True
) -> Tuple[int, int]
```

对整个数据集进行增强。`augmentation_factor` 控制每张原图生成的增强样本数（2.0 表示每张生成 2 张增强图，总量变为 3 倍）。小数部分按概率处理。返回 `(成功数量, 失败数量)`。

---

### 3. StoneDataset — PyTorch 数据集

**文件**: `data/dataset.py`

#### `class StoneDataset(Dataset)`

```python
class StoneDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        transforms = None,
    )
```

标准 PyTorch Dataset。根据 `split` 自动加载训练集（含标签）或测试集（无标签）。

**目录结构要求**:
```
root/
├── train_images/
├── train_labels.csv      (id, label 两列)
├── test_images/
└── sample_submission.csv (id 列)
```

##### `__getitem__()`

```python
def __getitem__(self, index)
```

返回 `(image, label)`（训练模式）或 `(image, image_path)`（测试模式）。图像经过 `transforms` 处理（如果提供）。

##### `__len__()`

```python
def __len__(self) -> int
```

返回样本总数。

#### 辅助函数

```python
def resolve_train_paths(root) -> Tuple[str, str]
# 返回 (image_dir, csv_path)

def resolve_test_paths(root) -> Tuple[str, str]
# 返回 (image_dir, template_csv_path)
```

---

### 4. LoRA 微调训练

**文件**: `src/fine-tuning_(training)/train_lora.py`

核心训练脚本，使用 QLoRA（4-bit 量化 + LoRA）微调 Qwen3-VL-8B-Instruct。

#### `class EpochLossCallback(TrainerCallback)`

```python
class EpochLossCallback(TrainerCallback):
    def __init__(self)
    def on_epoch_begin(self, args, state, control, **kwargs)
    def on_step_end(self, args, state, control, logs=None, **kwargs)
    def on_epoch_end(self, args, state, control, logs=None, **kwargs)
```

自定义回调：记录每个 step 的 loss，在 epoch 结束时打印平均 loss 并与上一 epoch 对比。

#### `class DebugTrainer(Trainer)`

```python
class DebugTrainer(Trainer):
    def __init__(self, *args, **kwargs)
    def get_train_dataloader(self)
    def evaluation_loop(
        self,
        dataloader,
        description,
        prediction_loss_only,
        ignore_keys=None,
        metric_key_prefix="eval"
    ) -> EvalLoopOutput
```

扩展 Trainer：自动注册 `EpochLossCallback`，重写 `evaluation_loop` 以在每 batch 后释放显存，避免 OOM。

#### `class QwenVLDataCollator`

```python
class QwenVLDataCollator:
    def __init__(self, processor)
    def __call__(self, examples) -> Dict[str, torch.Tensor]
```

动态数据整理器：加载图像并缩放到 720x720 以内，构建多模态输入（`input_ids`, `attention_mask`, `pixel_values`, `image_grid_thw`），正确构建 `labels`（仅 assistant 回复部分参与 loss 计算，user 部分 mask 为 -100）。

#### `train()`

```python
def train()
```

主训练函数，完整流程：
1. 加载基座模型 `Qwen3-VL-8B-Instruct`（4-bit 量化）
2. 配置 LoRA（`r=64`, `lora_alpha=32`, `dropout=0.1`，目标模块包括 `q_proj`, `v_proj`, `k_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`）
3. 加载 JSONL 数据集并按 90/10 划分训练/验证集
4. 创建 `DebugTrainer` 和 `QwenVLDataCollator`
5. 开始训练（3 epochs, `lr=1e-4`, `batch_size=2`, `gradient_accumulation=4`, cosine 调度器）
6. 训练结束后用完整验证集评估，保存最终 LoRA adapter

#### `compute_metrics()`

```python
def compute_metrics(eval_pred) -> Dict[str, float]
```

评估指标函数，当前仅返回 `eval_loss`。

#### `downloading_model.py`

```python
# 从 ModelScope 下载基座模型
model_path = snapshot_download('qwen/Qwen3-VL-8B-Instruct', local_dir='../models/Qwen3-VL-8B-Instruct')
```

---

### 5. SFT 数据集生成

**文件**: `src/fine-tuning_(training)/gen_SFT_data.py`

#### `gen_sft_data()`

```python
def gen_sft_data(
    csv_path: str,
    image_dir: str,
    output_jsonl: str,
    augment_factor: float = 0.0,
)
```

从 CSV 标注和图像目录生成 Qwen3-VL 专用的 SFT JSONL 数据集。每条记录包含 `image_path`、`instruction`（中文提示，要求忽略文字专注于石头视觉特征）和 `output`（`是`/`否` + 理由）。如果 `augment_factor > 0`，会调用 `ImageAugmentor` 生成增强图像并一并写入。

---

### 6. 模型测试与提交

**文件**: `src/testing/test_model.py`

#### `load_model_and_quick_test()`

```python
def load_model_and_quick_test(
    lora_model_name: str = "qwen3-vl-lora-meteorite/final_adapter"
)
```

加载 LoRA 模型并对前 10 张测试图片进行快速推理测试，打印模型回答。

#### `load_model()`

```python
def load_model(
    lora_model_name: str = "..."
) -> Tuple[PeftModel, AutoProcessor, str]
```

加载基座模型 + LoRA adapter + processor，返回 `(model, processor, device)`。使用 4-bit 量化 (`BitsAndBytesConfig`)。

#### `predict()`

```python
def predict(
    model,
    processor,
    test_loader: DataLoader,
    device: str
) -> Dict[str, int]
```

批量预测测试集。返回 `{filename: label}` 字典，其中 label 通过规则判断（回答中包含"是"则判为 1，否则为 0）。使用给定的先验概率提示词。

**文件**: `src/testing/gen_submission_files.py`

#### `class TestImageDataset(Dataset)`

```python
class TestImageDataset(Dataset):
    def __init__(self, image_dir: str)
    def __getitem__(self, idx) -> Tuple[str, Image.Image]
    def __len__(self) -> int
```

自定义测试集，返回 `(filename, PIL.Image)` 对。

#### `collate_fn()`

```python
def collate_fn(batch) -> List
```

直接返回 batch 列表，不做额外处理。

#### `make_submission()`

```python
def make_submission(
    id_to_pred: Dict[str, int],
    template_csv_path: str,
    output_path: str,
)
```

根据预测结果字典和 `sample_submission.csv` 模板生成最终 `submission.csv`。包含缺失检测。

---

### 7. 训练集/验证集评估

**文件**: `src/testing/eval_train_val.py`

#### `load_model()`

```python
def load_model(checkpoint_path: str) -> Tuple[PeftModel, AutoProcessor, str]
```

加载基座模型 + LoRA adapter（4-bit 量化），返回 `(model, processor, device)`。

#### `split_dataset()`

```python
def split_dataset(
    dataset: Dataset,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[Dataset, Dataset]
```

按指定比例划分训练集和验证集（与训练脚本相同的划分方式）。

#### `extract_label()`

```python
def extract_label(output: str) -> int
```

从模型文本输出中提取标签：包含"是"不包含"否" → 1，包含"否" → 0，无法解析 → -1。

#### `evaluate_on_dataset()`

```python
def evaluate_on_dataset(
    model,
    processor,
    device,
    dataset,
    dataset_name: str,
    batch_size: int = 4,
) -> Optional[Dict]
```

在指定数据集上评估模型，返回包含 `accuracy`, `f1`, `precision`, `recall`, `confusion_matrix` 的字典。

#### `main()`

```python
def main(checkpoint_path: str = DEFAULT_CHECKPOINT)
```

主评估流程：加载模型 → 划分数据集 → 评估训练集子集（500 样本）→ 评估验证集 → 对比输出 → 保存 JSON 日志。

---

### 8. 可视化与辅助工具

#### `test_gd_sam2_visualization.py` — GroundingDINO（经测试效果差，后续仅保留sam2） + SAM2 可视化

**文件**: `src/testing/test_gd_sam2_visualization.py`

分别测试和可视化 GroundingDINO 目标检测和 SAM2.1 图像分割的效果。

```python
def load_grounding_dino() -> Tuple[AutoModel, AutoProcessor, str]
def load_sam2() -> Tuple[SAM2ImagePredictor, str]

def detect_with_grounding_dino(
    model, processor, device,
    image_pil: Image.Image,
    prompts: List[str] = [...]
) -> dict

def segment_with_sam2(
    predictor,
    image_np: np.ndarray,
    bbox: Optional[List[int]] = None,
    point: Optional[Tuple[int, int]] = None,
) -> dict

def visualize_detection(image_pil, detection_results, output_path)
def visualize_segmentation(image_pil, segmentation_result, output_path, title)
def visualize_combined(image_pil, detection_results, segmentation_result, output_path)
def get_mask_bbox(mask: np.ndarray) -> Tuple[int, int, int, int]
def test_single_image(image_path, gd_model, gd_processor, gd_device, sam2_predictor, prompts)
def main()
```

输出文件到 `data/visualization_results/`，包含检测框可视化、分割掩码、组合结果。

#### `ploting.py` — Loss 曲线绘制

**文件**: `src/ploting.py`

```python
def parse_loss_file(loss_file_path: str) -> Tuple[List, List]
# 解析 loss.txt，返回 [(epoch, loss), ...] 的训练和验证数据

def plot_loss_curves(train_data, eval_data, output_path: str)
# 绘制并保存 Training/Validation Loss 曲线
```

#### `test_gen_labels.py` — 视觉大模型 API 推理

**文件**: `src/data_processing_modules/test_gen_labels.py`

调用视觉语言模型 API（兼容 OpenAI 格式，默认使用 qwen3.5-plus）对训练集图片进行推理，生成详细的判断理由。

```python
def image_to_base64(image_path: str) -> str
# 将图片转为 base64 编码

def call_vl_api(
    image_path: str,
    api_key: str,
    api_base: str,
    model: str,
    prompt: Optional[str] = None,
    known_label: Optional[int] = None,
) -> Dict[str, Any]
# 调用视觉语言模型 API，返回推理结果

def get_labeled_images(csv_path: str, max_per_class: Optional[int] = None) -> List[Dict]
# 从 CSV 获取带标签的图片路径
```

#### 数据路径更新工具

**`update_train_data_path.py`**: 将 JSONL 中的 `train_images/` 路径替换为 `train_images_extracted/`，检查文件存在性。

**`update_train_data_augmented.py`**: 在 extracted JSONL 基础上追加增强图片的数据条目。

---

### 9. Shell 脚本

#### `run_train.sh`

启动 TensorBoard（端口 6007）并执行 `train_lora.py`。

#### `run_tensorBoard.sh`

仅启动 TensorBoard（端口 6007），日志目录 `data/logs/tensorboard`。

---

## 使用流程

### 快速开始

1. **下载基座模型**
   ```bash
   python src/fine-tuning_(training)/downloading_model.py
   ```

2. **提取石头主体**（SAM2）
   ```bash
   python src/data_processing_modules/StoneExtractor.py
   ```

3. **数据增强**（可选）
   ```bash
   python src/data_processing_modules/ImageAugmentor.py
   ```

4. **生成 SFT 训练数据**
   ```bash
   python src/fine-tuning_(training)/gen_SFT_data.py
   # 更新图片路径
   python src/data_processing_modules/update_train_data_path.py
   python src/data_processing_modules/update_train_data_augmented.py  # 如果做了增强
   ```

5. **启动训练**
   ```bash
   bash run_train.sh
   # 或直接 python src/fine-tuning_(training)/train_lora.py
   ```

6. **评估模型**
   ```bash
   python src/testing/eval_train_val.py --checkpoint <checkpoint_path>
   ```

7. **推理并生成提交文件**
   ```bash
   python src/testing/gen_submission_files.py
   ```

8. **可视化**
   ```bash
   python src/testing/test_gd_sam2_visualization.py  # GD+SAM2 可视化
   python src/ploting.py                              # Loss 曲线
   ```

---

## 关键技术参数

| 项目 | 值 |
|------|-----|
| 基座模型 | Qwen3-VL-8B-Instruct |
| 量化方式 | QLoRA 4-bit (NF4, double quant, bfloat16 compute) |
| LoRA rank | 64 |
| LoRA alpha | 32 |
| LoRA dropout | 0.1 |
| 目标模块 | q_proj, v_proj, k_proj, o_proj, gate_proj, up_proj, down_proj |
| 学习率 | 1e-4 |
| 调度器 | Cosine + 10% warmup |
| 权重衰减 | 0.05 |
| 训练 batch size | 2（gradient accumulation × 4 = effective 8） |
| 训练轮次 | 3 epochs |
| 验证集比例 | 10% |
| 图像最大尺寸 | 720×720 (thumbnail) |
| SAM2 模型 | sam2.1_hiera_large |
| 分割方法 | 中心点 prompt → 掩码边界裁剪 → 灰色背景填充 |
| 数据增强 | 随机旋转 ±30° + 随机缩放 0.8x~1.2x |

---

## 数据集说明

二分类任务：
- **label 0**: 非陨石（普通石头或地球岩石）
- **label 1**: 陨石

训练提示词（instruction）要求模型忽略图片中的文字、标签、标尺等信息，仅根据石头的视觉特征（颜色、纹理、形状、熔壳、气印等）进行判断。

目前训练采用的数据集为 `data/qwen3_sft_train_resampled_without_extracted.jsonl`，这是一个回答仅有”是”或”否”的伪大模型标注数据。该数据集过采样了非陨石5:1，以达到验证集 4:1 的非陨石:陨石比。该数据集的图片没有采用石头主体提取。该数据集采用了数据增强。验证集 F1-score=0.812。

### JSONL 数据集清单

所有 SFT 训练数据统一采用 JSON Lines 格式，每条记录包含三个字段：

```json
{“image_path”: “...”, “instruction”: “...”, “output”: “...”}
```

| 数据集 | 记录数 | 非陨石 (0) | 陨石 (1) | 比例 (0:1) | 图片来源 | instruction 风格 | 备注 |
|--------|--------|-----------|---------|-----------|---------|-----------------|------|
| `qwen2_train.jsonl` | 0 | 0 | 0 | — | — | — | 空文件，早期 Qwen2 实验遗留 |
| `qwen3_sft_train.jsonl` | 15,294 | 7,239 (47.3%) | 8,055 (52.7%) | 0.90:1 | `train_images/`（原始图片） | 详细描述型（”请观察纹理特征…首先给出'是'或'否'，然后简要说明理由”） | `gen_SFT_data.py` 直接生成，含 2x 数据增强。output 包含详细理由 |
| `qwen3_sft_train_simple.jsonl` | 15,156 | 7,121 (47.0%) | 8,035 (53.0%) | 0.89:1 | `train_images_extracted/`（SAM2 提取后） | 简单型（”请观察这张图片，判断是否为陨石？回答'是'或'否'”） | output 仅”是”或”否”，无详细理由。用于简化训练 |
| `qwen3_sft_train_extracted.jsonl` | 5,098 | 2,413 (47.3%) | 2,685 (52.7%) | 0.90:1 | `train_images_extracted/`（SAM2 提取后） | 详细描述型（同 `qwen3_sft_train.jsonl`） | 仅含原始图片（无增强），每条对应一张 SAM2 提取后的图片 |
| `qwen3_sft_train_extracted_augmented.jsonl` | 15,294 | 7,239 (47.3%) | 8,055 (52.7%) | 0.90:1 | `train_images_extracted/` + `train_images_extracted_augmented/` | 详细描述型（同 `qwen3_sft_train.jsonl`） | 由 `update_train_data_augmented.py` 生成：在 extracted 基础上追加 2x 增强图片数据 |
| `qwen3_sft_train_resampled.jsonl` | 8,901 | 7,121 (80.0%) | 1,780 (20.0%) | 4.00:1 | `train_images_extracted/` (3,001) + `train_images_extracted_augmented/` (5,900) | 简单型（同 `qwen3_sft_train_simple.jsonl`） | 过采样非陨石至 4:1 比例，使用 SAM2 提取图片+增强。用于缓解模型偏向陨石的类别不平衡问题 |
| `qwen3_sft_train_resampled_without_extracted.jsonl` | 8,562 | 7,135 (83.3%) | 1,427 (16.7%) | 5.00:1 | `train_images/` (2,904) + `train_images_augmented/` (5,658) | 简单型（同 `qwen3_sft_train_simple.jsonl`） | **当前训练用数据集**。过采样非陨石至 5:1 比例，不使用 SAM2 提取，使用原始图片+增强。验证集 F1=0.812 |
| `vl_model_reasoning.jsonl` | 10 | 5 (50%) | 5 (50%) | 1:1 | `train_images/`（原始图片） | — | 视觉大模型（qwen3.5-plus API）推理结果，每条包含 `has_meteorite`、`confidence`、`reasons`、`features_observed` 等结构化字段。用于分析大模型判断依据或知识蒸馏 |

### 数据集演化路径

```
qwen3_sft_train.jsonl              ← 初始版本：原始图片 + 详细描述 + 增强（15,294 条，~1:1）
    │
    ├── update_train_data_path.py ──→ qwen3_sft_train_extracted.jsonl        ← SAM2 提取后，仅原始图（5,098 条，~1:1）
    │                                       │
    │                                       ├── update_train_data_augmented.py ──→ qwen3_sft_train_extracted_augmented.jsonl ← 追加增强（15,294 条，~1:1）
    │                                       │
    │                                       └── 简化 instruction/output ──→ qwen3_sft_train_simple.jsonl ← 提取+简单标注+增强（15,156 条，~1:1）
    │                                                                                   │
    │                                                                                   └── 过采样非陨石 ──→ qwen3_sft_train_resampled.jsonl ← 提取+简单+过采样4:1（8,901 条）
    │
    └── 不用 SAM2 提取，直接简单标注 ──→ qwen3_sft_train_resampled_without_extracted.jsonl ← 原始+简单+过采样5:1（8,562 条，当前训练用）
```

### instruction 风格说明

项目中使用过两种 instruction 模板：

**详细描述型**（用于 `qwen3_sft_train.jsonl` 系列）：
> “请观察这张图片中石头的视觉特征（颜色、纹理、形状），判断其中是否有陨石？注意：忽略图片中的所有文字、标签、标尺等信息，仅根据石头本身的外观特征判断。回答时首先给出'是'或'否'，然后简要说明理由。”

对应的 output 包含详细理由，例如：
> “是。图中石头展示了陨石特有的视觉特征：表面有深色熔壳或气印（类似手指按压的凹坑），颜色呈深褐色或金属灰色……判断为陨石。”

**简单型**（用于 `qwen3_sft_train_simple.jsonl` 及后续 resampled 系列）：
> “请观察这张图片，判断是否为陨石？回答'是'或'否'。”

对应的 output 仅包含”是”或”否”，无详细理由。实验表明简单标注在分类任务上效果更好。

---

## 未来执行方案

基于中期报告识别出的核心问题——**训练-测试域偏移严重**、**标注信息过于简陋**、**主体提取策略尚未最优**——制定以下四个方向的执行方案。

### 方向一：增强标注质量，引导模型学习正确判据

**问题**：当前 SFT 数据仅包含”是/否”回答，缺乏判断理由，模型依赖预训练中的通用视觉逻辑，而非系统性地学习陨石判别规则（熔壳、气印、磁性等特征）。

**执行步骤**：

1. **生成高质量判别理由**
   - 利用强视觉语言模型 API（qwen3.5-plus / GPT-4o）为已有标签的训练样本生成详细判别理由
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

**执行步骤**：

1. **模拟测试集降质的数据增强**
   - 新增增强维度到 `ImageAugmentor` 或独立模块：
     - **降分辨率**：随机缩小图像至 256x256 再放大回原尺寸（模拟低分辨率）
     - **降低亮度**：随机调整亮度因子 0.5~0.9（模拟暗色图像）
     - **降低饱和度**：随机调整饱和度因子 0.3~0.7（模拟单调色彩）
     - **高斯模糊**：σ = 0.5~2.0（模拟运动模糊/失焦）
   - 注意：避免大角度旋转和仿射变换破坏陨石关键鉴别特征（熔壳、气印）

2. **多尺度训练**
   - 修改 `train_lora.py` 中 `QwenVLDataCollator` 的 `img.thumbnail()` 调用
   - 随机选择输入尺寸：`[384, 512, 720]`，使模型适应不同分辨率下的纹理表达
   - 不同尺寸共享同一模型权重，仅在训练时随机切换

3. **测试时增强（TTA）**
   - 在 `test_model.py` 的 `predict()` 函数中实现：
     - 水平翻转
     - 垂直翻转
     - 多尺度裁剪（如 720、512、384 三种尺寸分别推理）
   - 对各变换版本的 softmax 概率取平均，得到最终预测
   - 预期稳定预测结果，降低单次推理的方差

4. **验证方案**
   - 单独验证每种增强策略对 Kaggle F1 的贡献（消融实验）
   - 组合最优策略进行最终训练

### 方向三：优化目标区域提取与特征聚焦

**问题**：SAM2.1 基于中心点提示的分割可能保留无关区域或切割掉边缘判别特征；rembg 等显著性检测模型在陨石场景下精度有限。

**执行步骤**：

1. **更细粒度的分割方案**
   - 尝试 GroundingDINO 生成候选框 → SAM2 以框为 prompt 进行分割（已探索，GD 框选效果差，需换用 YOLOv8-obb 检测陨石边界框）
   - 裁剪后保留一定背景上下文（padding 从 10px 扩大至 30-50px），避免丢失边缘判别特征

2. **软聚焦替代硬分割**
   - 在不进行硬分割的前提下，引入注意力模块：
     - **CBAM**（Convolutional Block Attention Module）：通道 + 空间注意力
     - **SE**（Squeeze-and-Excitation）：通道注意力
   - 或引入可学习的提示向量（learnable prompt），使模型自行学会忽略背景、聚焦陨石主体

3. **验证对比**
   - 对比三种方案：无提取（当前最优）vs SAM2 中心点提取 vs YOLOv8-obb + SAM2 框提取
   - 指标：验证集 loss/F1 + Kaggle 测试集 F1

### 方向四：模型集成

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

