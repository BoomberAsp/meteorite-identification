import torch
import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import test_model

# 自定义测试数据集，返回 (filename, PIL.Image)
class TestImageDataset(Dataset):
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.image_files = []
        for f in sorted(os.listdir(image_dir)):
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                self.image_files.append(f)
        if not self.image_files:
            raise ValueError(f"No image files found in {image_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        filename = self.image_files[idx]
        path = os.path.join(self.image_dir, filename)
        image = Image.open(path).convert("RGB")
        return filename, image

# 简单的 collate_fn，直接返回 batch 列表（test_model.predict 会自己处理）
def collate_fn(batch):
    return batch

def make_submission(id_to_pred, template_csv_path, output_path):
    """生成 submission.csv 文件"""
    ids_df = pd.read_csv(template_csv_path)
    if "id" not in ids_df.columns:
        raise ValueError(f"{template_csv_path} must contain 'id' column")
    submission_df = ids_df.copy()
    submission_df["label"] = submission_df["id"].map(id_to_pred)
    if submission_df["label"].isna().any():
        missing_ids = submission_df.loc[submission_df["label"].isna(), "id"].head(5).tolist()
        raise RuntimeError(f"Missing predictions for some ids, examples: {missing_ids}")
    submission_df["label"] = submission_df["label"].astype(int)
    submission_df.to_csv(output_path, index=False)

if __name__ == "__main__":
    # 加载模型和处理器
    model, processor, device = test_model.load_model(lora_model_name="qwen3-vl-lora-meteorite_20260507_005818/final_adapter_20260507_005818")

    # 准备测试数据加载器
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_ROOT = os.path.join(BASE_DIR, "../../data")
    # TEST_DATA_DIR = os.path.join(DATA_ROOT, "test_images_extracted")
    TEST_DATA_DIR = os.path.join(DATA_ROOT, "test_images")
    dataset = TestImageDataset(TEST_DATA_DIR)
    batch_size = 4   # 根据显存调整，建议 2 或 4
    test_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn
    )

    # 进行预测
    id_to_pred = test_model.predict(model, processor, test_loader, device)

    # 生成提交文件
    OUTPUT_DIR = os.path.join(DATA_ROOT, "logs")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    submission_path = os.path.join(OUTPUT_DIR, "submission_05271719_biased_training.csv")
    template_path = os.path.join(DATA_ROOT, "sample_submission.csv")
    make_submission(id_to_pred, template_path, submission_path)
    print(f"Kaggle submission file saved to {submission_path}")