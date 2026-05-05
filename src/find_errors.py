import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np
from PIL import Image
import shutil
from dataset import MeteoriteDataset, get_transform
from model import get_model
import argparse

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. 重新创建完整数据集（不应用任何 transform，便于后续按验证集索引取图）
    full_dataset = MeteoriteDataset(
        img_dir=args.train_img_dir,
        label_file=args.train_label_csv,
        transform=None          # 暂不设置，验证时单独设定
    )

    # 2. 与训练时完全一致的验证集划分
    val_size = int(len(full_dataset) * args.val_ratio)
    train_size = len(full_dataset) - val_size
    train_subset, val_subset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )

    # 3. 为验证子集设置验证变换（不增强，仅 resize + normalize）
    val_transform = get_transform(input_size=args.input_size, train=False)
    # 注意：val_subset 的 dataset 是 full_dataset，需要将其 transform 改为验证变换
    # 但要小心，full_dataset 同时也被 train_subset 引用，这里我们只关心验证集，可以临时修改
    # 因为 train_subset 在本脚本中不会使用，所以直接修改 full_dataset.transform 是安全的
    full_dataset.transform = val_transform

    val_loader = DataLoader(
        val_subset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # 4. 获取验证集中每个样本的文件名和真实标签
    # val_subset.indices 是原始 full_dataset 中的索引列表，顺序与 DataLoader 输出的顺序一致（shuffle=False）
    val_indices = val_subset.indices
    val_filenames = [full_dataset.ids[i] for i in val_indices]
    val_labels = [full_dataset.labels[i] for i in val_indices]

    # 5. 加载训练好的模型
    model = get_model(model_name=args.model, num_classes=2, pretrained=True).to(device)
    checkpoint = torch.load(args.model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    # 6. 加载最佳分类阈值
    if os.path.exists(args.thresh_file):
        with open(args.thresh_file, 'r') as f:
            best_thresh = float(f.read().strip())
        print(f"Loaded threshold: {best_thresh}")
    else:
        best_thresh = 0.5
        print("Threshold file not found, using default 0.5")

    # 7. 遍历验证集进行预测
    all_probs = []
    all_preds = []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1]   # 正类概率
            preds = (probs > best_thresh).int()
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # 8. 找出错误样本
    error_indices = [i for i, (true, pred) in enumerate(zip(val_labels, all_preds)) if true != pred]
    print(f"Total validation samples: {len(val_labels)}")
    print(f"Error samples: {len(error_indices)} ({(len(error_indices)/len(val_labels))*100:.2f}%)")

    # 9. 输出错误样本详情，并保存错误图像
    if args.save_error_images:
        error_dir = os.path.join(args.error_dir, "error_images")
        os.makedirs(error_dir, exist_ok=True)

    for idx in error_indices:
        filename = val_filenames[idx]
        true_label = val_labels[idx]
        pred_label = all_preds[idx]
        prob = all_probs[idx]
        print(f"Error: {filename} | True: {true_label} | Pred: {pred_label} | Prob: {prob:.4f}")

        if args.save_error_images:
            # 原始图像路径
            src_path = os.path.join(args.train_img_dir, filename)
            if os.path.exists(src_path):
                dst_path = os.path.join(error_dir, f"{filename}_{true_label}_{pred_label}_{prob:.3f}.jpg")
                shutil.copy(src_path, dst_path)
    
    # 额外统计：混淆矩阵
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(val_labels, all_preds, labels=[0,1])
    print("\nConfusion Matrix:")
    print("         Pred 0    Pred 1")
    print(f"True 0:   {cm[0,0]}        {cm[0,1]}")
    print(f"True 1:   {cm[1,0]}        {cm[1,1]}")
    print(f"False Positives: {cm[0,1]}, False Negatives: {cm[1,0]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_img_dir', type=str, default='./train_images/train_images')
    parser.add_argument('--train_label_csv', type=str, default='./train_labels.csv')
    parser.add_argument('--val_ratio', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--input_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--model_path', type=str, default='/home/dataset/resnet50_output_nomixup/best_model.pth')
    parser.add_argument('--thresh_file', type=str, default='/home/dataset/resnet50_output_nomixup/best_thresh.txt', help='Path to best threshold file')
    parser.add_argument('--save_error_images', default=False, help='Save error images to disk')
    parser.add_argument('--error_dir', type=str, default='./error_analysis_test2', help='Directory to save error images')
    args = parser.parse_args()
    main(args)