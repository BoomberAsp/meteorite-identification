import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import f1_score
import numpy as np
from tqdm import tqdm
import copy
import matplotlib.pyplot as plt

from dataset import MeteoriteDataset, get_transform
from model import get_model
from utils import mixup_data, mixup_criterion, FocalLoss, label_smoothing_loss

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def train_one_epoch(model, loader, criterion, optimizer, device, mixup_alpha=0.2, label_smoothing=0.0):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for images, labels in tqdm(loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)

        if mixup_alpha > 0:
            images, labels_a, labels_b, lam = mixup_data(images, labels, mixup_alpha)
            outputs = model(images)
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
        else:
            outputs = model(images)
            if label_smoothing > 0:
                loss = label_smoothing_loss(outputs, labels, smoothing=label_smoothing)
            else:
                loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    if mixup_alpha == 0:
        epoch_f1 = f1_score(all_labels, all_preds, average='binary')
    else:
        epoch_f1 = None
    return epoch_loss, epoch_f1

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_f1_default = f1_score(all_labels, all_preds, average='binary')
    best_thresh = 0.5
    best_f1 = epoch_f1_default
    for thresh in np.arange(0.3, 0.8, 0.02):
        pred_thresh = (np.array(all_probs) > thresh).astype(int)
        f1_thresh = f1_score(all_labels, pred_thresh, average='binary')
        if f1_thresh > best_f1:
            best_f1 = f1_thresh
            best_thresh = thresh
    return epoch_loss, epoch_f1_default, best_thresh, best_f1

def save_checkpoint(epoch, model, optimizer, val_f1, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_f1': val_f1,
    }, path)

def plot_curves(train_losses, val_losses, train_f1s, val_f1s, save_path):
    epochs = range(1, len(train_losses)+1)
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curves')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    valid_train_f1s = [f1 if f1 is not None else np.nan for f1 in train_f1s]
    plt.plot(epochs, valid_train_f1s, 'b--', label='Training F1 (if available)', marker='o')
    plt.plot(epochs, val_f1s, 'r-', label='Validation F1 (optimal threshold)', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('F1 Score Curves')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Training curves saved to {save_path}")

def main(args):
    # ---------- 统一输出目录 ----------
    if args.output_dir is None:
        # 如果未指定，默认生成一个基于模型名和时间戳的目录
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"output_{args.model}_{timestamp}"
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"All outputs will be saved under: {args.output_dir}")

    # 重新组织所有输出路径，使其位于 output_dir 下
    # 最佳模型文件
    if args.save_path is None:
        args.save_path = os.path.join(args.output_dir, "best_model.pth")
    else:
        # 如果用户显式指定路径，也强制放到 output_dir 中（避免散落）
        args.save_path = os.path.join(args.output_dir, os.path.basename(args.save_path))
    # 阈值文件
    if args.thresh_file is None:
        args.thresh_file = os.path.join(args.output_dir, "best_thresh.txt")
    else:
        args.thresh_file = os.path.join(args.output_dir, os.path.basename(args.thresh_file))
    # epoch 检查点目录
    if args.save_all_epochs:
        if args.ckpt_dir is None:
            args.ckpt_dir = os.path.join(args.output_dir, "checkpoints")
        else:
            args.ckpt_dir = os.path.join(args.output_dir, os.path.basename(args.ckpt_dir))
        os.makedirs(args.ckpt_dir, exist_ok=True)
    # 曲线图目录
    if args.plot_dir is None:
        args.plot_dir = os.path.join(args.output_dir, "plots")
    else:
        args.plot_dir = os.path.join(args.output_dir, os.path.basename(args.plot_dir))
    os.makedirs(args.plot_dir, exist_ok=True)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 确定输入尺寸
    input_size = args.input_size
    if args.model.startswith('efficientnet'):
        if 'b0' in args.model:
            input_size = 224
        elif 'b1' in args.model:
            input_size = 240
        elif 'b2' in args.model:
            input_size = 260
        elif 'b3' in args.model:
            input_size = 300
        elif 'b4' in args.model:
            input_size = 380
    if args.model == 'resnet50':
        input_size = 224


    # 数据集
    full_dataset = MeteoriteDataset(
        img_dir=args.train_img_dir,
        label_file=args.train_label_csv,
        transform=None
    )
    val_size = int(len(full_dataset) * args.val_ratio)
    train_size = len(full_dataset) - val_size
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size],
                                            generator=torch.Generator().manual_seed(args.seed))

    train_transform = get_transform(input_size=input_size, train=True)
    val_transform = get_transform(input_size=input_size, train=False)
    train_subset.dataset.transform = train_transform
    val_subset.dataset.transform = val_transform

    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    # 模型
    model = get_model(model_name=args.model, num_classes=2, pretrained=True, weights_path=args.weights_path).to(device)

    # 分层学习率
    if args.different_lr:
        if hasattr(model, 'fc'):
            classifier_params = model.fc.parameters()
            base_params = [p for n, p in model.named_parameters() if 'fc' not in n]
        elif hasattr(model, 'classifier'):
            classifier_params = model.classifier.parameters()
            base_params = [p for n, p in model.named_parameters() if 'classifier' not in n]
        else:
            base_params = model.parameters()
            classifier_params = []
        optimizer = optim.Adam([
            {'params': base_params, 'lr': args.lr * 0.1},
            {'params': classifier_params, 'lr': args.lr}
        ])
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 损失函数
    if args.loss == 'cross_entropy':
        criterion = nn.CrossEntropyLoss()
    elif args.loss == 'focal':
        criterion = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
    else:
        raise ValueError(f"Unknown loss: {args.loss}")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 记录指标
    train_losses = []
    val_losses = []
    train_f1s = []
    val_f1s = []
    val_default_f1s = []

    best_f1 = 0.0
    best_thresh = 0.5
    best_epoch = 0
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(1, args.epochs+1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss, train_f1 = train_one_epoch(model, train_loader, criterion, optimizer, device,
                                               mixup_alpha=args.mixup_alpha,
                                               label_smoothing=args.label_smoothing)
        val_loss, val_f1_default, thresh, val_f1_best = validate(model, val_loader, criterion, device)

        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_f1s.append(train_f1)
        val_f1s.append(val_f1_best)
        val_default_f1s.append(val_f1_default)

        print(f"Train Loss: {train_loss:.4f}, Train F1: {train_f1 if train_f1 is not None else 'N/A'}")
        print(f"Val Loss: {val_loss:.4f}, Val F1 (default thresh): {val_f1_default:.4f}")
        print(f"Optimal threshold: {thresh:.4f}, Val F1 (optimal): {val_f1_best:.4f}")

        if args.save_all_epochs:
            ckpt_path = os.path.join(args.ckpt_dir, f"epoch_{epoch:03d}_val_f1_{val_f1_best:.4f}.pth")
            save_checkpoint(epoch, model, optimizer, val_f1_best, ckpt_path)
            print(f"Checkpoint saved to {ckpt_path}")

        if val_f1_best > best_f1:
            best_f1 = val_f1_best
            best_thresh = thresh
            best_epoch = epoch
            best_model_wts = copy.deepcopy(model.state_dict())
            print(f"Best model updated (epoch {epoch}, F1={best_f1:.4f})")

    # 保存最终最佳模型和阈值
    torch.save(best_model_wts, args.save_path)
    with open(args.thresh_file, 'w') as f:
        f.write(str(best_thresh))
    print(f"\nBest model saved to {args.save_path} (epoch {best_epoch}, F1={best_f1:.4f})")
    print(f"Optimal threshold saved to {args.thresh_file}")

    # 绘制曲线
    plot_path = os.path.join(args.plot_dir, "training_curves.png")
    plot_curves(train_losses, val_losses, train_f1s, val_f1s, plot_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 新增统一输出目录参数
    parser.add_argument('--output_dir', type=str, default="resnet50_lr5_dlr",
                        help='Root directory to save all outputs (models, thresholds, checkpoints, plots). '
                             'If not provided, a timestamped folder will be created automatically.')

    # 原有数据参数
    parser.add_argument('--train_img_dir', type=str, default='./train_images/train_images')
    parser.add_argument('--train_label_csv', type=str, default='./train_labels.csv')
    parser.add_argument('--val_ratio', type=float, default=0.2)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--input_size', type=int, default=224)
    parser.add_argument('--different_lr', type=bool, default=True)
    parser.add_argument('--loss', type=str, default='cross_entropy', choices=['cross_entropy', 'focal'])
    parser.add_argument('--focal_alpha', type=float, default=0.25)
    parser.add_argument('--focal_gamma', type=float, default=2.0)
    parser.add_argument('--mixup_alpha', type=float, default=0)
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--weights_path', type=str, default=None,
                        help='Path to local pretrained weights (if using timm model)')

    # 输出文件相关参数（一般都由 output_dir 自动派生，用户可选择性覆盖）
    parser.add_argument('--save_path', type=str, default=None,
                        help='Path to save best model. If None, will be placed under output_dir.')
    parser.add_argument('--thresh_file', type=str, default=None,
                        help='Path to save optimal threshold. If None, will be placed under output_dir.')
    parser.add_argument('--save_all_epochs', type=bool, default=True,
                        help='Save checkpoint for every epoch')
    parser.add_argument('--ckpt_dir', type=str, default=None,
                        help='Directory to save epoch checkpoints. If None, will be placed under output_dir.')
    parser.add_argument('--plot_dir', type=str, default=None,
                        help='Directory to save training curves plot. If None, will be placed under output_dir.')

    args = parser.parse_args()
    main(args)