import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from dataset import MeteoriteDataset, get_transform
from model import get_model
import argparse
import os

def predict_with_tta(model, loader, device, tta_transforms, num_tta=4):
    """使用测试时增强，返回每个样本的平均概率"""
    model.eval()
    all_probs = []
    all_ids = []
    for images, img_names in loader:
        batch_probs = 0.0
        for t in range(num_tta):
            # 对每个图像应用一种随机增强（例如水平翻转）
            aug_images = images.clone()
            if tta_transforms.get('flip', False) and np.random.rand() > 0.5:
                aug_images = torch.flip(aug_images, dims=[3])
            if tta_transforms.get('rotate', False):
                # 简单旋转90度示例，需要更多变换可扩展
                k = np.random.choice([0, 1, 2, 3])
                aug_images = torch.rot90(aug_images, k, dims=[2, 3])
            aug_images = aug_images.to(device)
            with torch.no_grad():
                outputs = model(aug_images)
                probs = torch.softmax(outputs, dim=1)[:, 1]
            batch_probs += probs.cpu()
        batch_probs /= num_tta
        all_probs.append(batch_probs)
        all_ids.extend(img_names)
    return np.concatenate(all_probs), all_ids

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载最佳阈值
    if os.path.exists(args.thresh_file):
        with open(args.thresh_file, 'r') as f:
            best_thresh = float(f.read().strip())
        print(f"Loaded threshold: {best_thresh}")
    else:
        best_thresh = 0.5
        print("Threshold file not found, using default 0.5")

    # 测试集 transform（与验证时一致）
    test_dataset = MeteoriteDataset(
        img_dir=args.test_img_dir,
        label_file=None,
        transform=get_transform(input_size=args.input_size, train=False)
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers)

    # 加载模型
    model = get_model(model_name=args.model, num_classes=2, pretrained=False).to(device)
    checkpoint = torch.load(args.model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    # 测试时增强设置（可配置）
    tta_transforms = {'flip': True, 'rotate': True}  # 根据需要开启
    probs, ids = predict_with_tta(model, test_loader, device, tta_transforms, num_tta=args.num_tta)

    # 根据阈值生成标签
    preds = (probs > best_thresh).astype(int)

    # 生成提交文件
    submission = pd.DataFrame({'id': ids, 'label': preds})
    submission.to_csv(args.output_csv, index=False, quoting=1)  # quoting=1 对应 csv.QUOTE_NONNUMERIC
    print(f"Submission saved to {args.output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_img_dir', type=str, default='./test_images/test_images')
    parser.add_argument('--model_path', type=str, default='/home/dataset/best_resnet50.pth')
    parser.add_argument('--model', type=str, default='resnet50') 
    parser.add_argument('--input_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--output_csv', type=str, default='./test_output/submission.csv')
    parser.add_argument('--thresh_file', type=str, default='/home/dataset/best_thresh.txt')
    parser.add_argument('--num_tta', type=int, default=4, help='Number of TTA augmentations')
    args = parser.parse_args()
    main(args)