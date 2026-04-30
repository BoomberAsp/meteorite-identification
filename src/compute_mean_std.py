import torch
from torch.utils.data import DataLoader
from dataset import MeteoriteDataset
from torchvision import transforms
import argparse

def compute_mean_std(dataset, batch_size=1):
    """逐个图像计算通道均值和标准差（避免尺寸不一致问题）"""
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)  # batch_size=1 避免尺寸冲突
    
    channels_sum = torch.zeros(3)
    channels_squared_sum = torch.zeros(3)
    total_pixels = 0
    
    for images, _ in loader:   # images shape: [1, 3, H, W]
        # 计算当前图像的像素总数（H*W）
        num_pixels = images.shape[2] * images.shape[3]
        # 当前图像的通道均值 (shape: [3])
        channel_mean = images.mean(dim=[0,2,3])   # 注意：images 有 batch 维度，实际形状 [1,3,H,W]
        # 当前图像的通道平方均值
        channel_squared_mean = (images ** 2).mean(dim=[0,2,3])
        
        channels_sum += channel_mean * num_pixels
        channels_squared_sum += channel_squared_mean * num_pixels
        total_pixels += num_pixels
    
    # 整体均值 = 所有像素值之和 / 总像素数
    mean = channels_sum / total_pixels
    # 整体方差 = E(x^2) - [E(x)]^2
    var = channels_squared_sum / total_pixels - mean ** 2
    std = torch.sqrt(var)
    return mean, std

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', type=str, default='./train_images/train_images')
    parser.add_argument('--label_file', type=str, default='./train_labels.csv')
    args = parser.parse_args()
    
    # 注意：这里 transform 只做 ToTensor，不进行归一化或缩放，保留原始尺寸
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = MeteoriteDataset(img_dir=args.img_dir, label_file=args.label_file, transform=transform)
    
    print("Computing mean and std...")
    mean, std = compute_mean_std(dataset)
    print(f"mean = {mean.tolist()}")
    print(f"std = {std.tolist()}")