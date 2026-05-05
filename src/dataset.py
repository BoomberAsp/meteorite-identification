import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class MeteoriteDataset(Dataset):
    def __init__(self, img_dir, label_file=None, transform=None, train=True):
        self.img_dir = img_dir
        self.transform = transform
        self.train = train

        if label_file is not None:
            self.df = pd.read_csv(label_file)
            self.ids = self.df['id'].values
            self.labels = self.df['label'].values
        else:
            self.ids = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            self.labels = None

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_name = self.ids[idx]
        img_path = os.path.join(self.img_dir, img_name)
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            image = Image.new('RGB', (300, 300))

        if self.transform:
            image = self.transform(image)

        if self.labels is not None:
            return image, self.labels[idx]
        else:
            return image, img_name

def get_transform(input_size=224, train=True):
    """返回数据增强流水线，input_size 可根据模型调整"""
    if train:
        return transforms.Compose([
            transforms.Resize((int(input_size*1.1), int(input_size*1.1))),
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            # transforms.Normalize(mean=[0.563, 0.546, 0.524], std=[0.301, 0.303, 0.311])

        ])
    else:
        return transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            # transforms.Normalize(mean=[0.563, 0.546, 0.524], std=[0.301, 0.303, 0.311])
        ])