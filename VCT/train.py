import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import os
from PIL import Image

class Viemo90K(Dataset):
    def __init__(self, 
        root_dir, txt, transform = None, 
    ):
        self.root_dir = root_dir
        self.transform = transform

        self.imagePaths = []

        with open(os.path.join(root_dir, txt)) as f:
            folders = f.readlines()
        
        for folder in folders:
            self.imagePaths += [
                os.path.join(folder, "im1.png"),
                os.path.join(folder, "im3.png"),
                os.path.join(folder, "im5.png"),
                os.path.join(folder, "im7.png"),
            ]

    def __len__(self):
        return len(self.imagePaths)

    def __getitem__(self, idx):
        image = Image.open(self.imagePaths[idx]).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image


transform = transforms.Compose([
    transforms.RandomCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 创建数据集和数据加载器
dataset = Viemo90K(
    root_dir = 'D:/vimeo_septuplet/sequences', 
    txt = "D:/vimeo_septuplet/sep_testlist.txt",
    transform = transform
)

dataloader = DataLoader(dataset, batch_size=32, shuffle = True, num_workers = 4)
