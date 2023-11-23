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
            folder = folder.strip()
            self.imagePaths += [
                os.path.join(self.root_dir, folder, "im1.png"),
                # os.path.join(self.root_dir, folder, "im3.png"),
                # os.path.join(self.root_dir, folder, "im5.png"),
                # os.path.join(self.root_dir, folder, "im7.png"),
            ]
        
        print(self.imagePaths[0: 5])

    def __len__(self):
        return len(self.imagePaths)

    def __getitem__(self, idx):
        image = Image.open(self.imagePaths[idx]).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image


transform = transforms.Compose([
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])