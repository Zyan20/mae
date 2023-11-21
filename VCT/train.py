import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import os
from PIL import Image

from VC import AEFormer

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
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 创建数据集和数据加载器
dataset = Viemo90K(
    root_dir = 'D:/vimeo_septuplet/sequences', 
    txt = "D:/vimeo_septuplet/sep_testlist.txt",
    transform = transform
)

dataloader = DataLoader(dataset, batch_size = 32, shuffle = True)

# 初始化模型、损失函数和优化器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AEFormer().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01, momentum=0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 10, gamma = 0.1)


# 训练模型
num_epochs = 10000
for epoch in range(num_epochs):
    model.train()

    for i, orignalImage in enumerate(dataloader):
        inputs = orignalImage.to(device)

        predTokens = model(inputs)

        predImage = model.unpatchify(predTokens)

        loss = criterion(orignalImage, predImage)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        scheduler.step()

        if (i + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(dataloader)}], Loss: {loss.item():.4f}')


