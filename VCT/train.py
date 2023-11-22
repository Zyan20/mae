import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import os
from PIL import Image

from AEFormer import AEFormer

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
                os.path.join(self.root_dir, folder, "im3.png"),
                os.path.join(self.root_dir, folder, "im5.png"),
                os.path.join(self.root_dir, folder, "im7.png"),
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

# 创建数据集和数据加载器
dataset = Viemo90K(
    root_dir = 'D:/vimeo_septuplet/sequences', 
    txt = "D:/vimeo_septuplet/sep_trainlist.txt",
    transform = transform
)

dataloader = DataLoader(dataset, batch_size = 256, shuffle = True)
print(len(dataloader))

# 初始化模型、损失函数和优化器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AEFormer(
    embed_dim = 128,
    decoder_embed_dim = 128
)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = 0.0001)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 2, gamma = 0.5, )
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma = 1 / 4)


# load pertrain
pretrain = False
epoch = 0

if pretrain:
    state = torch.load("./save/epoch6.pth")
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(optimizer["optimizer"])
    epoch = optimizer["epoch"]




# 训练模型
model.to(device)
model.train()
num_epochs = 200


for _ in range(num_epochs):
    for i, refs in enumerate(dataloader):

        refs = refs.to(device)

        predTokens, z, cls = model(refs)

        preds = model.unpatchify(predTokens)

        loss = criterion(refs, preds)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        if i % 10 == 0:
            lr = optimizer.state_dict()['param_groups'][0]['lr']
            print(f'Epoch [{epoch}/{num_epochs}], Step [{i}/{len(dataloader)}], Loss: {loss.item():.4f}, LR: {lr}')
        
        if epoch != 0 and epoch % 4 == 0:
            torch.save({
                "model": model.state_dict(), 
                'optimizer': optimizer.state_dict(), 
                'epoch':epoch
            })

            torch.save(model.state_dict(), f'./save/epoch{epoch}.pth')


    scheduler.step()
    epoch += 1

