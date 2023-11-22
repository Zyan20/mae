import torch
import torch.nn as nn

from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image
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


# load model
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

model = AEFormer(
    embed_dim = 128,
    decoder_embed_dim = 128
)

model.load_state_dict(torch.load("./save/epoch6.pth"))
model.eval()
criterion = nn.MSELoss()



transform = transforms.Compose([
    transforms.RandomCrop(224),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# dataset
dataset = Viemo90K(
    root_dir = 'D:/vimeo_septuplet/sequences',
    txt = "D:/vimeo_septuplet/sep_testlist.txt",
    transform = transform
)
dataloader = DataLoader(dataset, batch_size = 4, shuffle = True)


with torch.no_grad():
    model.eval()
    model.to(device)

    for i, images in enumerate(dataloader):
        images = images.to(device)

        predTokens, z, cls = model(images)

        predImages = model.unpatchify(predTokens)

        loss = criterion(images, predImages)

        save_image(images, "./save/orignal.png")
        save_image(predImages, "./save/pred.png")
        # save_image(cls, "./save/clsToken.png")

        print(loss.item())

        break



