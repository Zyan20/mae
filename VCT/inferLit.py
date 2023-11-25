import torch
from torch import nn
from torch.utils.data import DataLoader

from torchvision.utils import save_image
from torchvision import transforms

from torchstat import stat
from trainLit import AEFormerLit

from datasets import Viemo90K

import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


transform = transforms.Compose([
    transforms.RandomCrop(224),
    transforms.ToTensor(),
])

# dataset
dataset = Viemo90K(
    root_dir = '/root/autodl-tmp/vimeo_septuplet/sequences',    # edit here
    txt = "/root/autodl-tmp/vimeo_septuplet/sep_testlist.txt",
    transform = transform
)
dataloader = DataLoader(dataset, batch_size = 4, shuffle = True)


# model
aeFormer = AEFormerLit.load_from_checkpoint("./save/epoch=270-step=423730.ckpt")


with torch.no_grad():
    aeFormer.eval()

    for i, images in enumerate(dataloader):
        images = images.to(device)

        predTokens, z, cls = aeFormer(images)

        predImages = aeFormer.unpatchify(predTokens)

        mse = nn.functional.mse_loss(images, predImages)
        psnr = 10 * math.log10(1.0 / (mse))

        save_image(images, "./save/orignal.png")
        save_image(predImages, "./save/pred.png")

        print(psnr)

        break