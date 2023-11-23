import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from torchvision import transforms
from datasets import Viemo90K

from AEFormer import AEFormer
import math

# ======================
batch_size = 32

pretrain = False
checkpoint = "./save/epoch12.pth"

encoder_embed_dim = 128
encoder_depth = 4

decoder_embed_dim = 128
decoder_depth = 4

save_dir = "./save"
log_dir = "./save/log"
# ======================


# 初始化模型、损失函数和优化器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AEFormer(
    embed_dim = encoder_embed_dim,
    decoder_embed_dim = decoder_embed_dim,
    depth = encoder_depth,
    decoder_depth = decoder_depth
)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = 0.005)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 1, gamma = 0.5)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0 = 20, T_mult = 2, eta_min = 1e-5)
# scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma = 1 / 3)


# load pertrain
epoch = 0
global_step = 0

if pretrain:
    state = torch.load(checkpoint)

    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    epoch = state["epoch"]
    global_step = state["global_step"]



# 创建数据集和数据加载器
transform = transforms.Compose([
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = Viemo90K(
    root_dir = '/root/autodl-tmp/vimeo_septuplet/sequences',    # edit here
    txt = "/root/autodl-tmp/vimeo_septuplet/sep_trainlist.txt",
    transform = transform
)

dataloader = DataLoader(dataset, batch_size = 16, shuffle = True)
print(len(dataloader))


# Summary
writer = SummaryWriter(log_dir = log_dir)


# 训练模型
model.to(device)
model.train()
num_epochs = 1000

for _ in range(num_epochs):
    for i, refs in enumerate(dataloader):

        refs = refs.to(device)

        predTokens, z, cls = model(refs)

        preds = model.unpatchify(predTokens)

        loss = criterion(refs, preds)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        
        mse = loss.item()
        psnr = 10 * math.log10(1.0 / (mse))


        global_step += 1

        if i % 10 == 0:
            lr = optimizer.state_dict()['param_groups'][0]['lr']
            print(f'Epoch [{epoch}/{num_epochs}], Step [{i}/{len(dataloader)}], Loss: {mse:.4f}, PSNR: {psnr}, LR: {lr}')
            
            scheduler.step()
        
        if iter % 100 == 0:
            writer.add_scalar("MSE", mse, global_step = global_step)
            writer.add_scalar("MSE", psnr)

        


    if epoch % 3 == 0:
        torch.save({
            "model": model.state_dict(), 
            "optimizer": optimizer.state_dict(), 
            "epoch": epoch,
            "global_step": global_step

        }, f'./save/epoch{epoch}.pth')
        
    epoch += 1
    
#     if lr > 1e-6:
#         scheduler.step()
            

