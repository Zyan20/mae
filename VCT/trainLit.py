import os, math
from typing import List, Union
from lightning.pytorch.utilities.types import LRSchedulerPLType
from torch import optim, nn
from torch.utils.data import DataLoader

from torchvision import transforms
import lightning as L

from datasets import Viemo90K

from AEFormer import AEFormer



class AEFormerLit(L.LightningModule):
    def __init__(self):
        super().__init__()

        self.model = AEFormer(
            embed_dim = 128,
            decoder_embed_dim = 128,
            depth = 4,
            decoder_depth = 4
        )


    def training_step(self, refs, batch_idx):
        predTokens, z, cls = self.model(refs)

        preds = self.model.unpatchify(predTokens)

        loss = nn.functional.mse_loss(preds, refs)
        psnr = 10 * math.log10(1.0 / (loss))

        
        lr_scheduler = self.lr_schedulers()
        lr_scheduler.step()


        self.log("MSE", loss)
        self.log("PSNR", psnr)
        self.log("lr", self.optimizers().optimizer.state_dict()['param_groups'][0]['lr'])

        return loss
    

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr = 0.005)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 10, 2)

        return [optimizer], [scheduler]
    



if __name__ == "__main__":
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
    dataloader = DataLoader(dataset, batch_size = 256, shuffle = True)

    aeFormer = AEFormerLit()
    
    trainer = L.Trainer(
        max_epochs = 1000, 
        log_every_n_steps = 10,
        default_root_dir = "/root/tf-logs/"
    )
    
    trainer.fit(model = aeFormer, train_dataloaders = dataloader)

