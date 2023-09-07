import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision import transforms, utils
from sentence_transformers import SentenceTransformer
from net_cc import SHVNET
from pytorch_lightning import Trainer
import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from PIL import Image
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer
from torch import Tensor
from transformers import ViTFeatureExtractor, ViTModel

#device = torch.device( "cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Contra_Dataset(Dataset):
    def __init__(self):
        self.df = pd.read_pickle('conTra_data.pkl')
        self.image_processor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img1 = self.df.iloc[idx, 0]
        img2 = self.df.iloc[idx, 2]
        lang1 = self.df.iloc[idx, 1]
        lang2 = self.df.iloc[idx, 3]
        action = self.df.iloc[idx, 4]
    
        img1 = self.image_processor(img1, return_tensors="pt")
        img2 = self.image_processor(img2, return_tensors="pt")

        return img1["pixel_values"].squeeze(0), lang1, img2["pixel_values"].squeeze(0), lang2, action

class wavmosLit(pl.LightningModule):
    def __init__(
        self
        ):
        super(wavmosLit, self).__init__()
        self.model = SHVNET()
        self.memories = torch.randn(64, 5024, 768,device = device)

    def forward(self, text,image,memories):
        self._output = self.model(text,image,memories)
        return self._output

    def training_step(self, batch, batch_idx):
        img1, lang1, img2, lang2, action = batch
        outputs1 = self(img1,lang1,self.memories)
        outputs2 = self(img2,lang2,self.memories)
        loss =  loss_function(outputs1[:,0,:], outputs2[:,0,:],action)

        return {"loss": loss}

    def setup(self,stage = False):

        data = Contra_Dataset()
        train_ids = int(0.7* len(data))
        test_ids=len(data)-train_ids
        self.train_dataset,self.test_dataset=torch.utils.data.random_split(data,(train_ids,test_ids))

    def train_dataloader(self):
    
        train_loader = torch.utils.data.DataLoader(
                      self.train_dataset, 
                      batch_size=64,num_workers = 14,shuffle = True,pin_memory = True)
        return train_loader

    def val_dataloader(self):
        
        val_loader = torch.utils.data.DataLoader(
                      self.test_dataset,
                      batch_size=64,num_workers = 14,  shuffle = True,pin_memory = True)
        return val_loader
    
    def validation_step(self, batch, batch_idx):
        img1, lang1, img2, lang2, action = batch
        outputs1 = self(img1,lang1,self.memories)
        outputs2 = self(img2,lang2,self.memories)
        loss =  loss_function(outputs1[:,0,:], outputs2[:,0,:], action)

        return {"loss": loss}
    
    def training_epoch_end(self, training_step_outputs):
        avg_loss = torch.stack([x['loss'] for x in training_step_outputs]).mean()
        self.memories = torch.randn(64, 5024, 768,device = device)
        self.log("T_avg_loss", avg_loss,prog_bar = True)
        wandb.log({"T_avg_loss": avg_loss})

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.memories = torch.randn(64, 5024, 768, device= device)
        self.log("val_loss",avg_loss,prog_bar=True)
        wandb.log({"val_loss": avg_loss})
        return {'val_loss': avg_loss}#, 'log': tensorboard_logs}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=2*(10**(-4)))
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60], gamma=0.1)
        return [optimizer], [scheduler]

if __name__ == "__main__":
    run = wandb.init(project = 'VLN_Capstone')
    pl.utilities.seed.seed_everything(seed=42, workers=True)
    early_stopping = EarlyStopping('val_loss',patience=10,mode='min',verbose=True)
    checkpoint_callback = ModelCheckpoint(
     dirpath='/',
    filename='{epoch}-{val_loss:.2f}-{other_metric:.2f}'
    )
    loss_function = nn.CosineEmbeddingLoss()

    model = wavmosLit()
    trainer = Trainer(gpus =1,max_epochs=100, callbacks=[early_stopping])
    trainer.fit(model)
