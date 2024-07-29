import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader , Dataset
from torch.optim import AdamW
import torchvision
from torchvision import models
from torchvision import transforms as T

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger



class MLCNNet(nn.Module):
    
        def __init__(self,backbone,n_classes):
            super(MLCNNet,self).__init__()
            self.model = backbone
            self.classifier = nn.Sequential(nn.Linear(512, n_classes))
        def forward(self,x):
            # with torch.no_grad():
            x = self.model(x)
            x = self.classifier(x)
            return x
        

# ------------------------------- #
#       Pytorch Lightning ⚡       # 
#-------------------------------- #

class LitMLCNet(pl.LightningModule):
    
        def __init__(self,model):
            super().__init__();
            self.model = model
        
        def training_step(self,batch,batch_idx):
            x,y = batch
            outputs = self.model(x)
            loss = F.binary_cross_entropy_with_logits(outputs,y)
            self.log("train/loss",loss.item() / len(y), on_epoch=True, prog_bar=True, logger=True)
            return loss
        
        def validation_step(self,batch,batch_idx):
            x,y = batch
            outputs = self.model(x)
            loss = F.binary_cross_entropy_with_logits(outputs,y)
            self.log("val/loss",loss.item() / len(y))
        
        def predict_step(self,batch,batch_idx,dataloader_idx=0):
            x,y = batch
            preds  = self.model(x)
            return preds,y
        
        def configure_optimizers(self):
            optim = torch.optim.AdamW(self.parameters(),lr = 8e-5)
            return optim