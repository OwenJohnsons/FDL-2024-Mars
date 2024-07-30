# ---------------------------------------------------
# one of the pretrained models for multi-class
# multi-level classification problem, found here
# https://www.kaggle.com/code/altairfarooque/multi-label-image-classification-cv-3-0
#----------------------------------------------------


import model_config
from generated_data import *

import torch
import timm
import last_layer
from pytorch_lightning import LightningModule,Trainer
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torchsummary import summary
from sklearn.metrics import accuracy_score

import numpy as np

CFG = model_config.Config()
CFG.seed_everything()

root_dir  = '/home/arushi/'
train_dataset, test_dataset = get_datasets(root_dir)

# -------------------------------- #
#           DataLoaders            #
# -------------------------------- #

train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size = CFG.BATCH_SIZE,
                                               shuffle = True)
test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                               batch_size = 1,
                                               shuffle = False)

timm.list_models("resnet*")
backbone = timm.create_model("resnet34", pretrained=True, num_classes = 0)
backbone

# This is where we would want to change the number of classes for the output
# layer
Model = last_layer.MLCNNet(backbone, 10)

# Pytorch lightning module
pl_Model = last_layer.LitMLCNet(Model)       

trainer = pl.Trainer(default_root_dir = './',
                     max_epochs = 20,
                     log_every_n_steps = 5,
                     accelerator='gpu',
                     devices = 1,
                     callbacks=ModelCheckpoint(dirpath='/home/arushi/model_checkpoints/', filename='model_checkpoint')
                     )

trainer.fit(pl_Model,
            train_dataloader,
            test_dataloader)

summary(backbone, (3, 602, 602), device='cpu')

#---------------------------------------------------
# INFERENCE
#---------------------------------------------------

preds_labels = trainer.predict(pl_Model,test_dataloader)

# to save the output results

preds,labels =[],[]
for item in preds_labels:
    preds.append(torch.round(torch.sigmoid(item[0][0])).detach().numpy().tolist())
    labels.append(item[1][0].detach().numpy().tolist())

np.savetxt('predicted_test_labels_resnet34.txt', preds)

for i in range(len(preds)):
    print(f"{preds[i]}  -  {labels[i]}")

print(f"test_accuracy  - {accuracy_score(preds,labels)}")

#----------------------------------------------------------
# to give it to Owen for random forest
train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size = 1,
                                               shuffle = False)

preds_labels_train = trainer.predict(pl_Model,train_dataloader)

preds,labels =[],[]
for item in preds_labels_train:
    preds.append((torch.sigmoid(item[0][0])).detach().numpy().tolist())
    labels.append(item[1][0].detach().numpy().tolist())

np.savetxt('predicted_train_labels_resnet34.txt', preds)
