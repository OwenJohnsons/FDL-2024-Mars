import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics
# from focal_loss import *
import torchvision

class RNN(pl.LightningModule):
    def __init__(self, max_val_possible, embedding_dim, hidden_dim, output_dim, num_layers_rnn):
        super().__init__()
        self.save_hyperparameters()
        
        self.embedding = nn.Embedding(max_val_possible, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers=num_layers_rnn)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=0) #dont change
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, batch):
        x = batch
        embedded = self.embedding(x.long())
        output, _ = self.rnn(embedded)
        out = self.fc(output[:,-1,:])

        # if you use cross entropy loss then it implicitly computes softmax distro
        # out = self.softmax(out)

        return out
    
    
    
    def compute_loss(self, predictions, labels):
        criterion = nn.CrossEntropyLoss()
        return criterion(predictions, labels)
    
    
    def training_step(self, train_batch, batch_idx):
        inputs_train, labels_train = train_batch
        pred_train = self.forward(inputs_train)
        loss_train = self.compute_loss(pred_train.float(), labels_train.float())
        self.log('Batch Train Loss', loss_train.mean(), sync_dist=True, prog_bar=True)
        
        train_accuracy = torch.sum(torch.round(self.sigmoid(pred_train)) == labels_train)/labels_train.numel()
        self.log('Batch Multiclass Accuracy', train_accuracy, sync_dist=True, prog_bar=True)
        
        del inputs_train, labels_train
        
        return loss_train
    
    
    def validation_step(self, val_batch, batch_idx):
        inputs_val, labels_val = val_batch
        pred_val = self.forward(inputs_val)
        loss_val = self.compute_loss(pred_val.float(), labels_val.float())
        self.log('Batch Val Loss', loss_val.mean(), sync_dist=True, prog_bar=True)
        
        del inputs_val, labels_val
        
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer