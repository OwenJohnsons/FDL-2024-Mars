import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics
# from focal_loss import *
import torchvision

class LSTM(pl.LightningModule):
    def __init__(self, lstm_features, lstm_hidden_units, output_dim, num_layers_lstm):
        super().__init__()
        self.save_hyperparameters()
        self.lstm = nn.LSTM(input_size=lstm_features, 
                            hidden_size=lstm_hidden_units,
                            batch_first=True,
                            num_layers=num_layers_lstm)
        self.fc = nn.Linear(lstm_hidden_units, output_dim)

        self.num_layers = num_layers_lstm
        self.hidden_units = lstm_hidden_units
        self.sigmoid = nn.Sigmoid()

    def forward(self, batch):
        x = batch

        # re-initialize hidden and cell states to make it a stateless model
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_units).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_units).requires_grad_()
        
        h0, c0 = h0.to(x.device), c0.to(x.device)

        out, _ = self.lstm(x.float(), (h0,c0))
        out = self.fc(out[:,-1])

        # using cross entropy loss automatically implies a softmax computation on output so we dont need to do it
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
    