import torch.nn as nn
import torch

class LSTM(nn.Module):
    def __init__(self, lstm_features, lstm_hidden_units, output_dim, num_layers_lstm):
        super().__init__()
        self.lstm = nn.LSTM(input_size=lstm_features, hidden_size=lstm_hidden_units,
                            batch_first=True,
                            num_layers=num_layers_lstm)
        self.fc = nn.Linear(lstm_hidden_units, output_dim)
        self.softmax = nn.Softmax(dim=0) #dont change

        self.num_layers = num_layers_lstm
        self.hidden_units = lstm_hidden_units

    def forward(self, batch, device):
        x = batch

        # re-initialize hidden and cell states to make it a stateless model
        batch_size = len(batch)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_()
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_()

        h0, c0 = h0.to(device), c0.to(device)

        out, _ = self.lstm(x.float(), (h0,c0))
        out = self.fc(out[:,-1])

        # using cross entropy loss automatically implies a softmax computation on output so we dont need to do it
        # out = self.softmax(out)

        return out
