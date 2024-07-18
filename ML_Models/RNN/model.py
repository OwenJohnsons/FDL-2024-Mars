import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class RNN(nn.Module):
    def __init__(self, max_val_possible, embedding_dim, hidden_dim, output_dim, num_layers_rnn):
        super().__init__()
        self.embedding = nn.Embedding(max_val_possible, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers=num_layers_rnn)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=0) #dont change

    def forward(self, batch):
        x = batch
        embedded = self.embedding(x.long())
        output, _ = self.rnn(embedded)
        out = self.fc(output[:,-1,:])

        # if you use cross entropy loss then it implicitly computes softmax distro
        # out = self.softmax(out)

        return out
