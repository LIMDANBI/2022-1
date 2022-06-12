import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class BaseModel(nn.Module):

    def __init__(self, input_size, output_size, embedding_dim, hidden_dim, num_layers, batch_first):
        super(BaseModel, self).__init__()

        self.num_layers = num_layers
        self.batch_first = batch_first   
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(input_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, bidirectional=True, batch_first=batch_first)
        self.fc = nn.Linear(hidden_dim*2, output_size)

    # the size of x in forward is (seq_length, batch_size) if batch_first=False
    def forward(self, x):
        batch_size = x.size(0) if self.batch_first else x.size(1)

        #h_0: (num_layers * num_directions, batch_size, hidden_size)
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim)

        embedding = self.embedding(x)
        outputs, hidden = self.lstm(embedding, None) 
        outputs = outputs[:, -1, :] if self.batch_first else outputs[-1, :, :]
        output = self.fc(outputs)

        return output, hidden