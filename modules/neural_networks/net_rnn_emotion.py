import torch.nn as nn
import numpy as np
import torch

pos_encoding = {84.0: 1, 85.0: 2, 86.0: 3, 87.0: 4, 89.0: 5, 90.0: 6, 91.0: 7, 92.0: 8, 93.0: 9,
                94.0: 10, 95.0: 11, 96.0: 12, 97.0: 13, 99.0: 14, 100.0: 15, 101.0: 16, 03.0: 17, np.nan: 18}


class Net_Rnn_Emotion(nn.Module):
    def __init__(self, vocab_size=19, output_size=4, embedding_dim=179, hidden_dim=256, n_layers=1, drop_prob=0.5):
        super(Net_Rnn_Emotion, self).__init__()
        # Variables
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_cuda = torch.cuda.is_available()
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.batch_size = 50
        # initiate layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, batch_first=True)
        self.lin1 = nn.Linear(hidden_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, output_size)

    # return the initial hidden state of the lstm layer (hidden state and cell state)
    def init_hidden(self, batch_size):
        if self.is_cuda:
            print("Returning cuda hidden state")
            return (torch.randn(self.n_layers, batch_size, self.hidden_dim).cuda(), torch.randn(self.n_layers, batch_size, self.hidden_dim).cuda())
        else:
            return (torch.randn(self.n_layers, batch_size, self.hidden_dim), torch.randn(self.n_layers, batch_size, self.hidden_dim))

    # propagate features through netword
    def forward(self, x, hidden):
        x = x.long()

        if self.is_cuda:
            x = x.to(self.device)

        embeds = self.embedding(x)
        out, hidden = self.lstm(embeds, hidden)
        # take only the last hidden_dim output of the lstm
        out = self.lin1(out[:, -1, :])
        out = self.lin2(out)
        return out, hidden
