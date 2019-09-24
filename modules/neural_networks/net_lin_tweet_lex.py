import torch.nn as nn
import torch.nn.functional as F


class Net_Lin_Tweet_Lex(nn.Module):

    def __init__(self):
        # input = 8, output = 1
        super(Net_Lin_Tweet_Lex, self).__init__()
        self.lin1 = nn.Linear(4, 8)
        self.lin2 = nn.Linear(8, 4)

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return x

    def num_flat_features(self, x):
        self.size = x.size()[1:]
        self.num = 1
        for i in self.size:
            self.num *= i
        return self.num
