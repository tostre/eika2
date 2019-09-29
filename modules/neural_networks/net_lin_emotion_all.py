import torch.nn as nn
import torch.nn.functional as F


class Net_Lin_Emotion_All(nn.Module):

    def __init__(self):
        # initiate layers
        super(Net_Lin_Emotion_All, self).__init__()
        self.lin1 = nn.Linear(8, 8)
        self.lin2 = nn.Linear(8, 4)

    def forward(self, x):
        # define forward pass
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return x

    def num_flat_features(self, x):
        self.size = x.size()[1:]
        self.num = 1
        # calculate the number of features
        for i in self.size:
            self.num *= i
        return self.num
