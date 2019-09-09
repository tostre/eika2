import pandas
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import pandas as pd
import os
from sklearn.model_selection import train_test_split


class Lin_Net_Emotion(nn.Module):

	def __init__(self):
		# input = 8, output = 1
		super(Lin_Net_Emotion, self).__init__()
		self.lin1 = nn.Linear(8, 16)
		self.lin3 = nn.Linear(16, 16)
		self.lin2 = nn.Linear(16, 4)

    # Diese Methode propagiert den INput durch das Netz
	def forward(self, x):
		x = F.relu(self.lin1(x))
		x = F.relu(self.lin3(x))
		x = self.lin2(x)
		return x

	def num_flat_features(self, x):
		self.size = x.size()[1:]
		self.num = 1
		# berechnet die anzahl der features (zB 5x3)
		for i in self.size:
			self.num *= i
		return self.num





