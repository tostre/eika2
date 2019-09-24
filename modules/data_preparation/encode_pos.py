import torch.nn as nn
from torch.nn import functional as F
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np 

pos_encoding = {84.0: 1, 85.0: 2, 86.0: 3, 87.0: 4, 89.0: 5, 90.0: 6, 91.0: 7, 92.0: 8, 93.0: 9, 
				94.0: 10, 95.0: 11, 96.0: 12, 97.0: 13, 99.0: 14, 100.0: 15, 101.0: 16, 03.0: 17, np.nan: 18}


def encode(list_of_datasets): 
	# load datasats and merge them into one
	dataset = []
	for f in list_of_datasets:
		print("reading", f)
		dataset = pd.read_csv("../" + f)
	
		# replace all cell entries with the encodings
		for (columnName, columnData) in dataset.iteritems():
			columnData = columnData.map(pos_encoding)
			dataset[columnName] = columnData
		print("saving", f)
		dataset.to_csv("../" + f, index=False, float_format='%.3f')



encode(["crowdflower_pos.csv", "emoint_pos.csv", "tec_pos.csv"])
encode(["emotion_classification_1_pos.csv", "emotion_classification_2_pos.csv", "emotion_classification_3_pos.csv", "emotion_classification_4_pos.csv", "emotion_classification_5_pos.csv", "emotion_classification_6_pos.csv", "emotion_classification_7_pos.csv", "emotion_classification_8_pos.csv"])



















