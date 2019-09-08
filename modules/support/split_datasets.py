import pandas as pd
from sklearn.model_selection import train_test_split

datasets = ["crowdflower", "emoint", "emotion_classification_1", "emotion_classification_2", "emotion_classification_3", "emotion_classification_4", "emotion_classification_5", "emotion_classification_6", "emotion_classification_7", "emotion_classification_8"] 
datasets = ["tec"]


####
### Muss im /clean-Ordner gestartet werden
####



for dataset in datasets: 
	data = pd.read_csv(dataset + "_clean.csv", delimiter=",")
	train, test = train_test_split(data, test_size=0.3)

	train.to_csv(dataset + "_train.csv", sep=",", index=False, float_format='%.3f')
	test.to_csv(dataset + "_test.csv", sep=",", index=False, float_format='%.3f')
