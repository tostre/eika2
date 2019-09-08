
import pandas as pd

list_of_datasets = ["crowdflower", "emoint", "emotion_classification_1", "emotion_classification_2", "emotion_classification_3", "emotion_classification_4", "emotion_classification_5", "emotion_classification_6", "emotion_classification_7", "emotion_classification_8", "test"]
list_of_datasets = ["tec"]

####
### Muss im /clean-Ordner gestartet werden
####

num_dataset = 1
for dataset_name in list_of_datasets: 
	dataset = pd.read_csv(dataset_name + "_clean.csv", delimiter=",", dtype={"text": str, "affect": str, "intensity": float})

	cons_punct_count = [0]*len(dataset["text"])

	for index, row in dataset.iterrows(): 
		cons_punct_count[index] = 0	

		text = row["text"]

		for index2, char in enumerate(text):
			try:
				if text[index2] == "." and text[index2] == text[index2+1]:
					  cons_punct_count[index] += 1
			except IndexError:
				pass
			try:
				if text[index2] == "!" and text[index2] == text[index2+1]:
					  cons_punct_count[index] += 1
			except IndexError:
				pass
			try:
				if text[index2] == "?" and text[index2] == text[index2+1]:
					  cons_punct_count[index] += 1
			except IndexError:
				pass
			try:
				if text[index2] == "," and text[index2] == text[index2+1]:
					  cons_punct_count[index] += 1
			except IndexError:
				pass
	
	dataset["cons_punct_count"] = cons_punct_count
	dataset.to_csv(dataset_name + "_clean.csv", sep=",", index=False, float_format='%.3f')
	num_dataset += 1

