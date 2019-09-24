import pandas as pd
import spacy
import os
# needed for stemming
import nltk
from nltk.stem.porter import *


##
## muss im ordner mit den lexika gestartet werden

pd.options.mode.chained_assignment = None  # default='warn'
nlp = spacy.load("en_core_web_lg")
stemmer = nltk.stem.SnowballStemmer('english')
stems = []

current_dir = os.path.dirname(os.path.abspath(__file__))
list_of_datasets = os.listdir(current_dir)
list_of_datasets.remove("clean_lexica.py")

print("datasets: ", list_of_datasets) 

for dataset_name in list_of_datasets: 
	print("reading", dataset_name)
	dataset = pd.read_csv(dataset_name, delimiter=",", dtype={"text": str, "affect": str, "intensity": float})
	
	stems.clear()
	row_index = 0
	
	print("... appending")
	for index, row in dataset.iterrows():
		doc = nlp(row["text"])
		stems.append([stemmer.stem(token.text) for token in doc])

		if row_index % 10000 == 0 and row_index != 0:
			("... searching row " + row_index.__str__() + "/" + len(dataset).__str__())
		row_index = row_index + 1

	dataset["stems"] = stems

	print("... saving\n")
	dataset.to_csv("clean_" + dataset_name, sep=",", index=False, float_format='%.3f')

	


	
