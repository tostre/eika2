import pandas as pd
import spacy
import os
# needed for stemming
import nltk
from nltk.stem.porter import *


####
### Muss im /corpora-Ordner gestartet werden
####

 
#- Anzahl der Wörter aus dem Lexikon als Liste  [1,0,0,2,1], normalisiert (durch anzahl der wörter geteilt) 
#- ob mehrere satzzeichen hintereinander kommen (0 oder 1) 

class Clean: 

	def __init__(self):

		pd.options.mode.chained_assignment = None  # default='warn'
		self.nlp = spacy.load("en_core_web_lg")
		self.stemmer = nltk.stem.SnowballStemmer('english')
		print("reading directory\n")
		self.current_dir = os.path.dirname(os.path.abspath(__file__))
		self.list_of_datasets = os.listdir(self.current_dir)
		self.list_of_datasets.remove("clean_corpora.py")
		self.list_of_datasets.remove("zz_kann ich nicht nutzen")
		self.list_of_datasets = ["test.csv"]
		self.list_of_datasets = ["emoint.csv", "crowdflower.csv", "test.csv", "emotion_classification_1.csv", "emotion_classification_2.csv","emotion_classification_3.csv"]
		self.list_of_datasets = ["emotion_classification_4.csv", "emotion_classification_5.csv","emotion_classification_6.csv","emotion_classification_7.csv","emotion_classification_8.csv"]
		self.list_of_datasets = ["tec.csv"]		
		self.list_of_datasets = [dataset_name.replace(".csv", "") for dataset_name in self.list_of_datasets]

		self.emotion_mapping = {"happiness": 0, "sadness": 1, "anger": 2, "fear": 3}

		#self.list_of_datasets = ["emotion_classification_4.csv"]
		# load lexicon datasets and make them lists
		self.lex_happiness = pd.read_csv("../lexica/clean_happiness.csv", dtype={"text":str, "affect":str, "stems":object})
		self.lex_sadness = pd.read_csv("../lexica/clean_sadness.csv", dtype={"text":str, "affect":str, "stems":object})
		self.lex_anger = pd.read_csv("../lexica/clean_anger.csv", dtype={"text":str, "affect":str, "stems":object})
		self.lex_fear = pd.read_csv("../lexica/clean_fear.csv", dtype={"text":str, "affect":str, "stems":object})
		self.list_happiness = self.lex_happiness["stems"].tolist()
		self.list_sadness = self.lex_sadness["stems"].tolist()
		self.list_anger = pd.Series(self.lex_anger["stems"].tolist())
		self.list_fear = self.lex_fear["stems"].tolist()

	def extract_features(self):
		# extract features from dataset
		print(self.list_of_datasets) 
		print("---> extract features") 
		num_dataset = 1
		for dataset_name in self.list_of_datasets: 
			print("reading" , dataset_name, " (", num_dataset,  "/", len(self.list_of_datasets), ")")
			dataset = pd.read_csv(dataset_name + ".csv", delimiter=",", dtype={"text": str, "affect": str, "intensity": float})	

			stems = []
			word_count = []
			upper_word_count = []
			ent_word_count = []
			affect_words = []
			double_punctuation = []
			row_index = 0
			
			print("... analyzing sentence features")
			for index, row in dataset.iterrows():
				# save row as doc
				doc = self.nlp(row["text"])
				# extract features from row
				stems.append([self.stemmer.stem(token.text) for token in doc])
				word_count.append(len(doc))
				upper_word_count.append(sum([token.text.isupper() for token in doc])/len(doc))
				ent_word_count.append(len(doc.ents)/len(doc))
				# give update from time to time
				if row_index % 10000 == 0 and row_index != 0:
					("... searching row " + row_index.__str__() + "/" + len(dataset).__str__())
				row_index = row_index + 1

			# write data to dataset
			print("... writing to dataset")

			dataset["affect"] = dataset["affect"].map(self.emotion_mapping)

			dataset["word_count"] = word_count
			dataset["upper_word_count"] = upper_word_count
			dataset["ent_word_count"] = ent_word_count
			# save dataset
			print("... saving")
			dataset.to_csv(self.current_dir + "/clean/" + dataset_name + "_clean.csv", sep=",", index=False, float_format='%.3f')
			stems = pd.DataFrame(stems)
			stems.to_csv(self.current_dir + "/clean/" + dataset_name + "_stems.csv", sep=",", index=False, float_format='%.3f')
			num_dataset += 1

	def emotion_words(self): 
		print(self.list_of_datasets)
		print("---> emotion words")
		# check for emition words in datasets
		num_dataset = 1
		for dataset_name in self.list_of_datasets: 
			print("reading", dataset_name, "(", 1,  "/", len(self.list_of_datasets), ")")
			dataset_stems = pd.read_csv(self.current_dir + "/clean/" + dataset_name + "_stems.csv", delimiter=",", dtype={"text": str, "affect": str, "intensity": float})
			dataset = pd.read_csv(self.current_dir + "/clean/" + dataset_name + "_clean.csv", delimiter=",", dtype={"text": str, "affect": str, "intensity": float})	

			h_count = []
			s_count = []
			a_count = []
			f_count = []	
			consect_punt_count = []
			

			print("... looking for emotion words")
			for index, row in dataset_stems.iterrows():
				# scores for emotion words found in row
				h = 0
				s = 0
				a = 0
				f = 0		
				# create list from row
				row_list = row.tolist()
				# extract features from row
				for item in row_list: 
					if item in self.list_happiness: 
						h += 1
					elif item in self.list_sadness: 
						s += 1
					elif item in self.list_anger: 
						a += 1
					elif item in self.list_fear: 
						f += 1

				h_count.append(h/len(row_list))
				s_count.append(s/len(row_list))
				a_count.append(a/len(row_list))
				f_count.append(f/len(row_list))

			# save features in dataset
			dataset["h_count"] = h_count
			dataset["s_count"] = s_count
			dataset["a_count"] = a_count
			dataset["f_count"] = f_count

			#save dataset
			print("... saving")
			dataset.to_csv(self.current_dir + "/clean/" + dataset_name + "_clean.csv", sep=",", index=False, float_format='%.3f')
			num_dataset += 1

	def pos(self): 	
		print(self.list_of_datasets)
		print("---> find pos")
		# hole pos tags 
		num_dataset = 1
		for dataset_name in self.list_of_datasets: 
			print("reading", dataset_name, "(", 1,  "/", len(self.list_of_datasets), ")")
			dataset = pd.read_csv(dataset_name + ".csv", delimiter=",", dtype={"text": str, "affect": str, "intensity": float})

			pos = []
			
			print("... generating part-of-speech")
			for index, row in dataset.iterrows():
				doc = self.nlp(row["text"])
				row_pos = []		
				pos.append([token.pos for token in doc])
			print("...saving")
			ds=pd.DataFrame(pos)
			ds.to_csv(self.current_dir + "/clean/" + dataset_name + "_pos.csv", sep=",", index=False, float_format='%.3f')
			num_dataset += 1

	def punct(self): 
		print(self.list_of_datasets)
		print("---> finf punct")
		# look for consecutive PUNCT tags
		num_dataset = 1
		for dataset_name in self.list_of_datasets: 
			print("reading", dataset_name, "(", 1,  "/", len(self.list_of_datasets), ")")
			dataset = pd.read_csv(self.current_dir + "/clean/" + dataset_name + "_clean.csv", delimiter=",", dtype={"text": str, "affect": str, "intensity": float})
			dataset_pos = pd.read_csv(self.current_dir + "/clean/" + dataset_name + "_pos.csv", delimiter=",", dtype={"text": str, "affect": str, "intensity": float})

			cons_punct_count = [0]*len(dataset["text"])

			print("... looking for consecutive PUNCT")
			for index, row in dataset_pos.iterrows(): 
				cons_punct_count[index] = 0	
				row_list = row.tolist()
				for index2, item in enumerate(row_list[:-1]): 
					if item == "PUNCT" and item == row_list[index2+1]:
						cons_punct_count[index] += 1	
						
			print("...saving\n")
			dataset["cons_punct_count"] = cons_punct_count
			dataset.to_csv(self.current_dir + "/clean/" + dataset_name + "_clean.csv", sep=",", index=False, float_format='%.3f')
			num_dataset += 1



clean = Clean()

clean.extract_features()
clean.emotion_words()
clean.pos()
clean.punct()



















	

	
