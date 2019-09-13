import pandas as pd

list_of_datasets = ["emoint_pos.csv", "crowdflower_pos.csv", "emotion_classification_1_pos.csv", "emotion_classification_2_pos.csv","emotion_classification_3_pos.csv", "emotion_classification_4_pos.csv", "emotion_classification_5_pos.csv","emotion_classification_6_pos.csv","emotion_classification_7_pos.csv","emotion_classification_8_pos.csv"]
pos = []		

def count_unique_pos():
	for set in list_of_datasets: 
		dataset = pd.read_csv(set)
		print(set) 
		for index, row in dataset.iterrows(): 	 
			for element in row:
				if not pd.isna(element) and element not in pos:
					pos.append(element)		
	return pos

pos = [86.0, 95.0, 100.0, 84.0, 85.0, 92.0, 94.0, 91.0, 90.0, 97.0, 93.0, 96.0, 89.0, 87.0, 99.0, 101.0, 103.0]
pos.sort()

# create a one-hot-encoding for the pos tags
def encode_one_hot():
	encoding = dict.fromkeys(pos, ([]))
	
	pos_index = 0; 
	
	for key in encoding.keys(): 
		print("pos_index", pos_index) 
		l = [0] * len(pos)
		l[pos_index] = 1
		encoding[key] = l
		pos_index += 1

	for key, value in encoding.items():
		print(key, value) 

encode_one_hot()
