import spacy
import pandas
import json
import numpy as np


# Spacy: https://www.youtube.com/watch?v=BY1JD4SPt9o&list=PLJ39kWiJXSiz1LK8d_fyxb7FTn4mBYOsD&index=3
# Pandas: https://www.youtube.com/watch?v=vmEHCJofslg
class CsvTool:
    def __init__(self):
        pandas.options.mode.chained_assignment = None  # default='warn'
        # init all nlp variables
        self.nlp = spacy.load("en_core_web_lg")  # this model has built-in word embeddings
        self.doc = None
        # linguistic feature vectors
        self.embeddings = []
        self.lemmas = []
        self.pos = []
        self.is_upper = []
        self.num_sentences = []
        self.num_ents = []
        # data for cleaning up the dataset
        self.dataset = None
        self.drop_rows = None
        self.clean_rows = None
        self.datasets_to_merge = []
        self.merged_datasets = None
        self.affect_intensity_threshold = 0.4
        self.row_index = 0
        # List of datasets, has to be maintained manually atm
        self.list_of_corpora = [["corpora/", "test", ","],
                                ["corpora/", "crowdflower", ","],
                                ["corpora/", "emoint", "\t"],
                                ["corpora/", "isear", "\t"],
                                ["corpora/", "meld", ","],
                                ["corpora/", "emotion_classification_1", ","],
                                ["corpora/", "emotion_classification_2", ","],
                                ["corpora/", "emotion_classification_3", ","],
                                ["corpora/", "emotion_classification_4", ","],
                                ["corpora/", "emotion_classification_5", ","],
                                ["corpora/", "emotion_classification_6", ","],
                                ["corpora/", "emotion_classification_7", ","],
                                ["corpora/", "emotion_classification_8", ","]]
        self.list_of_corpora = [["corpora/", "emotion_classification_1", ","],
                                ["corpora/", "emotion_classification_2", ","],
                                ["corpora/", "emotion_classification_3", ","],
                                ["corpora/", "emotion_classification_4", ","],
                                ["corpora/", "emotion_classification_5", ","],
                                ["corpora/", "emotion_classification_6", ","],
                                ["corpora/", "emotion_classification_7", ","],
                                ["corpora/", "emotion_classification_8", ","]]
        self.list_of_lexica = [["lexica/", "nrc_intensity", "\t"],
                               ["lexica/", "wordinfo", ","]]


        self.cleanup_datasets(self.list_of_corpora)
        self.save_features_in_datasets(self.list_of_corpora, is_lexicon=False)
        #self.merge_datasets(self.list_of_corpora, is_lexicon=False)

        self.cleanup_datasets(self.list_of_lexica)
        self.save_features_in_datasets(self.list_of_lexica, is_lexicon=True)
        #self.merge_datasets(self.list_of_corpora, is_lexicon=False)




    # Removes unwanted emotion tags, forbidden strings and every delimiter but ","
    def cleanup_datasets(self, list_of_datasets):
        # emotion classification excluded, because it would take all day
        for dataset in list_of_datasets:  # [:-1]:
            print("cleaning up dataset", dataset[1])
            # Read the dataset
            self.dataset = self.load_dataset(dataset, "raw")

            # Choose only the rows annotated with 1 of the 5 emotions
            print("... remove unwanted affect tags")
            self.clean_rows = self.dataset.loc[(self.dataset["affect"] == "happiness") |
                                               (self.dataset["affect"] == "sadness") |
                                               (self.dataset["affect"] == "anger") |
                                               (self.dataset["affect"] == "fear") |
                                               (self.dataset["affect"] == "disgust")]

            # Reset the index numbers for the rows, else the drop_rows thing wouldn't work
            self.clean_rows.reset_index(inplace=True, drop=True)

            # Save all rows that contain links/mentions/hashtags in series object
            print("... remove links and mentions")
            self.drop_rows = []
            for index, row in self.clean_rows.iterrows():
                self.drop_rows.append(self.contains_forbidden_symbols(row, dataset[1]))
                self.update_log(self.row_index, self.dataset)

            # make a series from the drop-List
            self.drop_rows = pandas.Series(self.drop_rows)
            # Choose only the rows that don't contain links or mentions
            self.clean_rows = self.clean_rows[self.drop_rows]

            # replace special characters (emotion classification takes too long)
            print("... replace special characters")
            self.clean_rows["text"] = self.clean_rows["text"].str.replace("&amp", "and")
            self.clean_rows["text"] = self.clean_rows["text"].str.replace("&amp;", "and")
            self.clean_rows["text"] = self.clean_rows["text"].str.replace("&quot", "\"")
            self.clean_rows["text"] = self.clean_rows["text"].str.replace("&quot;", "\"")
            self.clean_rows["text"] = self.clean_rows["text"].str.replace("&lt", " ")
            self.clean_rows["text"] = self.clean_rows["text"].str.replace("&lt;", "<")
            self.clean_rows["text"] = self.clean_rows["text"].str.replace("&gt", ">")
            self.clean_rows["text"] = self.clean_rows["text"].str.replace("&gt;", ">")
            self.clean_rows["text"] = self.clean_rows["text"].str.replace("w/;", "with")

            # Save these datasets as a cleaned up corpus with delimiter ","
            self.clean_rows.reset_index(inplace=True, drop=True)
            if dataset[1] == "nrc_intensity":
                # delete the column intensity for nrc_intensity
                self.clean_rows = self.clean_rows.drop(columns="intensity")
                self.clean_rows.to_csv(dataset[0] + dataset[1] + "_clean.csv", sep=",", index=False, float_format='%.3f')
            else:
                self.clean_rows.to_csv(dataset[0] + dataset[1] + "_clean.csv", sep=",", index=False)
            print("... done\n")

    # analyses linguistic features of the csv and saves them in file
    def save_features_in_datasets(self, list_of_datasets, is_lexicon=False):
        for dataset in list_of_datasets:
            # clear lists from previous dataset
            self.embeddings.clear()
            self.lemmas.clear()
            self.pos.clear()
            self.is_upper.clear()
            self.num_sentences.clear()
            self.num_ents.clear()
            self.row_index = 0

            # Read the dataset
            self.dataset = self.load_dataset(dataset, "clean")

            print("searching for features in", dataset[1])
            # for every entry in the dataset
            for index, row in self.dataset.iterrows():
                self.doc = self.nlp(row["text"])
                if not is_lexicon:
                    # create lists for pos, lemma etc, and insert them into overall lemma/pos-lists
                    self.embeddings.append([list(token.vector) for token in self.doc if not token.is_stop])  # meaning of the words
                    self.lemmas.append([token.lemma for token in self.doc if not token.is_stop])  # lemmas of the words
                    self.pos.append([token.pos for token in self.doc if not token.is_stop])  # part-of-speech of the words
                    # self.is_upper.append(([token.text.isupper() for token in self.doc])) # if words are all upper case
                    #self.is_upper.append(len([token.text.isupper for token in self.doc]))
                    self.is_upper.append(list(token.text.isupper() for token in self.doc).count(True))
                    self.num_sentences.append(len(list(self.doc.sents)))  # number of sentences in text
                    self.num_ents.append(len(self.doc.ents))  # number of named entities in text
                else:
                    # difference to corpora: Stop words will not be removed
                    #self.embeddings.append([token.vector for token in self.doc])
                    self.embeddings.append([list(token.vector) for token in self.doc if not token.is_stop])
                    self.lemmas.append([token.lemma for token in self.doc])

                # give update on progress from time to time
                if self.row_index % 10000 == 0 and self.row_index != 0:
                    print("... searching row " + self.row_index.__str__() + "/" + len(self.dataset).__str__())
                self.row_index = self.row_index + 1

            # Write the linguistic lists into the dataFrame
            print("... saving features to the dataframe")
            # do this in any case
            self.dataset["embeddings"] = self.embeddings
            self.dataset["lemmas"] = self.lemmas
            # do this only for corpora
            if not is_lexicon:
                self.dataset["isupper"] = self.is_upper
                self.dataset["pos"] = self.pos
                self.dataset["num_sentences"] = self.num_sentences
                self.dataset["num_ents"] = self.num_ents

            # save the dataset to the file system
            self.dataset.to_csv(dataset[0] + dataset[1] + "_clean.csv", sep=",", index=False)
            print("... done\n")

    # merge all the datasets and save them in one file
    def merge_datasets(self, list_of_datasets, is_lexicon=False):
        # load all datasets to merge
        print("merging datasets")
        for dataset in list_of_datasets:
            print("... reading", dataset[1])
            # self.datasets_to_merge.append(pandas.read_csv(dataset[0] + dataset[1] + "_clean.csv", delimiter=","))
            if not is_lexicon:
                self.datasets_to_merge.append(pandas.read_csv(dataset[0] + dataset[1] + "_clean.csv", delimiter=",",
                                                              dtype={"text": str, "affect": str, "embeddings": object,
                                                                     "lemmas": object, "is_upper": int,
                                                                     "num_sentences": int, "num_ents": int}))
            else:
                self.datasets_to_merge.append(pandas.read_csv(dataset[0] + dataset[1] + "_clean.csv", delimiter=",",
                                                              dtype={"text": str, "affect": str, "embeddings": object,
                                                                     "lemmas": object}))

        # merge the datasets in the list
        print("... merging")
        self.merged_datasets = pandas.concat(self.datasets_to_merge, sort=False)
        if not is_lexicon:
            self.merged_datasets.to_csv("corpora/merged_corpora.csv", sep=",", index=False)
        else:
            self.merged_datasets.to_csv("lexica/merged_lexica.csv", sep=",", index=False)

        # read the merged dataset
        print("... removing duplicate entries")
        if not is_lexicon:
            self.dataset = pandas.read_csv("corpora/merged_corpora.csv", delimiter=",",
                                           dtype={"text": str, "affect": str, "embeddings": object,
                                                  "lemmas": object, "is_upper": int,
                                                  "num_sentences": int, "num_ents": int})
        else:
            self.dataset = pandas.read_csv("corpora/merged_corpora.csv", delimiter=",",
                                           dtype={"text": str, "affect": str, "embeddings": object,
                                                  "lemmas": object, "is_upper": int,
                                                  "num_sentences": int, "num_ents": int,
                                                  "intensity": float})
        # remove all duplicates
        self.dataset.drop_duplicates(subset="text", keep="first", inplace=True)

        # save the dataset again
        print("... saving")
        if not is_lexicon:
            self.dataset.to_csv("corpora/merged_corpora.csv", sep=",", index=False)
        else:
            self.dataset.to_csv("lexica/merged_lexica.csv", sep=",", index=False)
        print("... done\n")

    # loads a dataset
    def load_dataset(self, dataset, status, is_lexicon=False):
        if status == "raw":
            if not dataset[1] == "nrc_intensity":
                return pandas.read_csv(dataset[0] + "original/" + dataset[1] + ".csv",
                                       delimiter=dataset[2],
                                       dtype={"text": str, "affect": str})
            else:
                return pandas.read_csv(dataset[0] + "original/" + dataset[1] + ".csv",
                                       delimiter=dataset[2],
                                       dtype={"text": str, "affect": str, "intensity": float})
        elif status == "clean":
            return pandas.read_csv(dataset[0] + dataset[1] + "_clean.csv",
                                   delimiter=",",
                                   dtype={"text": str, "affect": str})
        elif status == "features":
            if not is_lexicon:
                return pandas.read_csv("corpora/merged_corpora.csv", delimiter=",",
                                       dtype={"text": str, "affect": str, "embeddings": object,
                                              "lemmas": object, "is_upper": int,
                                              "num_sentences": int, "num_ents": int})
            else:
                return pandas.read_csv("lexica/merged_lexica.csv", delimiter=",",
                                       dtype={"text": str, "affect": str, "embeddings": object,
                                              "lemmas": object})

    # check if a row contains forbidden symbold
    def contains_forbidden_symbols(self, row, dataset_name):
        if not dataset_name == "nrc_intensity":
            if row["text"].__contains__("http") \
                    or row["text"].__contains__("@") \
                    or row["text"].__contains__("#"):
                return False
        else:
            if row["text"].__contains__("http") \
                    or row["text"].__contains__("@") \
                    or row["text"].__contains__("#") \
                    or row["intensity"] < self.affect_intensity_threshold:
                return False

        return True

    # update log when dataset is very large
    def update_log(self, row_index, dataset):
        # give update on progress from time to time
        if row_index % 10000 == 0 and row_index != 0:
            print("... searching row " + row_index.__str__() + "/" + len(dataset).__str__())
        row_index = row_index + 1

    # Getter methods
    def get_list_of_corpora(self):
        return self.list_of_corpora

    def get_list_of_lexica(self):
        return self.list_of_lexica


# @todo
# vlt kann ich diese datensätze nicht nehmen:
# isear: beschreibungen von vorfällen
# meld (sind keine chatnachrichten, sondern sauber abgetippte tv-dialoge)
# tec (hashtags, außerdem spanische zeilen)

# features: satzzeichen


cleaner = CsvTool()
