import numpy as np
import empath
import torch
import spacy
import nltk
import pandas as pd
from modules.support.lin_all_tweet import Lin_Net_Tweet
from modules.support.lin_all_emotion import Lin_Net_Emotion


class Classifier:

    def __init__(self, topic_keywords):
        self.lexicon = empath.Empath()
        self.keyword_analysis = []
        self.topic_keywords = topic_keywords
        self.topics_set = None
        self.nlp = spacy.load("en_core_web_lg")
        self.emotion_mapping = {"happiness": 0, "sadness": 1, "anger": 2, "fear": 3}
        self.stemmer = nltk.stem.SnowballStemmer('english')

        # load neural networks
        self.net_lin_all_tweet = Lin_Net_Tweet()
        self.net_lin_all_tweet.load_state_dict(torch.load("../nets/lin_all_tweet.pt"))
        self.net_lin_all_emo = Lin_Net_Emotion()
        self.net_lin_all_emo.load_state_dict(torch.load("../nets/lin_all_emotion.pt"))

        # load emotion lexica
        self.lex_happiness = pd.read_csv("../lexica/clean_happiness.csv", delimiter=",", dtype={"text": str, "affect": str, "stems": str})
        self.lex_sadness = pd.read_csv("../lexica/clean_happiness.csv", delimiter=",", dtype={"text": str, "affect": str, "stems": str})
        self.lex_anger = pd.read_csv("../lexica/clean_happiness.csv", delimiter=",", dtype={"text": str, "affect": str, "stems": str})
        self.lex_fear = pd.read_csv("../lexica/clean_happiness.csv", delimiter=",", dtype={"text": str, "affect": str, "stems": str})
        # turn them into lists
        self.list_happiness = self.lex_happiness["stems"].tolist()
        self.list_sadness = self.lex_sadness["stems"].tolist()
        self.list_anger = pd.Series(self.lex_anger["stems"].tolist())
        self.list_fear = self.lex_fear["stems"].tolist()

        # init variables for extracting features from user input
        self.doc = None
        self.stemmed_input = None
        self.word_count = 0
        self.upper_word_count = 0
        self.ent_word_count = 0
        self.cons_punct_count = 0
        self.features = []
        self.lex_words = []
        self.all_features = []
        self.emotions = None

        # map for one-hot-encoding the spacy pos-tags (see script encode_pos.py)
        self.pos_encoding = {84.0: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             85.0: [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            86.0: [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             87.0: [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             89.0: [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             90.0: [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             91.0: [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             92.0: [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             93.0: [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                             94.0: [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                             95.0: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                             96.0: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                             97.0: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                             99.0: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                             100.0: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                             101.0: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                             103.0: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]}

    # searches for features in user input (word count, upper word count, etc)
    def get_input_features(self, user_input):
        self.doc = self.nlp(user_input)
        self.word_count = len(self.doc)
        self.upper_word_count = sum([token.text.isupper() for token in self.doc]) / len(self.doc)
        self.ent_word_count = len(self.doc.ents) / len(self.doc)
        self.cons_punct_count = 0

        for index, char in enumerate(user_input):
            try:
                if user_input[index] == "." and user_input[index] == user_input[index + 1]:
                    self.cons_punct_count += 1
            except IndexError:
                pass
            try:
                if user_input[index] == "!" and user_input[index] == user_input[index + 1]:
                    self.cons_punct_count += 1
            except IndexError:
                pass
            try:
                if user_input[index] == "?" and user_input[index] == user_input[index + 1]:
                    self.cons_punct_count += 1
            except IndexError:
                pass
            try:
                if user_input[index] == "," and user_input[index] == user_input[index + 1]:
                    self.cons_punct_count += 1
            except IndexError:
                pass

        return [self.word_count, self.upper_word_count, self.ent_word_count, self.cons_punct_count]

    # get_input_lexicon_words
    def get_input_lexicon_words(self, user_input):
        # stem the user input
        self.doc = self.nlp(user_input)
        self.stemmed_input = [self.stemmer.stem(token.text) for token in self.doc]
        # count the occurances of words from lexica
        h, s, a, f = 0, 0, 0, 0

        for word in self.stemmed_input:
            if word in self.list_happiness:
                h += 1
            elif word in self.list_sadness:
                s += 1
            elif word in self.list_anger:
                a += 1
            elif word in self.list_fear:
                f += 1

        # return normalized number of lexicon words
        return [(h / len(self.doc)), (s / len(self.doc)), (a / len(self.doc)), (f / len(self.doc))]

    # runs user input through neural net and returns detected emotions
    def get_emotions(self, user_message):
        # create different feature vectors
        self.features = self.get_input_features(user_message)
        self.lex_words = self.get_input_lexicon_words(user_message)
        self.all_features = [self.features[0], self.features[1], self.features[2], self.lex_words[0], self.lex_words[1], self.lex_words[2], self.lex_words[3], self.features[3]]

        # run input through net and round them
        self.emotions = self.net_lin_all_tweet(torch.Tensor(self.all_features)).tolist()
        # add disgust value to ouput (net does not give out disgust value)
        self.emotions.append(0)
        for index, value in enumerate(self.emotions):
            self.emotions[index] = round(value, 3)

        if user_message == "h" or user_message == "s" or user_message == "a" or user_message == "f" or user_message == "d" or user_message == "n":
            return self.get_emotions_debug(user_message)
        else:
            return {
                "input_emotions": np.asarray(self.emotions),
                "input_topics": self.get_topics(user_message),
            }

    def get_emotions_debug(self, user_message):
        if user_message == "h":
            return {
                "input_emotions": np.array([1.00, 0.00, 0.00, 0.00, 0.00]),
                "input_topics": np.array([1.00, 0.00, 0.00, 0.00, 0.00]),
            }
        elif user_message == "s":
            return {
                "input_emotions": np.array([0.00, 1.00, 0.00, 0.00, 0.00]),
                "input_topics": np.array([0.00, 1.00, 0.00, 0.00, 0.00]),
            }
        elif user_message == "a":
            return {
                "input_emotions": np.array([0.00, 0.00, 1.00, 0.00, 0.00]),
                "input_topics": np.array([0.00, 0.00, 1.00, 0.00, 0.00]),
            }
        elif user_message == "f":
            return {
                "input_emotions": np.array([0.00, 0.00, 0.00, 1.00, 0.00]),
                "input_topics": np.array([0.00, 0.00, 0.00, 1.00, 0.00]),
            }
        elif user_message == "d":
            return {
                "input_emotions": np.array([0.00, 0.00, 0.00, 0.00, 1.00]),
                "input_topics": np.array([0.00, 0.00, 0.00, 0.00, 1.00]),
            }
        elif user_message == "n":
            return {
                "input_emotions": np.array([0.00, 0.00, 0.00, 0.00, 0.00]),
                "input_topics": np.array([0.00, 0.00, 0.00, 0.00, 0.00]),
            }

    # analyzes and returns topics of the input using empath
    def get_topics(self, user_message):
        self.keyword_analysis = []
        self.topics_set = self.lexicon.analyze(user_message, categories=self.topic_keywords, normalize=True)
        for item in self.topic_keywords:
            self.keyword_analysis.append(round(self.topics_set[item], 2).__str__())
        return self.keyword_analysis
