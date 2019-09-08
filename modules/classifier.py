import numpy as np
import empath
import random
import torch
import spacy
import nltk


class Classifier:
    def __init__(self, topic_keywords):
        self.lexicon = empath.Empath()
        self.keyword_analysis = []
        self.topic_keywords = topic_keywords
        self.topics_set = None
        self.nlp = spacy.load("en_core_web_lg")
        self.emotion_mapping = {"happiness": 0, "sadness": 1, "anger": 2, "fear": 3}
        self.stemmer = nltk.stem.SnowballStemmer('english')
        self.net_lin = torch.load("example")

    # TODO die methode fertig machen
    # searches for features in user input
    def get_input_features(self, user_input):
        # word_count, upper_word_count, ent_word_count, cons_punct_count
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

    # TODO die methode fertig machen
    # get_input_lexicon_words
    def get_input_lexicon_words(self, user_input):
        h, s, a, f = 0
        h_count, s_count, a_count, f_count = 0
        self.doc = self.nlp(user_input)
        stems = []
        stems = [self.stemmer.stem(token.text) for token in self.doc]

        # load lexica # # # # # # #

        for item in stems:
            if item in self.list_happiness:
                h += 1
            elif item in self.list_sadness:
                s += 1
            elif item in self.list_anger:
                a += 1
            elif item in self.list_fear:
                f += 1

        h_count = (h / len(self.doc))
        s_count = (s / len(self.doc))
        a_count = (a / len(self.doc))
        f_count = (f / len(self.doc))

        return [h_count, s_count, a_count, f_count]

    # analyzes and returns general sentiment of the input
    def get_sentiment(self):
        pass

    # TODO die methode so erweitern, dass sie den user input durch das netz jagt (netz am besten
    # im konstruktor der methode schon laden
    # atm returns a list of random generated emotion values
    def get_emotions(self, user_message):
        self.net_in = torch.Tensor(user_message)
        self.output = self.net_lin(self.net_in)

        if user_message == "h" or user_message == "s" or user_message == "a" or user_message == "f" or user_message == "d" or user_message == "n":
            return self.get_emotions_debug(user_message)
        else:
            return {
                "input_emotions": np.round(np.random.rand(5), 3),
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
