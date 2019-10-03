import numpy as np
import empath
import torch
import spacy
import nltk
import pandas as pd
from spellchecker import SpellChecker
from nltk.corpus import wordnet
from modules.neural_networks.net_lin_emotion_all import Net_Lin_Emotion_All
from modules.neural_networks.net_lin_tweet_all import Net_Lin_Tweet_All
from modules.neural_networks.net_rnn_emotion import Net_Rnn_Emotion
from modules.neural_networks.net_rnn_tweet import Net_Rnn_Tweet


class Classifier:

    def __init__(self, topic_keywords, network, lex_happiness, lex_sadness, lex_anger, lex_fear, list_happiness, list_sadness, list_anger, list_fear):
        self.pos_encoding = {84.0: 1, 85.0: 2, 86.0: 3, 87.0: 4, 89.0: 5, 90.0: 6, 91.0: 7, 92.0: 8, 93.0: 9,
                             94.0: 10, 95.0: 11, 96.0: 12, 97.0: 13, 99.0: 14, 100.0: 15, 101.0: 16, 03.0: 17, np.nan: 18}

        self.nlp = spacy.load("en_core_web_lg")
        self.lexicon = empath.Empath()
        self.keyword_analysis = []
        self.topic_keywords = topic_keywords
        self.topics_set = None
        self.emotion_mapping = {"happiness": 0, "sadness": 1, "anger": 2, "fear": 3}
        self.stemmer = nltk.stem.SnowballStemmer('english')
        # load network
        self.cuda_available = torch.cuda.is_available()
        self.network_name = network
        self.network = None
        self.load_network(network)
        # assign emotion lexica
        self.lex_happiness = lex_happiness
        self.lex_sadness = lex_sadness
        self.lex_anger = lex_anger
        self.lex_fear = lex_fear
        self.list_happiness = list_happiness
        self.list_sadness = list_sadness
        self.list_anger = list_anger
        self.list_fear = list_fear
        # init variables for extracting features from user input
        self.spell = SpellChecker()
        self.corrected = []
        self.uncorrected = []
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

    # loads the specified network architecture
    def load_network(self, network_name):
        self.network_name = network_name
        if network_name == "net_lin_emotion_all":
            self.network = Net_Lin_Emotion_All()
            self.network.load_state_dict(torch.load("../nets/net_lin_emotion_all.pt"))
        elif network_name == "net_lin_tweet_all":
            self.network = Net_Lin_Tweet_All()
            self.network.load_state_dict(torch.load("../nets/net_lin_tweet_all.pt"))
        if self.cuda_available:
            if network_name == "net_rnn_emotion":
                self.network = Net_Rnn_Emotion()
                self.network.load_state_dict(torch.load("../nets/net_rnn_emotion.pt"))
            elif network_name == "net_rnn_tweet":
                self.network = Net_Rnn_Tweet()
                self.network.load_state_dict(torch.load("../nets/net_rnn_tweet.pt"))
        else:
            if network_name == "net_rnn_emotion":
                self.network = Net_Rnn_Emotion()
                self.network.load_state_dict(torch.load("../nets/net_rnn_emotion.pt", map_location=lambda storage, loc: storage))
            elif network_name == "net_rnn_tweet":
                self.network = Net_Rnn_Tweet()
                self.network.load_state_dict(torch.load("../nets/net_rnn_tweet.pt", map_location=lambda storage, loc: storage))

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

    # returns a pos-list of the user input
    def get_pos_list(self, user_input, seq_len):
        self.doc = self.nlp(user_input)
        self.pos_list = [token.pos for token in self.doc]
        # encode the pos-list
        for index, item in enumerate(self.pos_list):
            self.pos_list[index] = self.pos_encoding[self.pos_list[index]]
        # bring the list to to correct feature/sequence length
        if len(self.pos_list) > seq_len:
            self.pos_list = self.pos_list[:seq_len]
        else:
            while len(self.pos_list) < seq_len:
                self.pos_list.append(18)
        return self.pos_list

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
        # check if input is debug input
        if user_message == "h" or user_message == "s" or user_message == "a" or user_message == "f" or user_message == "d" or user_message == "n":
            return self.get_emotions_debug(user_message)
        else:
            self.user_message = self.correct_input(user_message)
            # run input through net and round them
            if self.network_name == "net_lin_emotion_all" or self.network_name == "net_lin_tweet_all":
                # create different feature vectors
                self.features = self.get_input_features(self.user_message)
                self.lex_words = self.get_input_lexicon_words(self.user_message)
                self.all_features = [self.features[0], self.features[1], self.features[2], self.lex_words[0], self.lex_words[1], self.lex_words[2], self.lex_words[3], self.features[3]]
                # run feature vector through network
                self.emotions = self.network(torch.Tensor(self.all_features)).tolist()
                self.highest_emotion = self.emotions.index(min(self.emotions))
                self.highest_emotion_score = max(self.emotions)
            elif self.network_name == "net_rnn_emotion":
                input = torch.Tensor([self.get_pos_list(self.user_message, 85)])
                hidden = self.network.init_hidden(1)
                self.emotions = self.network(input, hidden)[0][0].tolist()
                self.highest_emotion = self.emotions.index(min(self.emotions))
                self.highest_emotion_score = max(self.emotions)
            elif self.network_name == "net_rnn_tweet":
                input = torch.Tensor([self.get_pos_list(self.user_message, 179)])
                hidden = self.network.init_hidden(1)
                self.emotions = self.network(input, hidden)[0][0].tolist()
                self.highest_emotion = self.emotions.index(min(self.emotions))
                self.highest_emotion_score = max(self.emotions)

            # add disgust value to output (net does not give out disgust value), round values
            self.emotions.append(0)
            for index, value in enumerate(self.emotions):
                self.emotions[index] = round(value, 3)

            # return emotion package (emotions and topics)
            return {
                "input_emotions": np.asarray(self.emotions),
                "input_topics": self.get_topics(self.user_message)
            }

    # get debug-emotion-values
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

    # spellchecks and corrects the user's input
    def correct_input(self, user_input):
        # make list of all words
        words = user_input.split(" ")
        unknown_words = self.spell.unknown(words)
        # replace all unknown words
        for word in unknown_words:
            print("correction: ", word, self.spell.correction(word))
            user_input = user_input.replace(word, self.spell.correction(word))

        return user_input

    # looks for synonyms in a user input and returns them with an intensity score
    def get_synonyms(self, user_input, highest_emotion, emotion_score):
        # prepare needed datasets
        if highest_emotion == 0:
            syn_lex = self.lex_happiness
            syn_list = self.list_happiness
        elif highest_emotion == 1:
            syn_lex = self.lex_sadness
            syn_list = self.list_sadness
        elif highest_emotion == 2:
            syn_lex = self.lex_anger
            syn_list = self.list_anger
        elif highest_emotion == 3:
            syn_lex = self.lex_fear
            syn_list = self.list_fear

        # create list of all nouns in the user input and their synonyms
        syn_doc_1 = self.nlp(user_input)
        nouns = [token.text for token in syn_doc_1 if token.pos == 92]
        synonyms = []
        for index, noun in enumerate(nouns):
            synonyms.append([noun, {}])
            for syn in wordnet.synsets(noun):
                for l in syn.lemmas():
                    # replace underscore
                    found_syn = l.name()
                    syn_doc_2 = self.nlp(l.name())
                    # if word is noun, keep it
                    if syn_doc_2[0].pos == 92 and not "_" in found_syn:
                        synonyms[index][1][found_syn] = 0.0

        # look for emotion-intensity scores in lexicon for found synonyms
        for index, item in enumerate(synonyms):
            for synonym, score in item[1].items():
                if synonym in syn_list:
                    row = syn_lex.loc[syn_lex["text"] == synonym]
                    intensity = row["intensity"]
                    intensity = intensity.tolist()[0]
                    # berechnen die distanz zum emotion_input
                    distance = round(abs(intensity - emotion_score), 3)
                    synonyms[index][1][synonym] = distance
                else:
                    synonyms[index][1][synonym] = 2

        # create map with the highest rating synonyms
        switch_words = {}
        for index, item in enumerate(synonyms):
            scores_dict = item[1]
            switch_words[item[0]] = min(scores_dict, key=scores_dict.get)
        print("synonyms found", switch_words)
        return switch_words