import numpy as np
import empath
import torch
import nltk
from itertools import groupby
from modules.neural_networks.net_lin_emotion_all import Net_Lin_Emotion_All
from joblib import load
from modules.neural_networks.net_lin_tweet_all import Net_Lin_Tweet_All
from modules.neural_networks.net_rnn_emotion import Net_Rnn_Emotion
from modules.neural_networks.net_rnn_tweet import Net_Rnn_Tweet


class Classifier:
    # constructor
    def __init__(self, classifier_name, list_of_lexica, nlp):
        # init constants
        self.POS_MAPPING = {84.0: 1, 85.0: 2, 86.0: 3, 87.0: 4, 89.0: 5, 90.0: 6, 91.0: 7, 92.0: 8, 93.0: 9,
                            94.0: 10, 95.0: 11, 96.0: 12, 97.0: 13, 99.0: 14, 100.0: 15, 101.0: 16, 03.0: 17, np.nan: 18}
        self.EMOTION_MAPPING = {"happiness": 0, "sadness": 1, "anger": 2, "fear": 3}
        self.DEBUG_COMMANDS = ["h", "s", "a", "f", "d"]
        self.LIN_NETS_NAMES = ["net_lin_emotion_all", "net_lin_tweet_all"]
        self.RNN_NETS_NAMES = ["net_rnn_emotion", "net_rnn_tweet"]
        self.STEMMER = nltk.stem.SnowballStemmer('english')
        self.LIST_OF_LEXICA = list_of_lexica
        self.SEQ_LEN = {"norm_tweet": 81, "norm_emotion": 32}
        self.NLP = nlp

        # load network
        self.classifier_name = classifier_name
        self.classifier = None
        self.load_network(self.classifier_name)

    # loads the specified network architecture
    def load_network(self, classifier_name):
        self.classifier_name = classifier_name
        if "logistic_regression" in classifier_name:
            self.classifier = load("../models/logistic_regression/" + classifier_name + ".joblib")
        elif "random_forests" in classifier_name:
            self.classifier = load("../models/random_forests/" + classifier_name + ".joblib")
        elif "net" in classifier_name:
            self.classifier = Net_Lin_Emotion_All()
            self.classifier.load_state_dict(torch.load("../nets/net_lin_emotion_all.pt"))

    def extract_features(self, list_of_lexica, input_message, seq_len):
        doc = self.NLP(split_punct(input_message))
        doc = self.NLP(" ".join([token.text for token in doc if not token.is_stop and token.pos != 103]))
        if len(doc) != 0:
            pos = [token.pos for token in doc]
            stems = [stemmer.stem(token.text) for token in doc if token.pos != 97]
            emotion_words = get_emotion_words(stems, list_of_lexica)
            feature_vec = [
                (len(doc) / seq_len), (sum([token.text.isupper() for token in doc]) / len(doc)),
                (len(doc.ents) / len(doc)), get_cons_punct_count(pos),
                emotion_words[0] / len(doc), emotion_words[1] / len(doc), emotion_words[2] / len(doc), emotion_words[3] / len(doc)]
            return feature_vec, pos, stems
        return [], [], []

    def split_punct(text):
        replacement = [(".", " . "), (",", " , "), ("!", " ! "), ("?", " ? ")]
        for k, v in replacement:
            text = text.replace(k, v)
        return text

    def get_emotion_words(stems, list_of_lexica):
        emotion_words = np.zeros(4)
        for index, lexicon in enumerate(list_of_lexica):
            for stem in stems:
                if stem in lexicon:
                    emotion_words[index] = emotion_words[index] + 1
        return emotion_words

    def get_cons_punct_count(pos):
        cons_punct_count = 0
        for index, item in enumerate(pos[:-1]):
            if item == 97 and item == pos[index + 1]:
                cons_punct_count += 1
        return cons_punct_count

    # runs user input through classifier and returns detected emotions
    def get_emotions(self, user_input):
        doc = self.NLP(user_input)
        # check if input is debug input
        if user_input in self.DEBUG_COMMANDS:
            return self.get_emotions_debug(user_input)
        elif self.network_name in self.LIN_NETS_NAMES:
            # create different feature vectors
            features = self.get_input_features(user_input, doc)
            lex_words = self.get_input_lexicon_words(doc)
            all_features = [features[0], features[1], features[2], lex_words[0], lex_words[1], lex_words[2], lex_words[3], features[3]]
            # run feature vector through network
            emotions = self.network(torch.Tensor(all_features)).tolist()
        elif self.network_name in self.RNN_NETS_NAMES:
            if self.network_name == self.RNN_NETS_NAMES[0]:
                input_tensor = torch.Tensor([self.get_pos_list(179, doc)])
            elif self.network_name == self.RNN_NETS_NAMES[1]:
                input_tensor = torch.Tensor([self.get_pos_list(85, doc)])
            hidden = self.network.init_hidden(1)
            emotions = self.network(input_tensor, hidden)[0][0].tolist()

        # add disgust value to output (net does not give out disgust value), round values
        emotions = [round(entry, 3) for entry in emotions]
        emotions.append(0)

        # return emotion package (emotions and topics)
        return {
            "input_emotions": np.asarray(emotions)
        }

    # get debug-emotion-values
    def get_emotions_debug(self, user_message):
        if user_message == "h":
            return {
                "input_emotions": np.array([1.00, 0.00, 0.00, 0.00, 0.00])
            }
        elif user_message == "s":
            return {
                "input_emotions": np.array([0.00, 1.00, 0.00, 0.00, 0.00])
            }
        elif user_message == "a":
            return {
                "input_emotions": np.array([0.00, 0.00, 1.00, 0.00, 0.00])
            }
        elif user_message == "f":
            return {
                "input_emotions": np.array([0.00, 0.00, 0.00, 1.00, 0.00])
            }
        elif user_message == "d":
            return {
                "input_emotions": np.array([0.00, 0.00, 0.00, 0.00, 1.00])
            }
        elif user_message == "n":
            return {
                "input_emotions": np.array([0.00, 0.00, 0.00, 0.00, 0.00])
            }
