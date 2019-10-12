import numpy as np
import empath
import torch
import nltk
from itertools import groupby
from modules.neural_networks.net_lin_emotion_all import Net_Lin_Emotion_All
from modules.neural_networks.net_lin_tweet_all import Net_Lin_Tweet_All
from modules.neural_networks.net_rnn_emotion import Net_Rnn_Emotion
from modules.neural_networks.net_rnn_tweet import Net_Rnn_Tweet


class Classifier:
    # constructor
    def __init__(self, network, list_happiness, list_sadness, list_anger, list_fear, nlp):
        # init constants
        self.POS_MAPPING = {84.0: 1, 85.0: 2, 86.0: 3, 87.0: 4, 89.0: 5, 90.0: 6, 91.0: 7, 92.0: 8, 93.0: 9,
                            94.0: 10, 95.0: 11, 96.0: 12, 97.0: 13, 99.0: 14, 100.0: 15, 101.0: 16, 03.0: 17, np.nan: 18}
        self.EMOTION_MAPPING = {"happiness": 0, "sadness": 1, "anger": 2, "fear": 3}
        self.DEBUG_COMMANDS = ["h", "s", "a", "f", "d"]
        self.LIN_NETS_NAMES = ["net_lin_emotion_all", "net_lin_tweet_all"]
        self.RNN_NETS_NAMES = ["net_rnn_emotion", "net_rnn_tweet"]
        self.STEMMER = nltk.stem.SnowballStemmer('english')
        self.LIST_HAPPINES = list_happiness
        self.LIST_SADNESS = list_sadness
        self.LIST_ANGER = list_anger
        self.LIST_FEAR = list_fear
        self.NLP = nlp

        # load network
        self.network_name = network
        self.network = None
        self.load_network(network)

    # loads the specified network architecture
    def load_network(self, network_name):
        self.network_name = network_name
        if network_name == "net_lin_emotion_all":
            self.network = Net_Lin_Emotion_All()
            self.network.load_state_dict(torch.load("../nets/net_lin_emotion_all.pt"))
        elif network_name == "net_lin_tweet_all":
            self.network = Net_Lin_Tweet_All()
            self.network.load_state_dict(torch.load("../nets/net_lin_tweet_all.pt"))
        # load network on gpu if available
        if torch.cuda.is_available():
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

    def get_input_features(self, user_input, doc):
        print("get input features")
        word_count = len(doc)
        upper_word_count = sum([token.text.isupper() for token in doc]) / len(doc)
        ent_word_count = len(doc.ents) / len(doc)
        cons_punct_count = 0

        # count the number of consecutive puctuation
        punct_groups = groupby(user_input)
        punct_groups = [(label, sum(1 for _ in group)) for label, group in punct_groups]
        punct_groups = [item for item in punct_groups if ("." in item or "!" in item or "?" in item or "," in item) and item[1] > 1]
        for item in punct_groups:
            cons_punct_count += item[1]

        return [word_count, upper_word_count, ent_word_count, cons_punct_count]

    # returns a pos-list of the user input
    def get_pos_list(self, seq_len, doc):
        pos_list = [token.pos for token in doc]
        # encode the pos-list
        for index, item in enumerate(pos_list):
            pos_list[index] = self.POS_MAPPING[pos_list[index]]
        # bring the list to to correct feature/sequence length
        pos_list.extend([18] * seq_len)
        pos_list = pos_list[:seq_len]

        return pos_list

    # get_input_lexicon_words
    def get_input_lexicon_words(self, doc):
        # stem the user input
        stemmed_input = [self.STEMMER.stem(token.text) for token in doc]
        lexicon_words = [0, 0, 0, 0]
        # count the occurances of words from lexica
        lexicon_words[0] = len([word for word in stemmed_input if word in self.LIST_HAPPINES])
        lexicon_words[1] = len([word for word in stemmed_input if word in self.LIST_SADNESS])
        lexicon_words[2] = len([word for word in stemmed_input if word in self.LIST_ANGER])
        lexicon_words[3] = len([word for word in stemmed_input if word in self.LIST_FEAR])

        # normalize number of lexicon words over sentence length
        for index, num in enumerate(lexicon_words):
            lexicon_words[index] = lexicon_words[index] / len(doc)

        # return normalized number of lexicon words
        return lexicon_words

    # runs user input through neural net and returns detected emotions
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
