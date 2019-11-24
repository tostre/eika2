import numpy as np
import torch
import nltk
from models.neural_networks.neural_net import Lin_Net
from joblib import load
import gensim as gs


class Classifier:
    # constructor
    def __init__(self, classifier_data, list_of_lexica, nlp):
        # init constants
        self.POS_MAPPING = {84.0: 1, 85.0: 2, 86.0: 3, 87.0: 4, 89.0: 5, 90.0: 6, 91.0: 7, 92.0: 8, 93.0: 9,
                            94.0: 10, 95.0: 11, 96.0: 12, 97.0: 13, 99.0: 14, 100.0: 15, 101.0: 16, 03.0: 17, np.nan: 18}
        self.EMOTION_MAPPING = {"happiness": 0, "sadness": 1, "anger": 2, "fear": 3}
        self.DEBUG_COMMANDS = ["h", "s", "a", "f", "d"]
        self.LIN_NETS_NAMES = ["net_lin_emotion_all", "net_lin_tweet_all"]
        self.RNN_NETS_NAMES = ["net_rnn_emotion", "net_rnn_tweet"]
        self.STEMMER = nltk.stem.SnowballStemmer('english')
        self.LIST_OF_LEXICA = list_of_lexica
        self.NLP = nlp
        self.NUM_TOPICS_DICT = {"norm_tweet": 79, "norm_emotion": 186}
        self.NET_FEATURE_SETS = {"norm_emotion": [0, 2, 4, 5, 6, 7], "norm_tweet": [0, 2, 3, 4, 5, 6, 7]}
        self.NET_FEATURE_SET_SIZES = {"norm_emotion": 6, "norm_tweet": 7}
        self.SEQ_LEN_DICT = {"norm_tweet": 81, "norm_emotion": 32}

        # load network
        self.dataset = None
        self.seq = 0
        self.dic = None
        self.lda_model = None
        self.classifier_data = classifier_data
        self.classifier = None
        self.load_network(classifier_data)

    # loads the specified network architecture
    def load_network(self, classifier_data):
        self.classifier_data = classifier_data
        # load data according to feature set
        if classifier_data[2] == "full":
            input_dim = self.NET_FEATURE_SET_SIZES["norm_emotion" if classifier_data[1] == "norm_emotion" else "norm_tweet"]
        elif classifier_data[2] == "lex":
            input_dim = 4
        elif classifier_data[2] == "topics":
            input_dim = self.NUM_TOPICS_DICT["norm_emotion" if classifier_data[1] == "norm_emotion" else "norm_tweet"]
            self.dic = gs.corpora.Dictionary.load("../models/dictionary/{}_dictionary".format(classifier_data[1]))
            self.lda_model = gs.models.ldamulticore.LdaMulticore.load("../models/topic_models/{}_ldamodel".format(classifier_data[1]))
        # load classifier
        if classifier_data[0] == "lr":
            self.classifier = load("../models/logistic_regression/{}_{}_logistic_regression.joblib".format(classifier_data[1], classifier_data[2]))
        elif classifier_data[0] == "rf":
            self.classifier = load("../models/random_forests/{}_{}_random_forests.joblib".format(classifier_data[1], classifier_data[2]))
        elif classifier_data[0] == "net":
            self.classifier = Lin_Net(input_dim, 4, 256, 2)
            self.classifier.load_state_dict(torch.load("../models/neural_networks/net_lin_{}({})_5000.pt".format(classifier_data[1], classifier_data[2]), map_location=torch.device('cpu')))

    # runs user input through classifier and returns detected emotions
    def get_emotions(self, user_input):
        # check if input is debug input
        if user_input in self.DEBUG_COMMANDS:
            return self.get_emotions_debug(user_input)
        else:
            # get feature vectors based on what classifier is loaded
            if self.classifier_data[0] == "net" and self.classifier_data[2] == "full":
                features = self.extract_lex_features(user_input)
                features = [features[i] for i in self.NET_FEATURE_SETS[self.classifier_data[1]]]
            elif self.classifier_data[2] == "lex":
                features = self.extract_lex_features(user_input)
                features = features[-4:]
            elif self.classifier_data[2] == "topics":
                num_topics = self.NUM_TOPICS_DICT[self.dataset]
                features = self.extract_topic_features(user_input, num_topics)
            else:
                features = self.extract_lex_features(user_input)

            # classify with classifier
            if self.classifier_data[0] == "lr" or self.classifier_data[0] == "rf":
                emotions = self.classifier.predict_proba([features])
                emotions = emotions[0]
                input_class = self.classifier.predict([features])
                input_class = input_class[0]
            elif self.classifier_data[0] == "net":
                emotions = self.classifier(torch.Tensor(np.asarray(features)).float()).tolist()
                input_class = emotions.index(max(emotions))

            # add disgust value to output (net does not give out disgust value), round values
            emotions = [round(entry, 3) for entry in emotions]
            emotions.append(0)
            # return emotion package
            return {
                "input_emotions": np.asarray(emotions),
                "input_class": input_class
            }

    # transorms user input into a feature vector
    def extract_lex_features(self, input_message):
        seq_len = self.SEQ_LEN_DICT[self.classifier_data[1]]
        doc = self.NLP(self.split_punct(input_message))
        doc = self.NLP(" ".join([token.text for token in doc if not token.is_stop and token.pos != 103]))
        feature_vec = [range(0, len(self.NET_FEATURE_SETS[self.classifier_data[1]]))]
        if len(doc) != 0:
            pos = [token.pos for token in doc]
            stems = [self.STEMMER.stem(token.text) for token in doc if token.pos != 97]
            emotion_words = self.get_emotion_words(stems, self.LIST_OF_LEXICA)
            feature_vec = [
                len(doc) / seq_len, (sum([token.text.isupper() for token in doc]) / len(doc)),
                (len(doc.ents) / len(doc)), self.get_cons_punct_count(pos),
                emotion_words[0] / len(doc), emotion_words[1] / len(doc), emotion_words[2] / len(doc), emotion_words[3] / len(doc)]
        return feature_vec

    def extract_topic_features(self, input_message, num_topics):
        # tokenizing, stemming, converting to vec, etc.
        doc = self.NLP(self.split_punct(input_message))
        doc = self.NLP(" ".join([token.text for token in doc if not token.is_stop and token.pos != 103]))
        sentence = [self.STEMMER.stem(token.text) for token in doc if token.pos != 97]
        sentence = self.dic.doc2bow(sentence)
        # get the topics from the document, write probability per topics in list
        topics = self.lda_model.get_document_topics(sentence, minimum_probability=0.0)
        topic_vec = [topics[i][1] for i in range(num_topics)]
        return topic_vec

    def split_punct(self, text):
        replacement = [(".", " . "), (",", " , "), ("!", " ! "), ("?", " ? ")]
        for k, v in replacement:
            text = text.replace(k, v)
        return text

    def get_emotion_words(self, stems, list_of_lexica):
        emotion_words = np.zeros(4)
        for index, lexicon in enumerate(list_of_lexica):
            for stem in stems:
                if stem in lexicon:
                    emotion_words[index] = emotion_words[index] + 1
        return emotion_words

    def get_cons_punct_count(self, pos):
        pos.append(0)
        cons_punct_count = 0
        for index, item in enumerate(pos[:-1]):
            if item == 97 and item == pos[index + 1]:
                cons_punct_count += 1
        return cons_punct_count

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
