import numpy as np
import empath
import torch
import nltk
from models.neural_networks.neural_net import Lin_Net
from joblib import load
import gensim as gs


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
        self.SEQ_LEN_DICT = {"norm_tweet": 81, "norm_emotion": 32}
        self.NLP = nlp
        self.NUM_TOPICS_DICT = {"norm_tweet": 79, "norm_emotion": 186}

        # load network
        self.num_hidden_layers = 2
        self.hidden_dim = 256
        self.output_dim = 4
        self.input_dim = 0
        self.dataset = None
        self.seq = 0
        self.dic = None
        self.lda_model = None
        self.classifier_name = classifier_name
        self.classifier = None
        self.load_network(self.classifier_name)

    # loads the specified network architecture
    def load_network(self, classifier_name):
        print("loading network", classifier_name)
        # load network
        self.classifier_name = classifier_name
        if "logistic_regression" in classifier_name:
            self.classifier = load("../models/logistic_regression/" + classifier_name + ".joblib")
            print("loaded network", "../models/logistic_regression/" + classifier_name + ".joblib")
        elif "random_forests" in classifier_name:
            self.classifier = load("../models/random_forests/" + classifier_name + ".joblib")
        elif "net" in classifier_name:
            if "full" in classifier_name:
                self.input_dim = 8
            elif "lex" in classifier_name:
                self.input_dim = 4
            elif "topic" in classifier_name:
                self.input_dim = self.NUM_TOPICS_DICT["norm_emotion" if "emotion" in classifier_name else "norm_tweet"]
            self.classifier = Lin_Net(self.input_dim, self.output_dim, self.hidden_dim, self.num_hidden_layers)
            self.classifier.load_state_dict(torch.load("../models/neural_networks/"+classifier_name+".pt", map_location=torch.device('cpu')))
            print("loaded net")

        # load topic models
        if "topic" in classifier_name and "emotion" in classifier_name:
            self.dic = gs.corpora.Dictionary.load("../models/dictionary/norm_emotion_dictionary")
            self.lda_model = gs.models.ldamulticore.LdaMulticore.load("../models/topic_models/norm_emotion_ldamodel")
        elif "topic" in classifier_name and "tweet" in classifier_name:
            self.dic = gs.corpora.Dictionary.load("../models/dictionary/norm_tweet_dictionary")
            self.lda_model = gs.models.ldamulticore.LdaMulticore.load("../models/topic_models/norm_tweet_ldamodel")

        # set dataset name
        if "emotion" in classifier_name:
            self.dataset = "norm_emotion"
        elif "tweet" in classifier_name:
            self.dataset = "norm_tweet"

        # set sequence length for dataset
        self.seq = self.SEQ_LEN_DICT[self.dataset]

    # runs user input through classifier and returns detected emotions
    def get_emotions(self, user_input):
        # check if input is debug input
        if user_input in self.DEBUG_COMMANDS:
            return self.get_emotions_debug(user_input)
        else:
            # get feature vectors based on what classifier is loaded
            if "full" in self.classifier_name:
                features = self.extract_lex_features(user_input)
                features = features[0]
            elif "lex" in self.classifier_name:
                features = self.extract_lex_features(user_input)
                features = features[0][4:]
            elif "topics":
                num_topics = self.NUM_TOPICS_DICT[self.dataset]
                features = self.extract_topic_features(user_input, num_topics)

            # classify input by net or by simple classifier
            if "net" in self.classifier_name:
                emotions = self.classifier_name(features)
                input_class = max(emotions)
            else:
                emotions = self.classifier.predict_proba([features])
                emotions = emotions[0]
                input_class = self.classifier.predict([features])
                input_class = input_class[0]
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
        doc = self.NLP(self.split_punct(input_message))
        doc = self.NLP(" ".join([token.text for token in doc if not token.is_stop and token.pos != 103]))
        if len(doc) != 0:
            pos = [token.pos for token in doc]
            stems = [self.STEMMER.stem(token.text) for token in doc if token.pos != 97]
            emotion_words = self.get_emotion_words(stems, self.LIST_OF_LEXICA)
            feature_vec = [
                (len(doc) / self.seq), (sum([token.text.isupper() for token in doc]) / len(doc)),
                (len(doc.ents) / len(doc)), self.get_cons_punct_count(pos),
                emotion_words[0] / len(doc), emotion_words[1] / len(doc), emotion_words[2] / len(doc), emotion_words[3] / len(doc)]
            return feature_vec
        return [], [], []

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
