# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import nltk
#from nltk.tokenize import sent_tokenize, word_tokenize
#from nltk.corpus import stopwords
#from nltk.stem import PorterStemmer
#from nltk.stem import WordNetLemmatizer
import spacy
import textblob

class Test:
    def __init__(self):
        self.example_sentence = "The examples are all taken from real texts in the World English " \
                                "Corpus. We don't sit at our desks and make them up – we research " \
                                "the language and take the most typical uses and embed them in the " \
                                "entries. Users of MacmillanDictionary.com can be sure that the " \
                                "language they encounter here is up-to-date, accurate, and reflects " \
                                "the language as it is used in the 21st century. Mr. Smith said so. "

        self.nlp = spacy.load("en")
        self.doc = self.nlp(self.example_sentence)
        self.spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS

        self.input_data = []
        self.sentences = list(self.doc.sents)
        self.named_entities = []

        for sentence in self.sentences:
            self.input_data.append(self.get_input_data(sentence))
            self.named_entities.append(self.get_named_entities(sentence))

    def get_input_data(self, sentence):
        sentence = "required, require"
        self.a =  [
            sentence,
            [(token, token.text, token.lemma_, token.pos_, token.shape_, token.sentiment)
                 for token in sentence if not token.is_stop],
            sentence.ents]
        print(self.a)
        return self.a

    def get_features(self):
        # Anzahl an Keywords/Affekt-Wörter (emotional getaggte?)
        # Auftauchen und Anzahl (evtl gewichtet)
        # n-Gramme

        # speech acts (vlt finde ich eine liste?)

        # Anzahl der Pronomen und Namen (named entities)
        # ausrufe/interjektionen
        # Flüche/Beledigungen
        # Diminutive/Augmentative
        pass




    #### DEBUG METHODS

    def print_input_data(self, input_data):
        for index, entry in enumerate(input_data):
            print("sentence: ", entry[0])
            print("data: ", entry[1])
            print("ents: ", entry[2])

    def get_sentiment(self, user_input):
        blob = textblob.TextBlob(user_input)
        print(blob.sentiment)


test = Test()