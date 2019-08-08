import numpy as np
import empath
import random

class Classifier:
    def __init__(self, topic_keywords):
        self.lexicon = empath.Empath()
        self.keyword_analysis = []
        self.topic_keywords = topic_keywords
        self.topics_set = None

    # analyzes and returns general sentiment of the input
    def get_sentiment(self):
        pass

    # atm returns a list of random generated emotion values
    def get_emotions(self, user_message):
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


    # analyzses and returns general sentiment of the input
    def get_sentiment(self, input):
        pass

    # atm returns a list of random generated emotion values
   # def get_emotions(self, input):
    #    self.emotion_analysis = []
     #   self.emotion_analysis = [round(random.uniform(0, 1), 2),
      #                           round(random.uniform(0, 1), 2),
       #                          round(random.uniform(0, 1), 2),
        #                         round(random.uniform(0, 1), 2),
         #                        round(random.uniform(0, 1), 2)]
        #xreturn self.emotion_analysis


