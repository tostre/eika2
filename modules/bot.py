import chatterbot as cb
from chatterbot.trainers import ChatterBotCorpusTrainer
import logging
import spacy
from nltk.corpus import wordnet


# this analyzes user inputs and generates a response
class Bot:
    def __init__(self, lex_happiness, lex_sadness, lex_anger, lex_fear, list_happiness, list_sadness, list_anger, list_fear):
        # this is so i dont get a minor error message every turn
        logger = logging.getLogger()
        logger.setLevel(logging.CRITICAL)

        # assign emotion lexica and lists
        self.lex_happiness = lex_happiness
        self.lex_sadness = lex_sadness
        self.lex_anger = lex_anger
        self.lex_fear = lex_fear
        self.list_happiness = list_happiness
        self.list_sadness = list_sadness
        self.list_anger = list_anger
        self.list_fear = list_fear

        # initialise variables
        self.nlp = spacy.load("en_core_web_lg")
        self.response = None
        self.response_package = {}
        self.bot_state_package = {}
        self.bot = cb.ChatBot(
            "chatterbot",
            preprocessors=[
                'chatterbot.preprocessors.clean_whitespace'
            ]
        )

        self.train()

    def train(self):
        trainer = ChatterBotCorpusTrainer(self.bot)
        trainer.train("chatterbot.corpus.english.conversations")
        return "Training complete"

    # returns chatbot response, with some additional data
    def respond(self, user_message):
        if user_message == "h" or user_message == "s" or user_message == "a" or user_message == "f" or user_message == "d" or user_message == "n":
            return self.respond_debug(user_message)
        else:
            self.response = self.bot.get_response(user_message)

            self.response_package = {
                "response": self.response,
                "response_confidence": self.response.confidence
            }
            return self.response_package

    # Return generic respinse dicts for debug purposes
    def respond_debug(self, user_message):
        if user_message == "h":
            return {
                "response": "Debug input: h",
                "response_confidence": 100
            }
        elif user_message == "s":
            return {
                "response": "Debug input: s",
                "response_confidence": 100
            }
        elif user_message == "a":
            return {
                "response": "Debug input: a",
                "response_confidence": 100
            }
        elif user_message == "f":
            return {
                "response": "Debug input: f",
                "response_confidence": 100
            }
        elif user_message == "d":
            return {
                "response": "Debug input: d",
                "response_confidence": 100
            }
        elif user_message == "n":
            return {
                "response": "Debug input: n",
                "response_confidence": 100
            }

    # looks for synonyms in a user input and returns them with an intensity score
    def get_synonyms(self, response_package, highest_emotion, highest_score):
        bot_output = response_package["response"].text
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
        syn_doc_1 = self.nlp(bot_output)
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
                    distance = round(abs(intensity - highest_score), 3)
                    synonyms[index][1][synonym] = distance
                else:
                    synonyms[index][1][synonym] = 2

        # create map with the highest rating synonyms
        switch_words = {}
        for index, item in enumerate(synonyms):
            scores_dict = item[1]
            switch_words[item[0]] = min(scores_dict, key=scores_dict.get)
        print("synonyms found", switch_words)
        print(bot_output)
        # replace words in bot_output
        bot_output_list = bot_output.split(" ")
        for index, word in enumerate(bot_output_list):
            if word in switch_words:
                bot_output_list[index] = switch_words[bot_output_list[index]]
        bot_output = " ". join(bot_output_list)

        return {
            "response": bot_output,
            "response_confidence": response_package["response_confidence"]
        }
        return switch_words