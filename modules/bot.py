import chatterbot as cb
from chatterbot.trainers import ChatterBotCorpusTrainer
import logging
from nltk.corpus import wordnet
import random


# this analyzes user inputs and generates a response
class Bot:
    def __init__(self, lex_happiness, lex_sadness, lex_anger, lex_fear, list_happiness, list_sadness, list_anger, list_fear,
                 list_happiness_adj, list_sadness_adj, list_anger_adj, list_fear_adj, nlp):
        # this is so i dont get a minor error message every turn
        logger = logging.getLogger()
        logger.setLevel(logging.CRITICAL)

        # assign emotion lexica and lists
        self.lex_happiness = lex_happiness
        self.lex_sadness = lex_sadness
        self.lex_anger = lex_anger
        self.lex_fear = lex_fear
        self.lex_happiness_adj = list_happiness_adj
        self.lex_sadness_adj = list_sadness_adj
        self.lex_anger_adj = list_anger_adj
        self.lex_fear_adj = list_fear_adj
        self.list_happiness = list_happiness
        self.list_sadness = list_sadness
        self.list_anger = list_anger
        self.list_fear = list_fear

        # initialise variables
        self.is_debug_input = False
        self.debug_commands = ["h", "s", "a", "f", "d"]
        self.nlp = nlp
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
        if user_message in self.debug_commands:
            return self.respond_debug(user_message)
        else:
            self.is_debug_input = False
            self.response = self.bot.get_response(user_message)

            self.response_package = {
                "response": self.response,
                "response_confidence": self.response.confidence,
                "is_debug_input": False
            }
            return self.response_package

    # Return generic respinse dicts for debug purposes
    def respond_debug(self, user_message):
        return {"response": "debug_input", "response_confidence": 100, "is_debug_input": True}

    def modify_output(self, response_package, highest_emotion, highest_score):
        if self.is_debug_input:
            # start modifier-functions with appropriate dataset
            if highest_emotion == 0:
                bot_output = self.get_synonyms(response_package["response"].text, self.lex_happiness, self.list_happiness, highest_score)
                bot_output = self.insert_adjectives(bot_output, self.lex_happiness_adj, highest_score)
            elif highest_emotion == 1:
                bot_output = self.get_synonyms(response_package["response"].text, self.lex_sadness, self.list_sadness, highest_score)
                bot_output = self.insert_adjectives(bot_output, self.lex_sadness_adj, highest_score)
            elif highest_emotion == 2:
                bot_output = self.get_synonyms(response_package["response"].text, self.lex_anger, self.list_anger, highest_score)
                bot_output = self.insert_adjectives(bot_output, self.lex_anger_adj, highest_score)
            elif highest_emotion == 3:
                bot_output = self.get_synonyms(response_package["response"].text, self.lex_fear, self.list_fear, highest_score)
                bot_output = self.insert_adjectives(bot_output, self.lex_fear_adj, highest_score)

            return {
                "response": bot_output,
                "response_confidence": response_package["response_confidence"]
            }

        else:
            return {
                "response": response_package["response"],
                "response_confidence": response_package["response_confidence"]
            }

    # looks for synonyms in a user input and returns them with an intensity score
    def get_synonyms(self, bot_output, lexicon, lexicon_list, highest_score):
        # create list of all nouns in the user input and their synonyms
        doc = self.nlp(bot_output)
        nouns = [token.text for token in doc if token.pos == 92]
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
                if synonym in lexicon_list:
                    row = lexicon.loc[lexicon["text"] == synonym]
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
        # replace words in bot_output
        bot_output_list = bot_output.split(" ")
        for index, word in enumerate(bot_output_list):
            if word in switch_words:
                bot_output_list[index] = switch_words[bot_output_list[index]]
        bot_output = " ".join(bot_output_list)

        return bot_output

    # insert adjectives into the output:
    def insert_adjectives(self, bot_output, lexicon_adj, highest_score):
        # make lists from lexicon
        lexicon_text_list = lexicon_adj["text"].tolist()
        lexicon_intensity_list = lexicon_adj["intensity"].tolist()
        doc = self.nlp(bot_output)
        noun_indices = []
        new_adjectives = {}
        # Create dict from text and their pos-tags
        bot_output_dict = {
            "text": [token.text for token in doc],
            "pos": [token.pos for token in doc],
            "dep": [token.dep_ for token in doc]
        }

        # count how many nouns there are in the sentence
        if bot_output_dict["pos"][0] == 92:
            num_nouns = bot_output_dict["pos"].count(92) - 1
        else:
            num_nouns = bot_output_dict["pos"].count(92)
        # calc how many nouns should get an adjective (depending on emotion score)
        num_nouns = round(num_nouns * highest_score)

        # select which nouns can receive an adjective
        if num_nouns > 0:
            noun_indices = [i for i, item in enumerate(bot_output_dict["text"])
                            if bot_output_dict["pos"][i] == 92
                            and (bot_output_dict["dep"][i] == "nsubj"
                                 or bot_output_dict["dep"][i] == "dobj"
                                 or bot_output_dict["dep"][i] == "pobj")]

        # randomly select num_nouns amount of nouns that will receive an adjective
        random.shuffle(noun_indices)
        indices_to_replace = noun_indices[:num_nouns]
        # if noun is in first place in sentence, remove it
        if 0 in indices_to_replace:
            indices_to_replace.remove(0)


        # calc differences between highest score and intensities of words
        scores_dict = [round(abs(intensity - highest_score), 3) for intensity in lexicon_intensity_list]

        # get words with the lowest difference (number of noun_indices)
        for number in indices_to_replace:
            new_adjective_index = scores_dict.index(min(scores_dict))
            scores_dict[new_adjective_index] = 99
            new_adjectives[number] = lexicon_text_list[new_adjective_index]

        # insert adjectives into bot_output_phrase
        for key, item in new_adjectives.items():
            bot_output_dict["text"].insert(key, item)

        return " ".join(bot_output_dict["text"])
