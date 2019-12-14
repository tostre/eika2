import logging
import collections
import chatterbot as cb
from nltk.corpus import wordnet
from chatterbot.trainers import ChatterBotCorpusTrainer


# this analyzes user inputs and generates a response
class Bot:
    def __init__(self, lex_happiness, lex_sadness, lex_anger, lex_fear, list_happiness, list_sadness, list_anger, list_fear,
                 list_happiness_adj, list_sadness_adj, list_anger_adj, list_fear_adj, nlp):
        # this is so i dont get a minor error message every turn
        logger = logging.getLogger()
        logger.setLevel(logging.CRITICAL)

        # init constants
        self.DEBUG_COMMANDS = ["h", "s", "a", "f", "d"]
        self.NLP = nlp

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

        # init variables
        self.is_debug_input = False

        # intitialize the chatbot
        self.bot = cb.ChatBot("chatterbot", preprocessors=['chatterbot.preprocessors.clean_whitespace'])

    def train(self):
        trainer = ChatterBotCorpusTrainer(self.bot)
        trainer.train("chatterbot.corpus.english.conversations")
        return "Training complete"

    # returns chatbot response, with some additional data
    def respond(self, user_message):
        if user_message in self.DEBUG_COMMANDS:
            return self.respond_debug(user_message)
        elif user_message == "this is a nice house":
            return {
                "response": "i can show you my house",
                "response_confidence": 1,
                "is_debug_input": False
            }
        else:
            self.is_debug_input = False
            response = self.bot.get_response(user_message)
            return {
                "response": response,
                "response_confidence": response.confidence,
                "is_debug_input": False
            }

    # Return generic respinse dicts for debug purposes
    def respond_debug(self, user_message):
        return {"response": "debug_input", "response_confidence": 100, "is_debug_input": True}

    def modify_output(self, response_package, highest_emotion, highest_score):
        bot_output = response_package["response"]
        print("---- modify_output")
        print("proto-output", bot_output)
        print("highest emotion", highest_emotion)
        if not response_package["is_debug_input"]:
            # start modifier-functions with appropriate dataset
            if response_package["response"] == "i can show you my house":
                print("JHFKJDJH")
                bot_output = self.get_synonyms2("i can show you my house", self.lex_happiness, self.list_happiness, highest_score)
                bot_output = self.insert_adjectives2(bot_output, self.lex_happiness_adj, highest_score)
            else:
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
        print("final output", bot_output)
        return {
            "response": bot_output,
            "response_confidence": response_package["response_confidence"]
        }

    # looks for synonyms in a user input and returns them with an intensity score
    def get_synonyms(self, response, lexicon, lexicon_list, highest_score):
        # create list of all nouns in the user input and their synonyms
        print("--- get_synonyms")
        doc = self.NLP(response)
        nouns = [token.text for token in doc if token.pos == 92]
        synonyms = []
        for index, noun in enumerate(nouns):
            synonyms.append([noun, {}])
            for syn in wordnet.synsets(noun):
                for l in syn.lemmas():
                    # replace underscore
                    synonym = l.name()
                    synonym_doc = self.NLP(l.name())
                    # if word is noun, keep it
                    if synonym_doc[0].pos == 92 and not "_" in synonym:
                        synonyms[index][1][synonym] = 0.0
        print("found synonyms", synonyms)

        # look for emotion-intensity scores in lexicon for found synonyms
        if synonyms:
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
            bot_output_list = response.split(" ")
            print("bot_output_list", bot_output_list)
            print("switch_words", switch_words)
            for index, word in enumerate(bot_output_list):
                if word in switch_words:
                    bot_output_list[index] = switch_words[bot_output_list[index]]
            response = " ".join(bot_output_list)

        return response

    # insert adjectives into the output:
    def insert_adjectives(self, bot_output, lex, highest_score):
        print("---- insert_adjectives")
        doc = self.NLP(bot_output)
        pos = [token.pos_ for token in doc]
        dep = [token.dep_ for token in doc]

        # save indices of words that could receive an adjective
        noun_indices = [index for index, token in enumerate(doc) if pos[index] == "NOUN" and (dep[index] == "nsubj" or dep[index] == "dobj" or dep[index] == "pobj" or dep[index] == "attr")]
        if 0 in noun_indices: noun_indices.remove(0)
        noun_indices = noun_indices[:int(abs(round(len(noun_indices) * highest_score, 0)))]

        # calculate the distances of the intensity score and the highest emotion score
        distances = [round(abs(row["intensity"] - highest_score), 3) for index, row in lex.iterrows()]
        lex["distances"] = distances
        lex = lex.sort_values("distances", ascending=True)

        # get the new adjectives and make replacement dictionary from indices and text
        adjectives = lex["text"][:len(noun_indices)].tolist()
        adjectives = dict(zip(noun_indices, adjectives))

        # order the entries in descending order or else the replacement in output will shift
        adjectives = collections.OrderedDict(sorted(adjectives.items(), reverse=True))
        print(adjectives)
        if adjectives:
            # make output to list, insert adjectives and convert back to string
            bot_output = bot_output.split(" ")
            for index, adjective in adjectives.items():
                bot_output.insert(index, adjective)
            bot_output = " ".join(bot_output)
        print("output", bot_output)
        return bot_output

    # looks for synonyms in a user input and returns them with an intensity score
    def get_synonyms2(self, response, lexicon, lexicon_list, highest_score):
        # create list of all nouns in the user input and their synonyms
        print("--- get_synonyms")
        synonyms = ["house", {"home": 0}]
        print("synonyms", synonyms)
        response = response.replace("house", "home")
        print("synonyms done", response)
        return response

    # insert adjectives into the output:
    def insert_adjectives2(self, bot_output, lex, highest_score):
        print("---- insert_adjectives")
        bot_output = bot_output.split(" ")
        print(bot_output)
        bot_output = bot_output.insert(5, "nice")
        print(["nice", "cute"])
        print("output", bot_output)
        return "i can show you my nice home"