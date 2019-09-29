import chatterbot as cb
from chatterbot.trainers import ChatterBotCorpusTrainer
import logging


# this analyzes user inputs and generates a response
class Bot:
    def __init__(self):
        # this is so i dont get a minor error message every turn
        logger = logging.getLogger()
        logger.setLevel(logging.CRITICAL)

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
    def respond(self, user_message, replace_words):
        if user_message == "h" or user_message == "s" or user_message == "a" or user_message == "f" or user_message == "d" or user_message == "n":
            return self.respond_debug(user_message)
        else:
            self.response = self.bot.get_response(user_message)
            # replace certain words with synonyms
            for key, synonym in replace_words.items():
                return self.response.replace(key, synonym)

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

# @todo die mthide zum synonym-finden hierher verschienen, dann gucken, dass man oben bei der replace-methode den text rausbekommt und nicht das statement