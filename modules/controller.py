from modules.frame import Frame
from modules.bot import Bot
from modules.character import Character
from modules.classifier import Classifier
from modules.character_manager import Character_Manager
import configparser
import logging
import pandas as pd
import spacy
from spellchecker import SpellChecker


# controller
# class, every subsystem is initialized, and passed down to the classes that need them here
# this enables the system to be highly modular, every component (classifier, bot, character) can be switched
class Controller:
    def __init__(self):
        # create default personalities
        print("loading characters")
        cm = Character_Manager()
        cm.save("character_default")
        cm.save("character_stable")
        cm.save("character_empathetic")
        cm.save("character_irascible")

        # set up logging
        logging.basicConfig(level=logging.INFO, filename='../logs/app.log', filemode="w", format='%(asctime)s %(name)s/%(levelname)s - - %(message)s', datefmt='%d.%m.%y %H:%M:%S')
        self.logger = logging.getLogger("controller")
        self.logger.setLevel(logging.INFO)

        # read config file and save values in variables
        print("loading config")
        self.config = configparser.ConfigParser()
        self.config.read("../config/config.ini")
        self.botname = self.config.get("default", "botname")
        self.username = self.config.get("default", "username")
        self.network_name = self.config.get("net", "network")

        # initialize emotional variables
        print("loading lexica")
        self.lex_happiness = pd.read_csv("../lexica/clean_happiness.csv", delimiter=",", dtype={"text": str, "affect": str, "stems": str}, float_precision='round_trip')
        self.lex_sadness = pd.read_csv("../lexica/clean_sadness.csv", delimiter=",", dtype={"text": str, "affect": str, "stems": str}, float_precision='round_trip')
        self.lex_anger = pd.read_csv("../lexica/clean_anger.csv", delimiter=",", dtype={"text": str, "affect": str, "stems": str}, float_precision='round_trip')
        self.lex_fear = pd.read_csv("../lexica/clean_fear.csv", delimiter=",", dtype={"text": str, "affect": str, "stems": str}, float_precision='round_trip')
        self.list_happiness = self.lex_happiness["stems"].tolist()
        self.list_sadness = self.lex_sadness["stems"].tolist()
        self.list_anger = pd.Series(self.lex_anger["stems"].tolist())
        self.list_fear = self.lex_fear["stems"].tolist()
        self.lex_happiness_adj = pd.read_csv("../lexica/clean_happiness_adj.csv", delimiter=",", dtype={"text": str, "intensity": float}, float_precision='round_trip')
        self.lex_sadness_adj = pd.read_csv("../lexica/clean_happiness_adj.csv", delimiter=",", dtype={"text": str, "intensity": float}, float_precision='round_trip')
        self.lex_anger_adj = pd.read_csv("../lexica/clean_happiness_adj.csv", delimiter=",", dtype={"text": str, "intensity": float}, float_precision='round_trip')
        self.lex_fear_adj = pd.read_csv("../lexica/clean_happiness_adj.csv", delimiter=",", dtype={"text": str, "intensity": float}, float_precision='round_trip')

        # initialize ml-variables
        self.nlp = spacy.load("en_core_web_lg")
        self.spell = SpellChecker()

        # create bot, responsible for generating answers and classifier, for analysing the input
        self.character = Character(self.config.getboolean("default", "firstlaunch"))
        self.classifier = Classifier(self.network_name, self.list_happiness, self.list_sadness, self.list_anger, self.list_fear, self.nlp)
        self.bot = Bot(self.lex_happiness, self.lex_sadness, self.lex_anger, self.lex_fear, self.list_happiness, self.list_sadness, self.list_anger,
                       self.list_fear, self.lex_happiness_adj, self.lex_sadness_adj, self.lex_anger_adj, self.lex_fear_adj, self.nlp)

        # create frame and update widgets with initial values
        print("initialising gui")
        self.frame = Frame(self.botname, self.username, self.character.get_emotional_state(), self.character.get_emotional_history())
        self.frame.register_subscriber(self)
        self.frame.show()

        # save all session data after the frame is closed
        self.save_session()
        logging.shutdown()

    # takes the users intent (per gui interaction) and starts the corresponding methods
    def handle_intent(self, intent, input_message=None, character=None, network=None):
        if intent == "load_character":
            self.character.load(character)
            self.frame.update_diagrams(self.character.get_emotional_state(), self.character.get_emotional_history())
        elif intent == "get_response":
            if input_message and input_message != "":
                self.handle_input(input_message)
        elif intent == "retrain_bot":
            self.bot.train()
        elif intent == "reset_state":
            self.character.reset_bot()
            self.frame.update_diagrams(self.character.get_emotional_state(), self.character.get_emotional_history())
        elif intent == "process_corpora":
            self.frame.update_log(["Warning! This operation may take a while. Check the python console for further updates"], clear=True)
            self.csv.cleanup_datasets(self.csv.get_list_of_corpora())
            self.csv.save_features_in_datasets(self.csv.get_list_of_corpora())
            self.csv.merge_datasets(self.csv.get_list_of_corpora())
        elif intent == "process_lexica":
            self.frame.update_log(["Warning! This operation may take a while. Check the python console for further updates"], clear=True)
            self.csv.cleanup_datasets(self.csv.get_list_of_lexica())
            self.csv.save_features_in_datasets(self.csv.get_list_of_lexica(), True)
            self.csv.merge_datasets(self.csv.get_list_of_lexica(), True)
        elif intent == "change_network":
            self.classifier.load_network(network)
            self.network_name = network

    # take user input, generate new data an update ui
    def handle_input(self, user_input):
        # user_input = self.correct_input(user_input)
        # update all modules
        response_package = self.bot.respond(user_input)
        ml_package = self.classifier.get_emotions(user_input)
        state_package = self.character.update_emotional_state(ml_package.get("input_emotions"))
        response_package = self.bot.modify_output(response_package, state_package["highest emotion"], state_package["highest_score"])

        # update gui
        self.frame.update_chat_out(user_input, response_package.get("response").__str__())
        self.frame.update_log([("network: " + self.network_name), ml_package, state_package, response_package])
        self.frame.update_diagrams(state_package.get("emotional_state"), state_package.get("emotional_history"))

    # corrects user input
    def correct_input(self, user_input):
        # make list of all words
        words = user_input.split(" ")
        unknown_words = self.spell.unknown(words)
        # replace all unknown words
        for word in unknown_words:
            print("correction: ", word, self.spell.correction(word))
            user_input = user_input.replace(word, self.spell.correction(word))

        return user_input

    # handles saving data when closing the program
    def save_session(self):
        # saves current character state
        self.character.save()

        # set the first launch variable to false
        self.config.set("default", "firstlaunch", "NO")
        self.config.set("net", "network", self.network_name)
        # save new value in file
        with open("../config/config.ini", "w") as f:
            self.config.write(f)


controller = Controller()
