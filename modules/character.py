import numpy as np
import logging


class Character:
    # constructs a character instance
    def __init__(self, first_launch):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        # init emotional state variables
        self.emotional_state = np.zeros(5)
        self.emotional_history = np.zeros((5, 5))
        # modifier variables
        self.empathy_modifier = 0
        self.state_modifier = 0
        self.mean_delta = 0
        self.result = 0

        # in the case of the first launch, load default values, else load previous state
        self.load("character_default") if first_launch else self.load("character_saved")

    # loads character variables from a npz file
    def load(self, file):
        # irgendein update hat die ladefunktion zweschossen. Daher muss ich jetzt
        # diesen workaround anwenden, damits keinen fehler schmeißt
        self.np_load_old = np.load
        np.load = lambda *a, **k: self.np_load_old(*a, allow_pickle=True, **k)
        # load characters and values
        self.character_npz = np.load("../characters/" + file + ".npz")
        self.trait_values = self.character_npz.get("trait_values")
        self.max_values = self.character_npz.get("max_values")
        self.emotional_state = self.character_npz.get("emotional_state")
        self.emotional_history = self.character_npz.get("emotional_history")
        self.empathy_functions = self.character_npz.get("empathy_functions")
        self.decay_modifiers = self.character_npz.get("decay_modifiers")
        self.state_modifiers = self.character_npz.get("state_modifiers_values")
        self.state_modifiers_threshold = self.character_npz.get("state_modifiers_threshold")
        self.delta_function = self.character_npz.get("delta_function")
        self.relationship_status = self.character_npz.get("relationship_status").item()
        # In den Dateien format.py und npyio.py des NumbyModules habe ich allow_pickles=True
        # gesetzt. Andernfalls wird hier ein Fehler geworfen
        self.relationship_modifiers = self.character_npz.get("relationship_modifiers").item()

        self.logger.info(f"Session start. {file} loaded")
        np.load = self.np_load_old

    # saves the current character in a npz file
    def save(self):
        np.savez("../characters/character_saved",
                 trait_values=self.trait_values,
                 max_values=self.max_values,
                 emotional_state=self.emotional_state,
                 emotional_history=self.emotional_history,
                 empathy_functions=self.empathy_functions,
                 decay_modifiers=self.decay_modifiers,
                 state_modifiers_values=self.state_modifiers,
                 state_modifiers_threshold=self.state_modifiers_threshold,
                 delta_function=self.delta_function,
                 relationship_status=self.relationship_status,
                 relationship_modifiers=self.relationship_modifiers)
        self.logger.info("Session end. character_saved saved.")

    # resets character state (emotional state/history)
    def reset_bot(self):
        self.emotional_state = self.trait_values.copy()
        self.emotional_history = np.zeros((5, 5))
        self.emotional_history[0] = self.emotional_state.copy()

    # updates internal emotional state/history based on input emotions
    def update_emotional_state(self, input_emotions):
        # Saves the change for one emotion caused by all input emotions
        emotional_state_old = self.emotional_state.copy()
        modifier = np.zeros((5, 3))

        # Apply relationship modifier (on input emotions)
        if self.relationship_status in self.relationship_modifiers:
            input_emotions *= self.relationship_modifiers[self.relationship_status]

        # build modifier array
        for emotion, value in enumerate(input_emotions, start=0):
            # get empathy modifiers, state modifiers
            modifier[emotion, 0] = self.decay_modifiers[emotion]
            modifier[emotion, 1] = self.get_empathy_modifier(input_emotions, emotion)
            modifier[emotion, 2] = self.get_state_modifier(emotion)
            self.emotional_state[emotion] = emotional_state_old[emotion] + (np.sum(modifier[emotion, 0:-1]) * modifier[emotion, 2])
        # apply delta modifier
        self.emotional_state[0] += self.get_delta_modifier(self.emotional_state, emotional_state_old)

        # update emotional state and history
        self.emotional_state = self.clean_state(self.emotional_state)
        self.emotional_history = np.insert(self.emotional_history[0:-1], 0, self.emotional_state, 0)

        return {
            "emotional_state": self.emotional_state,
            "emotional_history": self.emotional_history,
            "highest emotion": np.argmax(self.emotional_state),
            "highest_score": max(self.emotional_state)}

    # Returns value of modifier based on the interaction function between two emotions
    def get_empathy_modifier(self, input_emotions, emotion):
        empathy_modifier = 0
        for i, function in enumerate(self.empathy_functions[emotion], start=0):
            empathy_modifier += self.linear_function(input_emotions[i], function)
        return empathy_modifier

    # Lowers/Raises influence of other mods based on height of emo-value
    def get_state_modifier(self, emotion):
        self.state_modifier = np.mean(self.state_modifiers[emotion])
        if self.emotional_state[emotion] <= self.state_modifiers_threshold:
            return 1
        else:
            return self.state_modifier

    # Raises happiness if negative emotions fell enough
    def get_delta_modifier(self, emotional_state_new, emotional_state_old):
        # Calc mean of the differences of the last four emotions (hence the delete 0)
        self.mean_delta = np.mean(np.delete(emotional_state_new - emotional_state_old, 0))
        if self.mean_delta < 0:
            return self.linear_function(self.mean_delta, self.delta_function)
        else:
            return 0

    # checks max and min value und sets values accordingly and rounds values
    def clean_state(self, emotional_state):
        # check if all emotions are in range of trait and max values
        for index, value in enumerate(emotional_state, start=0):
            if value < self.trait_values[index]:
                emotional_state[index] = self.trait_values[index]
            elif value > self.max_values[index]:
                emotional_state[index] = self.max_values[index]
        # Round emotional state values so they can be used for further calculations
        emotional_state = np.round(self.emotional_state, 3)
        return emotional_state

    # Returns the calculation of a linear function
    def linear_function(self, x, function):
        # a function is an array and built as such:
        # f[0] = m (steigung), f[1] = b (Achsenabschnitt), f[2] = t (threshhold), f[3] = m (max-wert den die funktion annhemen kann)
        self.result = (function[0] * x) + function[1]

        # check if function result is within min (threshold) and max value
        if abs(self.result) <= function[2] or x == 0:
            return 0
        elif abs(self.result) >= function[3]:
            return round(function[3], 3)
        else:
            return round(self.result, 3)

    # Returns the current emotional state
    def get_emotional_state(self):
        return self.emotional_state

    # Returns the emotional history
    def get_emotional_history(self):
        return self.emotional_history
