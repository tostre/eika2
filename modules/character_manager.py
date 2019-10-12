import numpy as np


class Character_Manager:
    def __init__(self):
        self.trait_values = []
        self.max_values = []
        self.emotional_state = []
        self.emotional_history = None

        self.relationship_status = " "
        self.relationship_modifiers = {}
        self.decay_modifiers = []
        self.empathy_functions = []
        self.state_modifiers_threshold = 0
        self.state_modifiers_values = []
        self.delta_function = []
        pass

    def save(self, file):
        if file == "character_stable":
            self.get_stable_character()
        elif file == "character_empathetic":
            self.get_empathetic_character()
        elif file == "character_irascible":
            self.get_irascible_character()
        elif file == "character_default":
            self.get_default_character()

        self.relationship_status = "neutral"
        self.relationship_modifiers = {
            "love": 1.3,
            "friendship": 1.1,
            "neutral": 1,
            "dislike": 0.3
        }

        np.savez("../characters/" + file,
                 trait_values=self.trait_values,
                 max_values=self.max_values,
                 emotional_state=self.emotional_state,
                 emotional_history=self.emotional_history,
                 empathy_functions=self.empathy_functions,
                 decay_modifiers=self.decay_modifiers,
                 state_modifiers_values=self.state_modifiers_values,
                 state_modifiers_threshold=self.state_modifiers_threshold,
                 delta_function=self.delta_function,
                 relationship_status=self.relationship_status,
                 relationship_modifiers=self.relationship_modifiers)

    def get_stable_character(self):
        # h s a f d
        self.trait_values = [0.100, 0.000, 0.000, 0.000, 0.000]
        self.max_values = [0.900, 0.800, 0.800, 0.800, 0.800]
        self.emotional_state = self.trait_values.copy()
        self.emotional_history = np.zeros((5, 5))
        self.emotional_history[0] = self.emotional_state.copy()
        # decay mod related variables
        self.decay_modifiers = np.array([-0.01, -0.02, -0.01, -0.02, -0.05])
        # empathy mod
        self.empathy_functions = np.array([
            [[0.075, 0, 0, 1], [-0.1, 0, 0, 1], [-0.06, 0, 0, 1], [-0.08, 0, 0, 1], [-0.1, 0, 0, 1]],
            [[-0.10, 0, 0.1, 1], [0.05, 0, 0, 1], [0, 0, 0, 1], [0.05, 0, 0, 1], [0, 0, 0, 1]],
            [[-0.15, 0, 0.1, 1], [0, 0, 0, 1], [0.06, 0, 0, 1], [0, 0, 0, 1], [0.01, 0, 0, 1]],
            [[-0.01, 0, 0, 1], [0, 0, 0, 1], [0.00, 0, 0, 1], [0.05, 0, 0, 1], [0.07, 0, 0, 1]],
            [[-0.1, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0.2, 0, 0, 1]]
        ])
        # state modifier related variables
        self.state_modifiers_threshold = 0.75
        self.state_modifiers_values = [
            [1, 1, 0.9, -1, 1],  # lines show the emotion being influenced (happiness)
            [0.9, 1.1, 1.1, -1, 1],  # sadness
            [0.9, 1.1, 1.1, -1, 1],  # anger
            [0.9, 1.1, 1.1, -1, 1],  # fear
            [0.9, 1.1, 1.1, -1, 1]  # disgust
        ]
        # delta mod realated variables
        self.delta_function = [-0.5, 0, 0, 1]

    def get_empathetic_character(self):
        # h s a f d
        self.trait_values = [0.150, 0.150, 0.10, 0.10, 0.10]
        self.max_values = [1.00, 1.000, 9.000, 9.000, 9.000]
        self.emotional_state = self.trait_values.copy()
        self.emotional_history = np.zeros((5, 5))
        self.emotional_history[0] = self.emotional_state.copy()
        # decay mod related variables
        self.decay_modifiers = np.array([-0.01, -0.01, -0.03, -0.03, -0.1])
        # empathy mod
        self.empathy_functions = np.array([
            [[0.12, 0, 0, 1], [-0.0, 0, 0, 1], [-0.10, 0, 0, 1], [-0.10, 0, 0, 1], [-0.05, 0, 0, 1]],
            [[-0.00, 0, 0.1, 1], [0.12, 0, 0, 1], [0.03, 0, 0, 1], [0.00, 0, 0, 1], [0, 0, 0, 1]],
            [[-0.0, 0, 0.1, 1], [0, 0, 0, 1], [0.05, 0, 0, 1], [0, 0, 0, 1], [0.00, 0, 0, 1]],
            [[-0.0, 0, 0, 1], [0, 0, 0, 1], [0.00, 0, 0, 1], [0.05, 0, 0, 1], [0.05, 0, 0, 1]],
            [[-0.0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0.25, 0, 0, 1]]
        ])
        # state modifier related variables
        self.state_modifiers_threshold = 0.9
        self.state_modifiers_values = [
            [1, 1, 1, 1, 1],  # lines show the emotion being influenced (happiness)
            [1, 1, 1, 1, 1],  # sadness
            [0.7, 1, 1, 1, 1],  # anger
            [0.7, 1, 1, 1, 1],  # fear
            [0.7, 1, 1, 1, 1]  # disgust
        ]
        # delta mod realated variables
        self.delta_function = [-0.5, 0, 0, 1]

    def get_irascible_character(self):
        # h s a f d
        self.trait_values = [0.000, 0.000, 0.200, 0.000, 0.000]
        self.max_values = [0.500, 0.700, 1.000, 0.700, 0.700]
        self.emotional_state = self.trait_values.copy()
        self.emotional_history = np.zeros((5, 5))
        self.emotional_history[0] = self.emotional_state.copy()
        # decay mod related variables
        self.decay_modifiers = np.array([-0.1, -0.05, -0.01, -0.05, -0.05])
        # empathy mod
        self.empathy_functions = np.array([
            [[0.10, 0.00, 0.00, 0.15], [-0.1, 0, 0, 0.15], [-0.05, 0, 0, 0.15], [-0.05, 0, 0, 0.15], [0, 0, 0, 0.15]],
            [[-0.1, 0.00, 0.10, 1.00], [0.05, 0, 0, 1], [0, 0, 0, 1], [0.05, 0, 0, 1], [0, 0, 0, 1]],
            [[-0.05, 0.0, 0.50, 1.00], [0, 0, 0, 1], [0.15, 0.05, 0, 1], [0, 0, 0, 1], [0.05, 0.05, 0, 1]],
            [[-0.01, 0, 0, 0.01], [0, 0, 0, 0.01], [0.01, 0, 0, 0.01], [0.05, 0, 0, 0.01], [0.05, 0, 0, 0.01]],
            [[-0.1, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0.05, 0, 0, 1]]
        ])
        # state modifier related variables
        self.state_modifiers_threshold = 0.55
        self.state_modifiers_values = [
            [1.0, 0.9, 0.7, 1.0, 1.0],  # lines show the emotion being influenced (happiness)
            [0.9, 1.0, 0.7, 1.0, 1.0],  # sadness
            [1.0, 1.0, 1.5, 1.0, 1.2],  # anger
            [0.9, 1.0, 0.7, 1.0, 1.0],  # fear
            [0.9, 1.0, 1.0, 1.0, 1.0]  # disgust
        ]
        # delta mod realated variables
        self.delta_function = [-0.05, 0, 0, 1]

    def get_default_character(self):
        self.trait_values = [0.100, 0.100, 0.100, 0.100, 0.100]
        self.max_values = [0.900, 0.900, 0.900, 0.900, 0.900]
        self.emotional_state = self.trait_values.copy()
        self.emotional_history = np.zeros((5, 5))
        self.emotional_history[0] = self.emotional_state.copy()
        # decay mod related variables
        self.decay_modifiers = np.array([-0.01, -0.01, -0.01, -0.01, -0.1])
        # empathy functions
        self.empathy_functions = np.array([
            [[0.05, 0, 0, 1], [0, 0, 0, 1], [-0.05, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1]],  # lines show the emotion being influenced (happiness)
            [[-0.05, 0, 0.1, 1], [0.05, 0, 0, 1], [0, 0, 0, 1], [0.05, 0, 0, 1], [0, 0, 0, 1]],  # sadness
            [[-0.05, 0, 0.1, 1], [0, 0, 0, 1], [0.05, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1]],  # anger
            [[0, 0, 0, 1], [0, 0, 0, 1], [0.05, 0, 0, 1], [0.05, 0, 0, 1], [0.05, 0, 0, 1]],  # fear
            [[0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0.2, 0, 0, 1]]  # digust
        ])
        # state modifier related variables
        self.state_modifiers_threshold = 0.75
        self.state_modifiers_values = [
            [1, 1, 0.9, -1, 1],  # lines show the emotion being influenced (happiness)
            [0.9, 1.1, 1.1, -1, 1],  # sadness
            [0.9, 1.1, 1.1, -1, 1],  # anger
            [0.9, 1.1, 1.1, -1, 1],  # fear
            [0.9, 1.1, 1.1, -1, 1]  # disgust
        ]
        # delta mod realated variables
        self.delta_function = [-0.2, 0, 0, 1]
