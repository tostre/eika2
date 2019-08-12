from __future__ import absolute_import, division, print_function, unicode_literals
import functools
import numpy as np
import tensorflow as tf
import pandas

class TF:
    def __init__(self):
        self.categories = ["happiness", "sadness", "anger", "fear", "disgust"]

        # ZTum Trainieren muss man die Label (categories in Zahlen umwandeln. Am einfachsten ist es den Index in der LIste zu nehmen:
        for category in self.categories:
            category_num = self.categories.index(category)
            print(category, category_num)

        # Data sets
        IRIS_TRAINING = "iris_training.csv"
        IRIS_TEST = "iris_test.csv"


        self.training_data = pandas.read_csv("datasets/lexicon.csv", delimiter=",",
                        dtype={"text": str, "affect": str, "embeddings": object,
                               "lemmas": object})
        self.test_data = pandas.read_csv("datasets/lexicon_test.csv", delimiter=",",
                                             dtype={"text": str, "affect": str, "embeddings": object,
                                                    "lemmas": object})


tf_test = TF()
