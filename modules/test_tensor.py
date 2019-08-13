from __future__ import absolute_import, division, print_function, unicode_literals
import functools
import numpy as np
import tensorflow as tf
import pandas
import torch
import torch.nn as nn

class TF:
    def __init__(self):
        # Ausgabe von Panda-Dataframes nicht verkürzen
        pandas.set_option('display.max_rows', 500)
        pandas.set_option('display.max_columns', 500)
        pandas.set_option('display.width', 1000)

        self.categories = ["happiness", "sadness", "anger", "fear", "disgust"]

        # ZTum Trainieren muss man die Label (categories in Zahlen umwandeln. Am einfachsten ist es den Index in der LIste zu nehmen:
        # for category in self.categories:
        #     category_num = self.categories.index(category)
         #   print(category, category_num)

    def erstes_netz(self):
        pass

    def rumprobieren(self):
            # Matrix mit 5 Zeilen und 3 Spalten
            a = torch.randn(2, 3)
            b = torch.rand(2, 3)
            # Fragen ob Cuda da ist, dann die Befehle unten auführen
            # Dann werden die Berechnungen dieser Tensoren auf der GraKa ausgeführt
            if torch.cuda.is_available():
                print("GraKa is available")
                a = a.cuda()
                b = b.cuda()
            # Matrizen einfach addieren
            z = torch.add(a, b)
            print("a+b", z)

            # Tensor: Vektor
            c = torch.randn(5)
            print(c)
            # Tensor: 3D-Matrix
            d = torch.randn(5, 2, 2)
            print(d)
            print(d.size())
            # Tensoren mit Listen erstellen
            e = torch.Tensor([[1, 2, 3, 4, 5], [3, 5, 7, 3, 6]])
            print(e)



tf_test = TF()
