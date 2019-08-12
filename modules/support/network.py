import pandas
import torch

def csv():
    dataset = pandas.read_csv("test.csv", dtype={"text": str, "affect": str, "lemmas": object, "pos": int})
    categories_map = {"happiness": 0, "sadness": 1, "anger": 2, "fear": 3, "disgust": 4}
    dataset["affect"] = dataset["affect"].map(categories_map)
    print(1, dataset.head())
    # erster wert: zeile, zwiter wert: spalte
    print(dataset.iloc[0, 1])
    print(2, dataset.iloc[0, 1:].values)
    print(3, dataset.iloc[0, 1:])


def torch():
    x = torch.rand(5, 3)
    print(x)
    print(x.size())
    print(type(x))

csv()