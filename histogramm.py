import pandas as pd
import os
from collections import Counter
import matplotlib.pyplot as mp
from pprint import pprint


class Histogrammer:

    def __init__(self):
        self.cnt = Counter()

    def count_occurences(self):

        lst = []
        with os.scandir('./datasets/train-labels-task2-technique-classification') as files:
            for file in files:
                lst.append('./datasets/train-labels-task2-technique-classification/' + file.name)

        for file in lst:
            with open(file) as f:
                for line in f.readlines():
                    array = line.split('\t')
                    self.cnt[array[1]] += 1

        pprint(self.cnt)

    def plot_graph(self):
        labels, values = zip(*self.cnt.items())

        mp.barh(labels, values)
        mp.xticks(rotation=90)
        mp.tight_layout()
        mp.show()
