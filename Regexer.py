import re
import os
from pprint import pprint


class TextTagger:

    def __init__(self):
        pass

    def tuple_articles_and_indices(self):
        tuple_dict = {}
        with os.scandir('./datasets/train-articles') as folder:
            for file in folder:
                tuple_dict[re.split("\.", file.name)[0]] = []

        with os.scandir('./datasets/train-labels-task1-span-identification') as folder:
            for file in folder:
                key = re.split('\.', file.name)[0]

                with open(file, 'r') as article:
                    for line in article.readlines():
                        array = line.rstrip().split('\t')
                        tuple_dict[key].append((array[1], array[2]))

        return tuple_dict

    def tuple_words_indices(self):
        tuple_dict = {}
        text_list = []
        with os.scandir('./datasets/train-articles') as folder:
            for file in folder:
                with open(file, 'r', encoding='utf-8') as f:
                    text = f.read()
                    text_list = text.split()
                    text_list = [(x, text.index(x)) for x in text_list]
                    tuple_dict[file.name] = text_list

        return tuple_dict
