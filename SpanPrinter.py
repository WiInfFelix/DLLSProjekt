import os
import time
from pprint import pprint
import re

TRAIN_ARTICLES = './datasets/train-articles/'
LABELS_TASK1 = './datasets/train-labels-task1-span-identification/'


def get_article_number(article_name: str):
    p = re.compile(r'\d+')
    article_number = p.search(article_name).group(0)
    # print(f"Found article number: {article_number}")
    return article_number


def get_article_spans(file_number, lst_train_labels):
    for entry in lst_train_labels:
        res = []
        try:
            if file_number in entry:
                with open(entry) as f:
                    lines = f.readlines()
                    for line in lines:
                        spans = line.strip().split('\t')
                        span_tuple = (spans[1], spans[2])
                        res.append(span_tuple)
                    return res
        except:
            pprint(f"No labels where found for the article: {file_number}")

    return None


class SpanPrint:

    def __init__(self):
        self.spans_per_article = []

    def print_spans(self):
        lst_train_labels = []
        lst_train_articles = []
        with os.scandir(LABELS_TASK1) as files:
            for file in files:
                lst_train_labels.append(LABELS_TASK1 + file.name)

        for file in lst_train_labels:
            with open(file) as f:
                self.spans_per_article = f.readlines()
                self.spans_per_article = [x.strip().split('\t') for x in self.spans_per_article]

        with os.scandir(TRAIN_ARTICLES) as files:
            for file in files:
                lst_train_articles.append(TRAIN_ARTICLES + file.name)

        for file in lst_train_articles:
            file_number = get_article_number(file)
            spans = get_article_spans(file_number, lst_train_labels)
            with open(file, encoding='utf-8') as f:
                article = f.read()
                for entry in spans:
                    print(article[int(entry[0]):int(entry[1])])
