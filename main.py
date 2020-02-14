from histogramm import Histogrammer
from pprint import pprint
import os
import spacy
import re
from tqdm import tqdm

TRAIN_ARTICLES = 'datasets/train-articles'
TRAIN_LABELS = 'datasets/train-labels-task1-span-identification'


class Article:

    def __init__(self, article_id, words):
        self.id = article_id
        self.words = words


class ArticleLabels:

    def __init__(self, article_id, spans):
        self.article_id = article_id
        self.spans = spans


def main():
    nlp = spacy.load("en_core_web_sm")

    id_getter = re.compile(r'\d+')

    articles_list = []
    span_list = []

    num_file = len(os.listdir(TRAIN_ARTICLES))
    print(f'{num_file} files have been found!')

    for entry in tqdm(os.scandir(TRAIN_ARTICLES)):
        with open(entry, encoding="utf-8") as file:
            text = file.read()
            tokens = nlp(text)
            article_id = id_getter.search(entry.name).group(0)
            article = Article(article_id, tokens)
            articles_list.append(article)

    print(f'Finished scanning articles. {len(articles_list)} have been read in...')

    for entry in tqdm(os.scandir(TRAIN_LABELS)):
        with open(entry) as file:
            tuples = []
            article_id = 0
            for line in file.readlines():
                split_line = line.strip().split('\t')
                span_tuple = (split_line[1], split_line[2])
                article_id = split_line[0]
                tuples.append(span_tuple)
            article_spans = ArticleLabels(article_id, tuples)
            span_list.append(article_spans)

    print(f'Finished reading in!')
    print(f'Found {len(articles_list)} articles and {len(span_list)} label files!')

    print('Test Output of classes:')

    print(articles_list[0].words[:10])
    print(span_list[0].spans[:5])

    input_list = []

    for text in tqdm(articles_list):
        tagged_text = []
        for span_input in span_list:
            if span_input.article_id == text.id:
                text_tuples = []
                for token in text.words:
                    text_tuples.append((token.text, token.idx))
                print(text_tuples[:2])
                for text_spans in span_input.spans:
                    for tup in text_tuples:
                        if int(text_spans[0]) <= int(tup[1]) <= int(text_spans[1]):
                            tagged_text.append((tup[0], 'PROP'))
                        else:
                            tagged_text.append((tup[0], 'o'))
        input_list.append(tagged_text)

    print('Finished tagging')
    pprint(input_list[:1])


if __name__ == '__main__':
    main()
