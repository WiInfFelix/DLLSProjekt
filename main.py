from histogramm import Histogrammer
from pprint import pprint
import os
import spacy
import re
from tqdm import tqdm
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

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


def get_tagged_text():
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
                for tup in text_tuples:
                    is_prop = False
                    for text_spans in span_input.spans:
                        if int(text_spans[0]) <= int(tup[1]) <= int(text_spans[1]):
                            tagged_text.append((tup[0], 'PROP'))
                            is_prop = True
                            break
                    if not is_prop:
                        tagged_text.append((tup[0], 'o'))



        input_list.extend(tagged_text)

    print('Finished tagging')
    # pprint(input_list[:250])

    print(f'There are a total of {len(input_list)} words in the input array!')
    return input_list

    le = LabelEncoder()

    df = pd.DataFrame(input_list, columns=['word', 'boolProp'])
    word_encoder = LabelEncoder()
    bin_encoder = LabelEncoder()

    word_num = word_encoder.fit_transform(df['word'])
    bin_num = bin_encoder.fit_transform(df['boolProp'])

    input_dim = len(word_encoder.classes_)
    output_dim = len(word_encoder.classes_)

    print(len(word_encoder.classes_), '::::', len(bin_encoder.classes_))


################################################################################


## Install data by running the following code:
# import nltk
# nltk.download('brown')
# nltk.download('universal_tagset')

def get_data(train_test_size):
    input_list = get_tagged_text()
    ## Take a subset
    corpus_words = [x[0] for x in input_list]
    corpus_tags = [x[1] for x in input_list]

    word_encoder = LabelEncoder()
    pos_encoder = LabelEncoder()

    corpus_words_num = word_encoder.fit_transform(corpus_words)
    corpus_tags_num = pos_encoder.fit_transform(corpus_tags)

    input_dim = len(word_encoder.classes_)
    output_dim = len(pos_encoder.classes_)

    return word_encoder, pos_encoder, corpus_words_num, corpus_tags_num, input_dim, output_dim


def prepare_data(left_context_len, right_context_len, train_test_size):
    _, _, x_data, y_data, input_dim, output_dim = get_data(train_test_size)

    train_data = [(x_data[i - left_context_len:i + right_context_len + 1], y_data[i]) for i in
                  range(left_context_len, len(x_data) - right_context_len)]
    x_train = np.array([pair[0] for pair in train_data])
    y_train = np.array([pair[1] for pair in train_data])

    return input_dim, output_dim, x_train, y_train


# can be used instead of prepare_data to get training data that is split on the sentence level
def prepare_data_sentences(train_test_size):
    word_encoder, pos_encoder, brown_words_num, brown_tags_num, input_dim, output_dim = get_data(train_test_size)

    x_data_sents, y_data_sents = [], []
    x_data_sent, y_data_sent = [], []

    dot_label = word_encoder.transform(['.'])[0]
    dot_label_tags = pos_encoder.transform(['.'])[0]

    # split on sentences
    for word, tag in zip(brown_words_num, brown_tags_num):

        if word == dot_label and tag == dot_label_tags:
            if len(x_data_sent) > 0:
                x_data_sents.append(x_data_sent)
                y_data_sents.append(y_data_sent)
                x_data_sent, y_data_sent = [], []

        x_data_sent.append(word)
        y_data_sent.append(tag)

    return input_dim, output_dim, x_data_sents, y_data_sents


# seq_len (int), input_dim (int), output_dim (int), embedding_dim (int), learning_rate (float)
# TODO: extend this with your LSTM code!
def build_graph(seq_len, input_dim, output_dim, embedding_dim, learning_rate):
    ## input
    x = tf.placeholder(tf.int32, (None, seq_len))
    y = tf.placeholder(tf.int32, (None))

    embeddings = tf.Variable(
        tf.random_uniform([input_dim, embedding_dim], -1.0, 1.0))

    ## embedd input
    x_embedd = tf.reshape(tf.nn.embedding_lookup(embeddings, x), [-1, embedding_dim * seq_len])

    ## linear model
    W = tf.Variable(tf.random_uniform([embedding_dim * seq_len, output_dim], -0.01, 0.01, dtype=tf.float32))
    b = tf.Variable(tf.random_uniform([output_dim], -0.01, 0.01, dtype=tf.float32))
    pred = tf.matmul(x_embedd, W) + b

    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=pred))

    pred_argmax = tf.argmax(tf.nn.softmax(pred), axis=1)

    ## define the optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    # function must return x (placeholder), y (placeholder),  optimizer (optimizer op),
    # loss (loss op),  pred_argmax (argmax of logits as tensor)
    return x, y, optimizer, loss, pred_argmax


def main():
    # model size parameters
    left_context_len = 4
    right_context_len = 0

    # set this higher to get a better model
    train_test_size = 500
    embedding_dim = 100

    ## Hyperparemeters: experiment with these, too
    learning_rate = 0.001
    epochs = 10

    seq_len = left_context_len + 1 + right_context_len
    input_dim, output_dim, x_data, y_data = prepare_data(left_context_len, right_context_len, train_test_size)
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data)
    x, y, optimizer, loss, pred_argmax = build_graph(seq_len, input_dim, output_dim, embedding_dim, learning_rate)

    ## start the session
    with tf.Session() as sess:

        ## initalize parameters
        sess.run(tf.global_variables_initializer())
        train_dict = {x: x_train, y: y_train}
        test_dict = {x: x_test, y: y_test}

        print("Initial training loss: " + str(sess.run(loss, train_dict)))
        print("Initial test loss: " + str(sess.run(loss, test_dict)))

        for i in range(epochs):
            ## run the optimizer
            epoch_data = list(zip(x_train, y_train))
            np.random.shuffle(epoch_data)
            for x_sample, y_sample in tqdm(epoch_data):
                train_dict_local = {x: [x_sample], y: [y_sample]}
                sess.run(optimizer, train_dict_local)
            print("Training loss after epoch " + str(i + 1) + ":" + str(sess.run(loss, train_dict)))

        print("Test loss after training:" + str(sess.run(loss, test_dict)))
        print(sess.run(pred_argmax, test_dict))


if __name__ == '__main__':
    main()
