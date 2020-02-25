import os
from tensorflow import keras
import tensorflow as tf
import tensorflow_datasets as tfds
tfds.disable_progress_bar()

from tqdm import tqdm
from spacy.lang.en import English

import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn import preprocessing 


from keras.models import Sequential
from keras.layers import Dense, LSTM, InputLayer, Bidirectional, TimeDistributed, Embedding, Activation


pickle_list = "/content/drive/My Drive/Master Wirtschaftsinformatik/3. Semester /DLLS/DLLSProjekt/list.p"
pickle_texts = "/content/drive/My Drive/Master Wirtschaftsinformatik/3. Semester /DLLS/DLLSProjekt/texts.p"

dataset_folder = "/content/drive/My Drive/Master Wirtschaftsinformatik/3. Semester /DLLS/DLLSProjekt/datasets"
train_texts = "/content/drive/My Drive/Master Wirtschaftsinformatik/3. Semester /DLLS/DLLSProjekt/datasets/train-articles"
train_labels = "/content/drive/My Drive/Master Wirtschaftsinformatik/3. Semester /DLLS/DLLSProjekt/datasets/train-labels-task1-span-identification"
model_save_file = "/content/drive/My Drive/Master Wirtschaftsinformatik/3. Semester /DLLS/DLLSProjekt/keras_model.h5"


left_context = 4
right_context = 0

from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM
from sklearn.preprocessing import LabelEncoder
from keras_contrib.layers import CRF


def get_data(train_test_size):
    input_list = pickle.load(open(pickle_list, "rb"))
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
    word_encoder, pos_encoder, corpus_words_num, corpus_tags_num, input_dim, output_dim = get_data(train_test_size)

    x_data_sents, y_data_sents = [], []
    x_data_sent, y_data_sent = [], []

    dot_label = word_encoder.transform(['.'])[0]
    dot_label_tags = pos_encoder.transform(['.'])[0]

    # split on sentences
    for word, tag in zip(corpus_words_num, corpus_tags_num):

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
    train_test_size = 50000
    embedding_dim = 1000

    ## Hyperparemeters: experiment with these, too
    learning_rate = 0.001
    epochs =  10

    seq_len = left_context_len + 1 + right_context_len
    input_dim, output_dim, x_data, y_data = prepare_data(left_context_len, right_context_len, train_test_size)
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data)
    x, y, optimizer, loss, pred_argmax = build_graph(seq_len, input_dim, output_dim, embedding_dim, learning_rate)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 4} ) 
    sess = tf.Session(config=config) 
    keras.backend.set_session(sess)

  
    model = Sequential()
    model.add(Embedding(input_dim, output_dim=1000))
    model.add(LSTM(128))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
    
#    model = load_model(model_save_file)

    model.fit(x_train, y_train, batch_size=32, epochs=10)
    #model.save(model_save_file)  # creates a HDF5 file 'my_model.h5'
    score = model.evaluate(x_test, y_test, batch_size=32)
    print(score)



if __name__ == '__main__':
    main()
