#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy as np
np.random.seed(1337) # for reproducibility

import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    warnings.filterwarnings("ignore",category=FutureWarning)
    warnings.filterwarnings("ignore",category=UserWarning)
    import sklearn
    import h5py     
    import keras
import os
import codecs
import theano
import gc
import itertools
import sklearn
import jellyfish
import collections as col
import pandas as pd
from collections import Counter 
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.models import Model
from keras.models import load_model
from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from sklearn.cross_validation import StratifiedKFold
from nltk import tokenize
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from multisvmwrapper import SVMWrapper
from multilmwrapper import LMWrapper
from sklearn.naive_bayes import MultinomialNB
from skmultilearn.problem_transform import LabelPowerset

# Set parameters:
max_features = 65000            # Maximum number of tokens in vocabulary
maxlen = 200                    # Maximum Length of each Sentence
maxsents = 9                    # Maximum Number of Sentences (5 for Death Certificate + 1 for Autopsy Report + 1 for Clinical Information Bulletin)
maxsents_co = 5                 # Number of Sentences in Death Certificate
batch_size = 32                 # Batch size given to the model while training
embedding_dims = 175            # Embedding Dimensions
nb_epoch = 50                   # Number of epochs for training
validation_split = 0.25         # Percentage of the dataset used in validation                                                         
gru_output_size = 175           # GRU output dimension
kernel_size = 5
filters = 50
pool_size = 4

print('Loading data...')
# Shape of each line in dataset:
# 'Full ICD-10 code of underlying death cause' <> 'Death Certificate' <> 'Clinical Information Bulletin' <> 'Autopsy Report' <> 'Full ICD-10 codes present in Death Certificate'
texts = [ line.rstrip('\n') for line in codecs.open('dataset_example.txt', 
         encoding="iso_8859-1") ]                                                    

# labels_cid is a list of the ICD-10 full code for the underlying death cause for each dataset entry
labels_cid = list([ line.split('<>')[0][:-1] for line in texts ])

# labels_cid_aux is a list of the ICD-10 full codes present in the death certificate
labels_cid_aux = [ line.split('<>')[10].replace("'","") for line in texts ]
labels_cid_aux = [x[2:-2] for x in labels_cid_aux]
labels_cid_aux = [x.split(', ') for x in labels_cid_aux]
print('Found %s unique ICD-10 codes.' % len(set([item for sublist in labels_cid_aux for item in sublist])))

# Using sklearn package attribute an integer to each code that occures resulting in the variables:
# labels_int, labels_int_3char, labels_int_aux 
le4 = preprocessing.LabelEncoder()
le4_aux = preprocessing.LabelEncoder()

char4 = le4.fit(labels_cid)
char4_aux = le4_aux.fit([item for sublist in labels_cid_aux for item in sublist])

labels_int = char4.transform(labels_cid)
labels_int_aux = np.copy(labels_cid_aux)

for i in range(len(labels_int_aux)):
    labels_int_aux[i] = char4_aux.transform(labels_int_aux[i])

part_1a = [ line.split('<>')[1].lower() for line in texts ]
part_1b = [ line.split('<>')[2].lower() for line in texts ]
part_1c = [ line.split('<>')[3].lower() for line in texts ]
part_1d = [ line.split('<>')[4].lower() for line in texts ]
part_2 = [ line.split('<>')[5].lower() for line in texts ]
bic = [ line.split('<>')[6].lower() for line in texts ]
bic_admiss = [ line.split('<>')[7].lower() for line in texts ]
bic_sit = [ line.split('<>')[8].lower() for line in texts ]
ra = [ line.split('<>')[9].lower() for line in texts ]

print('Converting ICD-9 codes into one-hot vectors...')
# e.g. J189 (in labels_cid) -> 3 (in labels_int) -> [0, 0, 0, 1, 0, (...), 0] (in labels)
      
labels_int = np.asarray(labels_int)
labels_int_aux = np.asarray(labels_int_aux)

labels = to_categorical(labels_int)

num_classes=1+max([max(x) for x in labels_int_aux])
 
labels_aux = np.zeros((len(labels), num_classes), dtype=np.float64)
for i in range(len(labels_int_aux)):
    labels_aux[i,:] = sum( to_categorical(list(set(labels_int_aux[i])), num_classes=num_classes))

print('Spliting the data into a training set and a validation set...')

X_train_1a, X_test_1a, y_train, y_test = train_test_split(part_1a, labels_aux, stratify = labels_cid, test_size = 0.25, random_state=42)
X_train_1a, X_test_1a, y_train_aux, y_test_aux = train_test_split(part_1a, labels, stratify = labels_cid, test_size = 0.25, random_state=42)

X_train_1b, X_test_1b, y_train, y_test = train_test_split(part_1b, labels_aux, stratify = labels_cid, test_size = 0.25, random_state=42)
X_train_1c, X_test_1c, y_train, y_test = train_test_split(part_1c, labels_aux, stratify = labels_cid, test_size = 0.25, random_state=42)
X_train_1d, X_test_1d, y_train, y_test = train_test_split(part_1d, labels_aux, stratify = labels_cid, test_size = 0.25, random_state=42)
X_train_2, X_test_2, y_train, y_test = train_test_split(part_2, labels_aux, stratify = labels_cid, test_size = 0.25, random_state=42)
X_train_bic, X_test_bic, y_train, y_test = train_test_split(bic, labels_aux, stratify = labels_cid, test_size = 0.25, random_state=42)
X_train_bic_admiss, X_test_bic_admiss, y_train, y_test = train_test_split(bic_admiss, labels_aux, stratify = labels_cid, test_size = 0.25, random_state=42)
X_train_bic_sit, X_test_bic_sit, y_train, y_test = train_test_split(bic_sit, labels_aux, stratify = labels_cid, test_size = 0.25, random_state=42)
X_train_ra, X_test_ra, y_train, y_test = train_test_split(ra, labels_aux, stratify = labels_cid, test_size = 0.25, random_state=42)

print('Tokenizing vocabulary...')

tokenizer = Tokenizer(num_words = max_features)
tokenizer.fit_on_texts(X_train_1a+X_train_1b+X_train_1c+X_train_1d+X_train_2+X_train_bic+X_train_bic_admiss+X_train_bic_sit+X_train_ra)

# attribute an integer to each token that occures in the texts 
# conversion of each dataset entry in a (7,200) shape matrix resulting in variables:

print('Computing Training Set...')

# data is a (5,200) matrix for the strings in death certificates
X_train = np.zeros((len(X_train_1a), maxsents, maxlen), dtype = 'int32')

print(' - Loading discharge summaries...')

discharge_summary = [X_train_1a, X_train_1b, X_train_1c, X_train_1d, X_train_2, X_train_bic, X_train_bic_admiss, X_train_bic_sit, X_train_ra]
for m in range(len(discharge_summary)):
    part = discharge_summary[m]
    for i, sentences in enumerate(part):
        sentences = tokenize.sent_tokenize( sentences )
        k = 0
        for j, sent in enumerate(sentences):
            if j < maxsents:
                wordTokens = text_to_word_sequence(sent)
                for _ , word in enumerate(wordTokens):
                    if k < maxlen and tokenizer.word_index[word] < max_features:
                        X_train[i,m,k] = tokenizer.word_index[word]
                        k = k + 1

X_train = np.int32([[item for sublist in line for item in sublist] for line in X_train])

word_index = tokenizer.word_index

np.save('DICT.npy', word_index)
np.save('FULL_CODES.npy', le4)
np.save('AUX_CODES.npy', le4_aux)

print('Found %s unique tokens.' % len(word_index))

print('Computing Testing Set...')

X_test = np.zeros((len(X_test_1a), maxsents, maxlen), dtype = 'int32')

print(' - Loading discharge summaries...')

discharge_summary = [X_test_1a, X_test_1b, X_test_1c, X_test_1d, X_test_2, X_test_bic, X_test_bic_admiss, X_test_bic_sit, X_test_ra]
for m in range(len(discharge_summary)):
    part = discharge_summary[m]
    for i, sentences in enumerate(part):
        sentences = tokenize.sent_tokenize( sentences )
        k = 0
        for j, sent in enumerate(sentences):
            wordTokens = text_to_word_sequence(sent)
            for _ , word in enumerate(wordTokens):
                # if the word is out-of-vocabulary it is substituted by the most
                # similar word in the dictionary
                if word_index.get(word) == None: 
                    aux = [(jellyfish.jaro_winkler(k,word),v) for k,v in word_index.items()]
                    if k < maxlen and max(aux)[1] < max_features:
                        X_test[i,m,k] = max(aux)[1]
                        k = k + 1
                else:
                    if k < maxlen and word_index.get(word) < max_features:
                        X_test[i,m,k] = word_index.get(word)
                        k = k + 1

X_test = np.int32([[item for sublist in line for item in sublist] for line in X_test])

#%%
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

print('Build model...')
model = LMWrapper()
#model = SVMWrapper()

print('Fitting model...')
model.fit(X_train, y_train)
#model.save('modelo_baseline.h5')

print('Predicting...')
predictions = model.predict(X_test)[0].rows

one_hot_pred = np.zeros((len(y_test), num_classes), dtype=np.float64)
for i in range(len(predictions)):
    one_hot_pred[i,:] = sum( to_categorical(predictions[i], num_classes=num_classes))
    
#%%
print('Writing output...')

for i in range(len(predictions)):
    for j in range(len(predictions[i])): predictions[i][j] = le4_aux.inverse_transform(predictions[i][j])

np.savetxt('pred_baseline.txt', predictions, delimiter=" ", fmt="%s")

y_t = [np.where(x != 0)[0].astype(str) for x in y_test]

for i in range(len(y_t)):
    for j in range(len(y_t[i])):
        y_t[i][j] = le4_aux.inverse_transform(int(y_t[i][j]))

np.savetxt('true_baseline.txt', y_t, delimiter=" ", fmt="%s")

y_t_aux = [np.where(x != 0)[0].astype(str) for x in y_test_aux]

for i in range(len(y_t_aux)):
    for j in range(len(y_t_aux[i])):
        y_t_aux[i][j] = le4.inverse_transform(int(y_t_aux[i][j]))

np.savetxt('true_baseline_main.txt', y_t_aux, delimiter=" ", fmt="%s")

print('\n -> MULTI-LABEL PERFORMANCE METRICS:')

print('\n      -> Accuracy: %s' % sklearn.metrics.accuracy_score(y_true=y_test,y_pred=one_hot_pred))
print('\n      -> Precision: %s' % sklearn.metrics.precision_score(y_true=y_test,y_pred=one_hot_pred, average='weighted'))
print('\n      -> Recall: %s' % sklearn.metrics.recall_score(y_true=y_test,y_pred=one_hot_pred, average='weighted'))
print('\n      -> F1: %s' % sklearn.metrics.f1_score(y_true=y_test,y_pred=one_hot_pred, average='weighted'))
print('\n      -> Hamming Loss: %s' % sklearn.metrics.hamming_loss(y_true=y_test,y_pred=one_hot_pred))
print('\n      -> Exact Match: %s' % (1-sklearn.metrics.zero_one_loss(y_true=y_test,y_pred=one_hot_pred)))