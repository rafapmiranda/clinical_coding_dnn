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
import time
import codecs
import theano
import jellyfish
import gc
import itertools
import pandas as pd
import collections as col
from collections import Counter 
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.core import Masking
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import GlobalAveragePooling1D
from keras.layers import GlobalAveragePooling2D
from keras.layers import GlobalMaxPooling1D
from keras.layers import GlobalMaxPooling2D
from keras.layers import Reshape
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Input
from keras.layers import Flatten
from keras.layers import Convolution1D, MaxPooling1D
from keras.layers import Conv1D, MaxPool1D, Conv2D, MaxPool2D, Embedding, Dropout, LSTM, GRU, Bidirectional, TimeDistributed
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.models import load_model
from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from sklearn.cross_validation import StratifiedKFold
from nltk import tokenize
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.decomposition import NMF
from attention import AttLayer
from sparse_att import SparseAttention
from svmwrapper import SVMWrapper
from lmwrapper import LMWrapper

#Start
time_start = time.clock()

# Stopping Criteria for training the model
earlyStopping = EarlyStopping(monitor = 'loss', patience=1, verbose=0, mode='auto')
modelChekpoint = ModelCheckpoint(filepath = 'modelo_full.h5', monitor = 'loss', verbose=0, save_best_only=True, mode='auto')

# Set parameters:
max_features = 150000           # Maximum number of tokens in vocabulary
maxlen = 30                     # Maximum Length of each Sentence
maxsents = 35                   # Maximum Number of Sentences (9 for Discharge diagnosis + 1 for Internment reason + 25 for Clinical summary)
batch_size = 15                 # Batch size given to the model while training
embedding_dims = 150            # Embedding Dimensions
maxchars_word = 15              # Maximum number of chars in each token
nb_epoch = 25                   # Number of epochs for training
validation_split = 0.25         # Percentage of the dataset used in validation                                                         
gru_output_size = 150           # GRU output dimension
final_output = embedding_dims+gru_output_size

print('Loading data...')

print('Computation parameters: (%s, (%s, %s, %s))' % (max_features, maxsents, maxlen, maxchars_word))
# Shape of each line in dataset:
# 'Full ICD-9 code of main diagnosis' <> 'Discharge Diagnosis' <> 'Internment Reason' <> 'Clinical Summary' <> 'Full ICD-9 codes present in Discharge summary'
texts = [ line.rstrip('\n') for line in codecs.open('dataset_example_hba_full.txt', encoding="iso-8859-1") ]                                                    
print('\nDataset size is %s.' % len(texts))
# labels_cid is a list of the ICD-9 full code for the main diagnosis for each dataset entry
labels_cid = list([ line.split('<>')[0][:-1] for line in texts ])
print('Found %s main ICD-9 full codes.' % len(set(labels_cid)))
# labels_cid_aux is a list of ICD-90 full code lists, present in each discharge summary
labels_cid_aux = [ line.split('<>')[38].replace("'","") for line in texts ]
labels_cid_aux = [x[2:-2] for x in labels_cid_aux]
labels_cid_aux = [x.split(', ') for x in labels_cid_aux]
print('Found %s total ICD-9 full codes.' % len(set([item for sublist in labels_cid_aux for item in sublist])))
# labels_cid_3_aux is identic to labels_cid_aux but the codes are truncated to 3 characters (ICD-9 block)
labels_cid_3_aux = np.copy(labels_cid_aux)
for i in range(len(labels_cid_3_aux)):
    labels_cid_3_aux[i] = [x[:3] for x in labels_cid_aux[i]]

# labels_cid_chap is identic to labels_cid_3_aux but, instead of ICD-9 blocks, we have integers between 1 and 19 (ICD-9 chapters)
labels_cid_chap = [[block[:3] for block in line] for line in labels_cid_aux]
for i in range(len(labels_cid_chap)):
    for j in range(len(labels_cid_chap[i])):
        if labels_cid_chap[i][j] >= '001' and labels_cid_chap[i][j] <= '139':
            labels_cid_chap[i][j] = 1 
        elif labels_cid_chap[i][j] >= '140' and labels_cid_chap[i][j] <= '239': 
            labels_cid_chap[i][j] = 2
        elif labels_cid_chap[i][j] >= '240' and labels_cid_chap[i][j] <= '279': 
            labels_cid_chap[i][j] = 3
        elif labels_cid_chap[i][j] >= '280' and labels_cid_chap[i][j] <= '289': 
            labels_cid_chap[i][j] = 4
        elif labels_cid_chap[i][j] >= '290' and labels_cid_chap[i][j] <= '319': 
            labels_cid_chap[i][j] = 5
        elif labels_cid_chap[i][j] >= '320' and labels_cid_chap[i][j] <= '389': 
            labels_cid_chap[i][j] = 6
        elif labels_cid_chap[i][j] >= '390' and labels_cid_chap[i][j] <= '459': 
            labels_cid_chap[i][j] = 7
        elif labels_cid_chap[i][j] >= '460' and labels_cid_chap[i][j] <= '519': 
            labels_cid_chap[i][j] = 8
        elif labels_cid_chap[i][j] >= '520' and labels_cid_chap[i][j] <= '579': 
            labels_cid_chap[i][j] = 9
        elif labels_cid_chap[i][j] >= '580' and labels_cid_chap[i][j] <= '629': 
            labels_cid_chap[i][j] = 10
        elif labels_cid_chap[i][j] >= '630' and labels_cid_chap[i][j] <= '679': 
            labels_cid_chap[i][j] = 11
        elif labels_cid_chap[i][j] >= '680' and labels_cid_chap[i][j] <= '709': 
            labels_cid_chap[i][j] = 12
        elif labels_cid_chap[i][j] >= '710' and labels_cid_chap[i][j] <= '739': 
            labels_cid_chap[i][j] = 13
        elif labels_cid_chap[i][j] >= '740' and labels_cid_chap[i][j] <= '759': 
            labels_cid_chap[i][j] = 14
        elif labels_cid_chap[i][j] >= '760' and labels_cid_chap[i][j] <= '779': 
            labels_cid_chap[i][j] = 15
        elif labels_cid_chap[i][j] >= '780' and labels_cid_chap[i][j] <= '799': 
            labels_cid_chap[i][j] = 16
        elif labels_cid_chap[i][j] >= '800' and labels_cid_chap[i][j] <= '999': 
            labels_cid_chap[i][j] = 17
        elif labels_cid_chap[i][j] >= 'V01' and labels_cid_chap[i][j] <= 'V91': 
            labels_cid_chap[i][j] = 18
        else: 
            labels_cid_chap[i][j] = 19

# Using sklearn package attribute an integer to each code that occures resulting in the variables:
# labels_int, labels_int_aux, labels_int_3_aux
le4 = preprocessing.LabelEncoder()
le4_aux = preprocessing.LabelEncoder()
le3_aux = preprocessing.LabelEncoder()

char4 = le4.fit(labels_cid)
char4_aux = le4_aux.fit([item for sublist in labels_cid_aux for item in sublist])       
char3_aux = le3_aux.fit([item for sublist in labels_cid_3_aux for item in sublist])

labels_int = char4.transform(labels_cid)

labels_int_aux = np.copy(labels_cid_aux)
labels_int_3_aux = np.copy(labels_cid_3_aux)

for i in range(len(labels_int_aux)):
    labels_int_aux[i] = char4_aux.transform(labels_int_aux[i])
    labels_int_3_aux[i] = char3_aux.transform(labels_int_3_aux[i])

part_1a = [ line.split('<>')[1] for line in texts ]
part_1b = [ line.split('<>')[2] for line in texts ]
part_1c = [ line.split('<>')[3] for line in texts ]
part_1d = [ line.split('<>')[4] for line in texts ]
part_1e = [ line.split('<>')[5] for line in texts ]
part_1f = [ line.split('<>')[6] for line in texts ]
part_1g = [ line.split('<>')[7] for line in texts ]
part_1h = [ line.split('<>')[8] for line in texts ]
part_1i = [ line.split('<>')[9] for line in texts ]
mi = [ line.split('<>')[10] for line in texts ]
rc_1 = [ line.split('<>')[11] for line in texts ]
rc_2 = [ line.split('<>')[12] for line in texts ]
rc_3 = [ line.split('<>')[13] for line in texts ]
rc_4 = [ line.split('<>')[14] for line in texts ]
rc_5 = [ line.split('<>')[15] for line in texts ]
rc_6 = [ line.split('<>')[16] for line in texts ]
rc_7 = [ line.split('<>')[17] for line in texts ]
rc_8 = [ line.split('<>')[18] for line in texts ]
rc_9 = [ line.split('<>')[19] for line in texts ]
rc_10 = [ line.split('<>')[20] for line in texts ]
rc_11 = [ line.split('<>')[21] for line in texts ]
rc_12 = [ line.split('<>')[22] for line in texts ]
rc_13 = [ line.split('<>')[23] for line in texts ]
rc_14 = [ line.split('<>')[24] for line in texts ]
rc_15 = [ line.split('<>')[25] for line in texts ]
rc_16 = [ line.split('<>')[26] for line in texts ]
rc_17 = [ line.split('<>')[27] for line in texts ]
rc_18 = [ line.split('<>')[28] for line in texts ]
rc_19 = [ line.split('<>')[29] for line in texts ]
rc_20 = [ line.split('<>')[30] for line in texts ]
rc_21 = [ line.split('<>')[31] for line in texts ]
rc_22 = [ line.split('<>')[32] for line in texts ]
rc_23 = [ line.split('<>')[33] for line in texts ]
rc_24 = [ line.split('<>')[34] for line in texts ]
rc_25 = [ line.split('<>')[35] for line in texts ]

age = [ int(line.split(' <> ')[36]) for line in texts ]
dep = [ line.split('<>')[37].lower() for line in texts ]

# Grouping of patient's age on age groups
for i in range(len(age)):
    if age[i] < 5: age[i] = 0
    elif age[i] < 15: age[i] = 1
    elif age[i] < 25: age[i] = 2
    elif age[i] < 45: age[i] = 3
    elif age[i] < 65: age[i] = 4
    else: age[i] = 5

# Convertion of discharge summary's department into a one-hot vector
led = preprocessing.LabelEncoder()
chard = led.fit(dep)
dep = chard.transform(dep)

age = np.asarray(age)
dep = np.asarray(dep)
labels_int = np.asarray(labels_int)
labels_int_aux = np.asarray(labels_int_aux)
labels_int_3_aux = np.asarray(labels_int_3_aux)
labels_cid_chap = np.asarray(labels_cid_chap)

# Conversion of the Full ICD-9 code into a one-hot vector
# e.g. 250 (in labels_cid) -> 3 (in labels_int) -> [0, 0, 0, 1, 0, (...), 0] (in labels)

age = to_categorical(age)
dep = to_categorical(dep)
labels = to_categorical(labels_int)    
          
num_classes=1+max([max(x) for x in labels_int_aux])    
labels_aux = np.zeros((len(labels), num_classes), dtype=np.float64)
for i in range(len(labels_int_aux)):
    labels_aux[i,:] = sum( to_categorical(list(set(labels_int_aux[i])),num_classes))
    
num_classes_3=1+max([max(x) for x in labels_int_3_aux])    
labels_3_aux = np.zeros((len(labels), num_classes_3), dtype=np.float64)
for i in range(len(labels_int_3_aux)):
    labels_3_aux[i,:] = sum( to_categorical(list(set(labels_int_3_aux[i])),num_classes_3))
    
num_classes_chap=1+max([max(x) for x in labels_cid_chap])
labels_chap = np.zeros((len(labels),num_classes_chap), dtype=np.float64)
for i in range(len(labels_cid_chap)):
    labels_chap[i,:] = sum( to_categorical(list(set(labels_cid_chap[i])),num_classes_chap))
    
#%%
print('Spliting the data into a training set and a validation set...')

X_train_1a, X_test_1a, y_train, y_test = train_test_split(part_1a, labels, stratify = labels_cid, test_size = 0.25, random_state=42)
X_train_1a, X_test_1a, y_train_aux, y_test_aux = train_test_split(part_1a, labels_aux, stratify = labels_cid, test_size = 0.25, random_state=42)
X_train_1a, X_test_1a, y_train_3_aux, y_test_3_aux = train_test_split(part_1a, labels_3_aux, stratify = labels_cid, test_size = 0.25, random_state=42)
X_train_1a, X_test_1a, y_train_chap, y_test_chap = train_test_split(part_1a, labels_chap, stratify = labels_cid, test_size = 0.25, random_state=42)

X_train_1b, X_test_1b, y_train, y_test = train_test_split(part_1b, labels, stratify = labels_cid, test_size = 0.25, random_state=42)
X_train_1c, X_test_1c, y_train, y_test = train_test_split(part_1c, labels, stratify = labels_cid, test_size = 0.25, random_state=42)
X_train_1d, X_test_1d, y_train, y_test = train_test_split(part_1d, labels, stratify = labels_cid, test_size = 0.25, random_state=42)
X_train_1e, X_test_1e, y_train, y_test = train_test_split(part_1e, labels, stratify = labels_cid, test_size = 0.25, random_state=42)
X_train_1f, X_test_1f, y_train, y_test = train_test_split(part_1f, labels, stratify = labels_cid, test_size = 0.25, random_state=42)
X_train_1g, X_test_1g, y_train, y_test = train_test_split(part_1g, labels, stratify = labels_cid, test_size = 0.25, random_state=42)
X_train_1h, X_test_1h, y_train, y_test = train_test_split(part_1h, labels, stratify = labels_cid, test_size = 0.25, random_state=42)
X_train_1i, X_test_1i, y_train, y_test = train_test_split(part_1i, labels, stratify = labels_cid, test_size = 0.25, random_state=42)
X_train_mi, X_test_mi, y_train, y_test = train_test_split(mi, labels, stratify = labels_cid, test_size = 0.25, random_state=42)
X_train_rc1, X_test_rc1, y_train, y_test = train_test_split(rc_1, labels, stratify = labels_cid, test_size = 0.25, random_state=42)
X_train_rc2, X_test_rc2, y_train, y_test = train_test_split(rc_2, labels, stratify = labels_cid, test_size = 0.25, random_state=42)
X_train_rc3, X_test_rc3, y_train, y_test = train_test_split(rc_3, labels, stratify = labels_cid, test_size = 0.25, random_state=42)
X_train_rc4, X_test_rc4, y_train, y_test = train_test_split(rc_4, labels, stratify = labels_cid, test_size = 0.25, random_state=42)
X_train_rc5, X_test_rc5, y_train, y_test = train_test_split(rc_5, labels, stratify = labels_cid, test_size = 0.25, random_state=42)
X_train_rc6, X_test_rc6, y_train, y_test = train_test_split(rc_6, labels, stratify = labels_cid, test_size = 0.25, random_state=42)
X_train_rc7, X_test_rc7, y_train, y_test = train_test_split(rc_7, labels, stratify = labels_cid, test_size = 0.25, random_state=42)
X_train_rc8, X_test_rc8, y_train, y_test = train_test_split(rc_8, labels, stratify = labels_cid, test_size = 0.25, random_state=42)
X_train_rc9, X_test_rc9, y_train, y_test = train_test_split(rc_9, labels, stratify = labels_cid, test_size = 0.25, random_state=42)
X_train_rc10, X_test_rc10, y_train, y_test = train_test_split(rc_10, labels, stratify = labels_cid, test_size = 0.25, random_state=42)
X_train_rc11, X_test_rc11, y_train, y_test = train_test_split(rc_11, labels, stratify = labels_cid, test_size = 0.25, random_state=42)
X_train_rc12, X_test_rc12, y_train, y_test = train_test_split(rc_12, labels, stratify = labels_cid, test_size = 0.25, random_state=42)
X_train_rc13, X_test_rc13, y_train, y_test = train_test_split(rc_13, labels, stratify = labels_cid, test_size = 0.25, random_state=42)
X_train_rc14, X_test_rc14, y_train, y_test = train_test_split(rc_14, labels, stratify = labels_cid, test_size = 0.25, random_state=42)
X_train_rc15, X_test_rc15, y_train, y_test = train_test_split(rc_15, labels, stratify = labels_cid, test_size = 0.25, random_state=42)
X_train_rc16, X_test_rc16, y_train, y_test = train_test_split(rc_16, labels, stratify = labels_cid, test_size = 0.25, random_state=42)
X_train_rc17, X_test_rc17, y_train, y_test = train_test_split(rc_17, labels, stratify = labels_cid, test_size = 0.25, random_state=42)
X_train_rc18, X_test_rc18, y_train, y_test = train_test_split(rc_18, labels, stratify = labels_cid, test_size = 0.25, random_state=42)
X_train_rc19, X_test_rc19, y_train, y_test = train_test_split(rc_19, labels, stratify = labels_cid, test_size = 0.25, random_state=42)
X_train_rc20, X_test_rc20, y_train, y_test = train_test_split(rc_20, labels, stratify = labels_cid, test_size = 0.25, random_state=42)
X_train_rc21, X_test_rc21, y_train, y_test = train_test_split(rc_21, labels, stratify = labels_cid, test_size = 0.25, random_state=42)
X_train_rc22, X_test_rc22, y_train, y_test = train_test_split(rc_22, labels, stratify = labels_cid, test_size = 0.25, random_state=42)
X_train_rc23, X_test_rc23, y_train, y_test = train_test_split(rc_23, labels, stratify = labels_cid, test_size = 0.25, random_state=42)
X_train_rc24, X_test_rc24, y_train, y_test = train_test_split(rc_24, labels, stratify = labels_cid, test_size = 0.25, random_state=42)
X_train_rc25, X_test_rc25, y_train, y_test = train_test_split(rc_25, labels, stratify = labels_cid, test_size = 0.25, random_state=42)

X_train_age, X_test_age, y_train, y_test = train_test_split(age, labels, stratify = labels_cid, test_size = 0.25, random_state=42)
X_train_dep, X_test_dep, y_train, y_test = train_test_split(dep, labels, stratify = labels_cid, test_size = 0.25, random_state=42)

#%%
# Attribute an integer to each token that occures in the texts 
tokenizer = Tokenizer(num_words = max_features)
tokenizer.fit_on_texts(X_train_1a+X_train_1b+X_train_1c+X_train_1d+X_train_1e+X_train_1f+X_train_1g+X_train_1h+X_train_1i+X_train_mi+X_train_rc1+X_train_rc2+X_train_rc3+X_train_rc4+X_train_rc5+X_train_rc6+X_train_rc7+X_train_rc8+X_train_rc9+X_train_rc10+X_train_rc11+X_train_rc12+X_train_rc13+X_train_rc14+X_train_rc15+X_train_rc16+X_train_rc17+X_train_rc18+X_train_rc19+X_train_rc20+X_train_rc21+X_train_rc22+X_train_rc23+X_train_rc24+X_train_rc25)
print('\nFound %s unique tokens.' % len(tokenizer.word_index))
word_count = [k[0] for k in tokenizer.word_counts.items() if k[1] > 1]
word_index = tokenizer.word_index
word_keys = list(word_index.keys())

for i in range(len(word_keys)):
    if word_keys[i] not in word_count: del word_index[word_keys[i]]
    
tokenizer.word_index = word_index

# Text representation based on characters and word case
case2Idx = {'start_token':1, 'allLower':2, 'allUpper':3, 'initialUpper':4, 'numeric':5, 'contains_digit':6, 'other':7, 'PADDING_TOKEN':0}
caseEmbeddings = np.identity(len(case2Idx), dtype='float32')
char2Idx = {"PADDING":0, "UNKNOWN":1}
for c in " 0123456789abcdefghijklmnopqrstuvwxyzàáâãéêíóôõúüABCDEFGHIJKLMNOPQRSTUVWXYZÀÁÂÃÉÊÍÓÔÕÚÜ.,-_()[]{}!?:;#'\"/\\%$`&=*+@^~|": char2Idx[c] = len(char2Idx)

def getCasing(word, caseLookup):   
    casing = 'other'
    numDigits = 0
    for char in word:
        if char.isdigit(): numDigits += 1
    if word == '873heGKe7I': casing = 'start_token'
    elif word == 'NUMBER': casing = 'numeric'
    elif word.islower(): casing = 'allLower'
    elif word.isupper(): casing = 'allUpper'
    elif word[0].isupper(): casing = 'initialUpper'
    elif numDigits > 0: casing = 'contains_digit'
    return caseLookup[casing]

def getChars(word, charLookup):
    aux = [ charLookup["PADDING"] ]
    for char in word:
        if char in charLookup: aux.append(charLookup[char])
        else: aux.append(charLookup["UNKNOWN"])
    while len(aux) < maxchars_word: aux.append(charLookup["PADDING"])
    return np.array( aux )

# Conversion of each dataset entry in a (35,30) shape matrix resulting in variables:
print('\nComputing Training Set...')

# Data is a (35,30) matrix for the strings in discharge summaries (X_train_char has a (35,30,15) shape because of character-level dimention)
X_train = np.zeros((len(X_train_1a), maxsents, maxlen), dtype = 'int32')
X_train_char = np.zeros((len(X_train_1a), maxsents, maxlen, maxchars_word), dtype = 'int8')
X_train_casing = np.zeros((len(X_train_1a), maxsents, maxlen), dtype = 'int8')

print(' - Loading discharge summaries...')

discharge_summary = [X_train_1a, X_train_1b, X_train_1c, X_train_1d, X_train_1e, X_train_1f, X_train_1g, X_train_1h, X_train_1i, X_train_mi, X_train_rc1, X_train_rc2, X_train_rc3, X_train_rc4, X_train_rc5, X_train_rc6, X_train_rc7, X_train_rc8, X_train_rc9, X_train_rc10, X_train_rc11, X_train_rc12, X_train_rc13, X_train_rc14, X_train_rc15, X_train_rc16, X_train_rc17, X_train_rc18, X_train_rc19, X_train_rc20, X_train_rc21, X_train_rc22, X_train_rc23, X_train_rc24, X_train_rc25]
for m in range(len(discharge_summary)):
    part = discharge_summary[m]
    for i, sentences in enumerate(part):
        sentences = tokenize.sent_tokenize( sentences )
        k = 0
        for j, sent in enumerate(sentences):
            if j < maxsents:
                wordTokens = text_to_word_sequence(sent, lower=False)
                for _ , word in enumerate(wordTokens):
                # if the word is out-of-vocabulary it is substituted by the most similar word in the dictionary
                    if word.lower() not in tokenizer.word_index: 
                        aux = [(jellyfish.jaro_winkler(k,word.lower()),v) for k,v in tokenizer.word_index.items()]
                        if k < maxlen and max(aux)[1] < max_features:
                            X_train[i,m,k] = max(aux)[1]
                            X_train_casing[i,m,k] = getCasing(word,case2Idx)
                            if len(word) > maxchars_word-1: word = word[:(maxchars_word-1)]
                            X_train_char[i,m,k,:] = getChars(word,char2Idx)
                            k = k + 1
                    else:
                        if k < maxlen and tokenizer.word_index[word.lower()] < max_features:
                            X_train[i,m,k] = tokenizer.word_index[word.lower()]
                            X_train_casing[i,m,k] = getCasing(word,case2Idx)
                            if len(word) > maxchars_word-1: word = word[:(maxchars_word-1)]
                            X_train_char[i,m,k,:] = getChars(word,char2Idx)
                            k = k + 1

word_index = tokenizer.word_index

print('\nSaving variables...')
np.save('DICT.npy', word_index)
np.save('MAIN.npy', le4)
np.save('FULL_CODES.npy', le4_aux)
np.save('BLOCKS.npy', le3_aux)

print('\nFound %s unique tokens.' % len(word_index))

print('\nComputing Testing Set...')

X_test = np.zeros((len(X_test_1a), maxsents, maxlen), dtype = 'int32')
X_test_char = np.zeros((len(X_test_1a), maxsents, maxlen, maxchars_word), dtype = 'int8')
X_test_casing = np.zeros((len(X_test_1a), maxsents, maxlen), dtype = 'int8')

print(' - Loading discharge summaries...')

discharge_summary = [X_test_1a, X_test_1b, X_test_1c, X_test_1d, X_test_1e, X_test_1f, X_test_1g, X_test_1h, X_test_1i, X_test_mi, X_test_rc1, X_test_rc2, X_test_rc3, X_test_rc4, X_test_rc5, X_test_rc6, X_test_rc7, X_test_rc8, X_test_rc9, X_test_rc10, X_test_rc11, X_test_rc12, X_test_rc13, X_test_rc14, X_test_rc15, X_test_rc16, X_test_rc17, X_test_rc18, X_test_rc19, X_test_rc20, X_test_rc21, X_test_rc22, X_test_rc23, X_test_rc24, X_test_rc25]
for m in range(len(discharge_summary)):
    part = discharge_summary[m]
    for i, sentences in enumerate(part):
        sentences = tokenize.sent_tokenize( sentences )
        k = 0
        for j, sent in enumerate(sentences):
            wordTokens = text_to_word_sequence(sent, lower=False)
            for _ , word in enumerate(wordTokens):
                # if the word is out-of-vocabulary it is substituted by the most similar word in the dictionary
                if word_index.get(word.lower()) == None: 
                    aux = [(jellyfish.jaro_winkler(k,word.lower()),v) for k,v in word_index.items()]
                    if k < maxlen and max(aux)[1] < max_features:
                        X_test[i,m,k] = max(aux)[1]
                        X_test_casing[i,m,k] = getCasing(word,case2Idx)
                        if len(word) > maxchars_word-1: word = word[:(maxchars_word-1)]
                        X_test_char[i,m,k,:] = getChars(word,char2Idx)
                        k = k + 1
                else:
                    if k < maxlen and word_index.get(word.lower()) < max_features:
                        X_test[i,m,k] = word_index.get(word.lower())
                        X_test_casing[i,m,k] = getCasing(word,case2Idx)
                        if len(word) > maxchars_word-1: word = word[:(maxchars_word-1)]
                        X_test_char[i,m,k,:] = getChars(word,char2Idx)
                        k = k + 1

#%%
print('Computing Initialization with Label Co-occurrence...')                        
                        
train_labels_full = [np.where(x != 0)[0] for x in y_train]
train_labels_aux = [np.where(x != 0)[0] for x in y_train_aux]
train_labels_3_aux = [np.where(x != 0)[0] for x in y_train_3_aux]

# Common labels to the single-label and to the multi-label scenario
train_labels_full_cid = [le4.inverse_transform(line)[0] for line in train_labels_full]
train_labels_aux_cid = [le4_aux.inverse_transform(line) for line in train_labels_aux]
common_labels_cid = np.array(list(set([item for sublist in train_labels_aux_cid for item in sublist]).intersection(set(train_labels_full_cid))))

# Single-label Main full-codes
init_m_full = np.zeros((y_train.shape[1],y_train.shape[1]), dtype=np.float32)
bias_full = np.zeros((y_train.shape[1]), dtype=np.float32)

def create_nmf(init_m):
    nmf = NMF(n_components=final_output)
    init_m = np.log2(init_m + 1)
    nmf.fit(init_m)
    init_m = nmf.components_
    return init_m

for n in range(len(train_labels_full)):
    row = [x for x in train_labels_aux_cid[n] if x in common_labels_cid]
    for i in row:
        for j in row:
            a = le4.transform([i])
            b = le4.transform([j])
            init_m_full[a,b] += 1

init_m_full = create_nmf(init_m_full)

# Multi-label Full-codes
init_m_aux = np.zeros((num_classes,num_classes), dtype=np.float32)
bias_aux = np.zeros((num_classes), dtype=np.float32)

for n in range(len(train_labels_aux)):
    for i in train_labels_aux[n]:
        for j in train_labels_aux[n]:
            init_m_aux[i,j] += 1

init_m_aux = create_nmf(init_m_aux)

# Multi-label Blocks
init_m_3 = np.zeros((num_classes_3,num_classes_3), dtype=np.float32)
bias_3 = np.zeros((num_classes_3), dtype=np.float32)

for n in range(len(train_labels_3_aux)):
    for i in train_labels_3_aux[n]:
        for j in train_labels_3_aux[n]:
            init_m_3[i,j] += 1

init_m_3 = create_nmf(init_m_3)

#Checkpoint
time_elapsed = (time.clock() - time_start)
print('Pre-processing: Time elapsed is',time_elapsed)
time_start = time.clock()

#%%
print('Build model...')

# Inputs
review_input_words = Input(shape=(maxsents,maxlen), dtype='int32')
review_input_casing = Input(shape=(maxsents,maxlen), dtype='int8')
review_input_chars = Input(shape=(maxsents,maxlen,maxchars_word), dtype='int8')
age_input = Input(shape=(X_train_age.shape[1],), dtype='float32')
dep_input = Input(shape=(X_train_dep.shape[1],), dtype='float32')
aux_input = Input(shape=(y_train_aux.shape[1],), dtype='float32')

# Embedding Layers
embedding_layer_words = Embedding(len(word_index), embedding_dims, embeddings_initializer=keras.initializers.RandomUniform(minval=-0.5, maxval=0.5), input_length=maxlen)
embedding_layer_casing = Embedding(caseEmbeddings.shape[0], 5, embeddings_initializer=keras.initializers.RandomUniform(minval=-0.5, maxval=0.5), trainable=False)
embedding_layer_character = Embedding(len(char2Idx),25,embeddings_initializer=keras.initializers.RandomUniform(minval=-0.5, maxval=0.5))

sentence_input_words = Input(shape=(maxlen,), dtype='int32')
embedded_sequences_words = embedding_layer_words(sentence_input_words)

sentence_input_casing = Input(shape=(maxlen,), dtype='int8')
embedded_sequences_casing = embedding_layer_casing(sentence_input_casing)

sentence_input_chars = Input(shape=(maxlen,maxchars_word), dtype='int8')
embedded_sequences_chars = TimeDistributed(embedding_layer_character)(sentence_input_chars)
embedded_sequences_chars = TimeDistributed(Bidirectional(GRU(50, return_sequences=False)))(embedded_sequences_chars)

review_words = TimeDistributed(Model(sentence_input_words, embedded_sequences_words))(review_input_words)
review_casing = TimeDistributed(Model(sentence_input_casing, embedded_sequences_casing))(review_input_casing)
review_chars = TimeDistributed(Model(sentence_input_chars, embedded_sequences_chars))(review_input_chars)

review_embedded = keras.layers.Concatenate()( [ review_words , review_casing , review_chars] )
print(review_embedded.shape)
review_embedded = TimeDistributed(TimeDistributed(Dense(embedding_dims,activation='tanh')))(review_embedded)

# Average of Word Embeddings
fasttext = GlobalAveragePooling2D()(review_embedded)
reshape = Reshape((maxsents*maxlen,embedding_dims))(review_embedded)

# Convolutional Neural Network
conv1 = Conv1D(embedding_dims, kernel_size=2, activation='tanh')(reshape)
conv1 = MaxPool1D(int((maxsents*maxlen)/5))(conv1)
conv1 = Flatten()(conv1)

conv2 = Conv1D(embedding_dims, kernel_size=4, activation='tanh')(reshape)
conv2 = MaxPool1D(int((maxsents*maxlen)/5))(conv2)
conv2 = Flatten()(conv2)

conv3 = Conv1D(embedding_dims, kernel_size=8, activation='tanh')(reshape)
conv3 = MaxPool1D(int((maxsents*maxlen)/5))(conv3)
conv3 = Flatten()(conv3)

concatenated_tensor = keras.layers.Concatenate(axis=1)([fasttext , conv1 , conv2 , conv3])
fasttext = Dense(units=embedding_dims, activation='tanh')(concatenated_tensor)

# Bidirectional GRU
l_gru_sent = TimeDistributed(Bidirectional(GRU(gru_output_size, return_sequences=True)))(review_embedded)
l_gru_sent = keras.layers.Concatenate()( [ l_gru_sent , Reshape((maxsents,maxlen,gru_output_size))( keras.layers.RepeatVector(maxsents*maxlen)(fasttext) ) ] )
l_dense_sent = TimeDistributed(TimeDistributed(Dense(units=gru_output_size)))(l_gru_sent)
l_att_sent = TimeDistributed(AttLayer())(l_dense_sent)

# Bidirectional GRU
l_gru_review = Bidirectional(GRU(gru_output_size, return_sequences=True))(l_att_sent)
l_gru_review = keras.layers.Concatenate()( [ l_gru_review , keras.layers.RepeatVector(maxsents)(fasttext) ] )
l_dense_review = TimeDistributed(Dense(units=gru_output_size))(l_gru_review)
postp = AttLayer()(l_dense_review)

# Memory Mechanism
aux_mem = Dense(units=(final_output), activation='tanh', weights=(init_m_aux.transpose(),np.zeros(gru_output_size+embedding_dims)), name='memory')(aux_input)
postp_aux = keras.layers.Concatenate( axis = 1 )( [ postp , fasttext , aux_mem , age_input , dep_input] )
postp = Dropout(0.05)(postp_aux)
postp = Dense(units=(final_output))(postp)

# Softmax/Sigmoid Output Layer
preds = Dense(units=y_train.shape[1], activation='softmax', weights=[init_m_full, bias_full], name='main')(postp)
preds_aux = Dense(units=y_train_aux.shape[1], activation='sigmoid', weights=[init_m_aux, bias_aux], name='full_code')(postp)
preds_3char = Dense(units=y_train_3_aux.shape[1], activation='sigmoid', weights=[init_m_3, bias_3], name='block')(postp)
preds_chap = Dense(units=y_train_chap.shape[1], activation='sigmoid', name='chap')(postp)

# Multi-label to Single-label Problem Adaption
y_train_flat = [item for sublist in train_labels_aux for item in sublist]
X_train_flat = []

for i in range(len(X_train)):
    for j in range(len(train_labels_aux[i])): X_train_flat.append(X_train[i])

y_train_flat = to_categorical(y_train_flat)
X_train_flat = np.array(X_train_flat)

model = LMWrapper()
model.fit(X_train_flat, y_train_flat)
model.save('modelo_baseline.h5')

#Checkpoint
time_elapsed = (time.clock() - time_start)
print('Baseline fit: Time elapsed is',time_elapsed)
time_start = time.clock()

X_train_aux = model.predict_prob(X_train)[0]
X_test_aux = model.predict_prob(X_test)[0]

model = Model(inputs = [review_input_words, review_input_casing, review_input_chars, age_input, dep_input, aux_input], outputs = [preds, preds_aux, preds_3char, preds_chap])
model.compile(loss=['categorical_crossentropy','binary_crossentropy','binary_crossentropy','binary_crossentropy'], optimizer='adam', metrics=['accuracy'], loss_weights = [0.85, 0.85, 0.75, 0.70])

#Checkpoint
time_elapsed = (time.clock() - time_start)
print('Baseline prob. prediction: Time elapsed is',time_elapsed)
time_start = time.clock()

print('Fit full model...')
model.fit([ X_train, X_train_casing, X_train_char, X_train_age, X_train_dep, X_train_aux ], [y_train, y_train_aux, y_train_3_aux, y_train_chap], batch_size=batch_size, epochs=nb_epoch, validation_data=([X_test,X_test_casing,X_test_char,X_test_age,X_test_dep,X_test_aux], [y_test, y_test_aux, y_test_3_aux, y_test_chap]), callbacks=[earlyStopping, modelCheckpoint])
model = load_model('modelo_full.h5', custom_objects = {"AttLayer": AttLayer, "LMWrapper":LMWrapper}

#Checkpoint
time_elapsed = (time.clock() - time_start)
print('Model fit: Time elapsed is',time_elapsed)
time_start = time.clock()

#%%
print('Predicting...')
[all_4, all_aux, all_3, all_c] = model.predict([X_test,X_test_casing,X_test_char,X_test_age,X_test_dep,X_test_aux], batch_size=3)

print('Writing output...')

cid_pred = np.zeros([len(y_test),7], dtype = object)

for i in range(len(y_test)):
    top3_4 = np.argsort(all_4[i])[-3:]
    top3_3 = np.argsort(all_3[i])[-3:]
    cid_pred[i][0] = le4.inverse_transform(np.argmax(y_test[i]))
    for j in [1,2,3]:
        cid_pred[i][j] = le4.inverse_transform(top3_4[-j])
        cid_pred[i][3+j] = le3_aux.inverse_transform(top3_3[-j])

np.savetxt('pred_full_nmf.txt', cid_pred, delimiter=" ", fmt="%s")

y_t = [np.where(x != 0)[0].astype(str) for x in y_test_aux]

for i in range(len(y_t)):
    for j in range(len(y_t[i])):
        y_t[i][j] = le4_aux.inverse_transform(int(y_t[i][j]))
    y_t[i] = y_t[i].tolist()
    
np.savetxt('true_baseline.txt', y_t, delimiter=" ", fmt="%s")

all_aux_bool = np.copy(all_aux)
    
for i in range(len(all_aux_bool)):
    all_aux_bool[i] = all_aux_bool[i] >= 0.5
    for j in range(len(all_aux_bool[i])):
        if all_aux_bool[i][j] != 0: all_aux_bool[i][j] = all_aux[i][j]
    
all_aux_bool = [[(k,v) for k,v in enumerate(line) if v != 0] for line in all_aux_bool]
all_aux_bool = [sorted(line, key=lambda x: x[1])[::-1] for line in all_aux_bool]
all_aux_bool = [[k[0] for k in line] for line in all_aux_bool]

for i in range(len(all_aux_bool)):
    for j in range(len(all_aux_bool[i])):
        all_aux_bool[i][j] = le4_aux.inverse_transform(all_aux_bool[i][j])

aux = [line[1] for line in cid_pred]

for i in range(len(all_aux_bool)):
    if all_aux_bool[i] == []: all_aux_bool[i].insert(0,aux[i])

np.savetxt('pred_baseline.txt', all_aux_bool, delimiter=" ", fmt="%s")

all_3_bool = np.copy(all_3)
    
for i in range(len(all_3_bool)):
    all_3_bool[i] = all_3_bool[i] >= 0.5
    for j in range(len(all_3_bool[i])):
        if all_3_bool[i][j] != 0: all_3_bool[i][j] = all_3[i][j]
    
all_3_bool = [[(k,v) for k,v in enumerate(line) if v != 0] for line in all_3_bool]
all_3_bool = [sorted(line, key=lambda x: x[1])[::-1] for line in all_3_bool]
all_3_bool = [[k[0] for k in line] for line in all_3_bool]

for i in range(len(all_3_bool)):
    for j in range(len(all_3_bool[i])):
        all_3_bool[i][j] = le3_aux.inverse_transform(all_3_bool[i][j])

aux = [line[1][:3] for line in cid_pred]

for i in range(len(all_3_bool)):
    if all_3_bool[i] == []: all_3_bool[i].insert(0,aux[i])

np.savetxt('pred_baseline_block.txt', all_3_bool, delimiter=" ", fmt="%s")

all_c_bool = np.copy(all_c)
    
for i in range(len(all_c_bool)):
    all_c_bool[i] = all_c_bool[i] >= 0.5
    for j in range(len(all_c_bool[i])):
        if all_c_bool[i][j] != 0: all_c_bool[i][j] = all_c[i][j]
    
all_c_bool = [[(k,v) for k,v in enumerate(line) if v != 0] for line in all_c_bool]
all_c_bool = [sorted(line, key=lambda x: x[1])[::-1] for line in all_c_bool]
all_c_bool = [[k[0] for k in line] for line in all_c_bool]

np.savetxt('pred_baseline_chap.txt', all_c_bool, delimiter=" ", fmt="%s")

print('Calculating MRR...')

all_aux_sort = np.copy(all_aux)
all_4_sort = np.copy(all_4)
all_3_sort = np.copy(all_3)
all_c_sort = np.copy(all_c)

cid = np.array([np.argmax(line) for line in y_test])
cid_h = np.array([np.where(x != 0)[0] for x in y_test_aux])
cid_3 = np.array([np.where(x != 0)[0] for x in y_test_3_aux])
cid_c = np.array([np.where(x != 0)[0] for x in y_test_chap])

for i in range(len(all_aux_sort)):
    all_aux_sort[i] = np.argsort(all_aux_sort[i])[::-1]
    all_4_sort[i] = np.argsort(all_4_sort[i])[::-1]
    all_3_sort[i] = np.argsort(all_3_sort[i])[::-1]
    all_c_sort[i] = np.argsort(all_c_sort[i])[::-1]
    cid[i] = np.where(all_4_sort[i] == cid[i])[0].astype(int)+1
    for j in range(len(cid_h[i])):
        cid_h[i][j] = np.where(all_aux_sort[i] == cid_h[i][j])[0].astype(int)[0]+1
    for j in range(len(cid_3[i])):
        cid_3[i][j] = np.where(all_3_sort[i] == cid_3[i][j])[0].astype(int)[0]+1
    for j in range(len(cid_c[i])):
        cid_c[i][j] = np.where(all_c_sort[i] == cid_c[i][j])[0].astype(int)[0]+1

cid_h = np.array([min(line) for line in cid_h])
cid_3 = np.array([min(line) for line in cid_3])
cid_c = np.array([min(line) for line in cid_c])

s = len(cid)

cid = [1/elem for elem in cid]
cid_h = [1/elem for elem in cid_h]
cid_3 = [1/elem for elem in cid_3]
cid_c = [1/elem for elem in cid_c]

print('\n      -> Mean Reciprocal Rank (FULL-CODE): %s' % (sum(cid)/s))
print('\n      -> Mean Reciprocal Rank: %s' % (sum(cid_h)/s))
print('\n      -> Mean Reciprocal Rank: %s' % (sum(cid_3)/s))
print('\n      -> Mean Reciprocal Rank: %s' % (sum(cid_c)/s))

#Checkpoint
time_elapsed = (time.clock() - time_start)
print('Prediction and output writing: Time elapsed is',time_elapsed)
