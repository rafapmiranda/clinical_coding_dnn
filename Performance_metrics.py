#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy as np

np.random.seed(1337) # for reproducibility

import pandas as pd
import codecs
#from keras.utils.np_utils import to_categorical
from sklearn import preprocessing 
from matplotlib import pyplot as plt
from sklearn import metrics
from Multi_label_metrics import ex_based_acc, ex_based_precision, ex_based_recall, ex_based_f1, multi_hamming_loss, exact_match, one_error

# LOAD PREDICTIONS

# The file predictions.txt has one array for each instance, organized in the following way:
# true label, 3 most probable 4 digit (full-codes) predicted labels, 3 most probable 3 digit (blocks) predicted labels

labels_pred = np.genfromtxt('pred_full_nmf.txt', dtype = 'str')
labels_pred_ml = np.array([ line.replace(',','').replace('[','').replace(']','').replace("'",'').rstrip('\n\r').split(' ') for line in codecs.open('pred_baseline.txt', encoding="iso_8859-1") ])

# labels_cid has the true labels
labels_cid = [x[0] for x in labels_pred]
labels_cid_ml = np.array([ line.replace(',','').replace('[','').replace(']','').replace("'",'').rstrip('\n\r').split(' ') for line in codecs.open('true_baseline.txt', encoding="iso_8859-1") ])
# labels_pred the predicted labels
labels_pred = [[x[1],x[2],x[3],x[4],x[5],x[6]] for x in labels_pred]

cid_4 = preprocessing.LabelEncoder()
cid_3 = preprocessing.LabelEncoder()
cid_1 = preprocessing.LabelEncoder()

# converting the 4 digit codes (full-codes) to integers
char_4 = cid_4.fit([x[:4] for x in labels_cid]+[x[:4] for x in [x[0] for x in labels_pred]])
# converting the 3 digit codes (blocks) to integers
char_3 = cid_3.fit([x[:3] for x in labels_cid]+[x[:3] for x in [x[0] for x in labels_pred]])
# converting the 1 digit codes (chapters) to integers
char_1 = cid_1.fit([x[:1] for x in labels_cid]+[x[:1] for x in [x[0] for x in labels_pred]])

# Integer values for the true labels
true_4 = char_4.transform([x[:4] for x in labels_cid])
true_3 = char_3.transform([x[:3] for x in labels_cid])
true_1 = char_1.transform([x[:1] for x in labels_cid])

# Integer values for the most probable predicted labels (full-code, block and chapter)
pred_4 = char_4.transform([x[:4] for x in [x[0] for x in labels_pred]])
pred_3 = char_3.transform([x[:3] for x in [x[0] for x in labels_pred]])
pred_1 = char_1.transform([x[:1] for x in [x[0] for x in labels_pred]])

# CLASS ACCURACY ANALYSIS

c_labels_cid = [x[:3] for x in labels_cid]
labels_cid_ml_b = np.array([[x[:3] for x in line] for line in labels_cid_ml])

c_labels_pred = [x[:3] for x in [x[0] for x in labels_pred]]
labels_pred_ml_b = np.array([ line.replace(',','').replace('[','').replace(']','').replace("'",'').rstrip('\n\r').split(' ') for line in codecs.open('pred_baseline_block.txt', encoding="iso_8859-1") ])
#%%
def icd9_chap(lst):
    c_labels_cid = lst[:]
    for i in range(len(c_labels_cid)):
        if c_labels_cid[i] >= '001' and c_labels_cid[i] <= '139': 
            c_labels_cid[i] = 1 
        elif c_labels_cid[i] >= '140' and c_labels_cid[i] <= '239': 
            c_labels_cid[i] = 2
        elif c_labels_cid[i] >= '240' and c_labels_cid[i] <= '279': 
            c_labels_cid[i] = 3
        elif c_labels_cid[i] >= '280' and c_labels_cid[i] <= '289': 
            c_labels_cid[i] = 4
        elif c_labels_cid[i] >= '290' and c_labels_cid[i] <= '319': 
            c_labels_cid[i] = 5
        elif c_labels_cid[i] >= '320' and c_labels_cid[i] <= '389': 
            c_labels_cid[i] = 6
        elif c_labels_cid[i] >= '390' and c_labels_cid[i] <= '459': 
            c_labels_cid[i] = 7
        elif c_labels_cid[i] >= '460' and c_labels_cid[i] <= '519': 
            c_labels_cid[i] = 8
        elif c_labels_cid[i] >= '520' and c_labels_cid[i] <= '579': 
            c_labels_cid[i] = 9
        elif c_labels_cid[i] >= '580' and c_labels_cid[i] <= '629': 
            c_labels_cid[i] = 10
        elif c_labels_cid[i] >= '630' and c_labels_cid[i] <= '679': 
            c_labels_cid[i] = 11
        elif c_labels_cid[i] >= '680' and c_labels_cid[i] <= '709': 
            c_labels_cid[i] = 12
        elif c_labels_cid[i] >= '710' and c_labels_cid[i] <= '739': 
            c_labels_cid[i] = 13
        elif c_labels_cid[i] >= '740' and c_labels_cid[i] <= '759': 
            c_labels_cid[i] = 14
        elif c_labels_cid[i] >= '760' and c_labels_cid[i] <= '779': 
            c_labels_cid[i] = 15
        elif c_labels_cid[i] >= '780' and c_labels_cid[i] <= '799': 
            c_labels_cid[i] = 16
        elif c_labels_cid[i] >= '800' and c_labels_cid[i] <= '999': 
            c_labels_cid[i] = 17
        elif c_labels_cid[i] >= 'V01' and c_labels_cid[i] <= 'V91': 
            c_labels_cid[i] = 18
        elif c_labels_cid[i] >= 'E000' and c_labels_cid[i] <= 'E999': 
            c_labels_cid[i] = 19
    return c_labels_cid

def icd10_chap(lst):
    labels_cid_chap = lst[:]
    for i in range(len(labels_cid_chap)):
        if labels_cid_chap[i] >= 'A00' and labels_cid_chap[i] <= 'B99':
            labels_cid_chap[i] = 1 
        elif labels_cid_chap[i] >= 'C00' and labels_cid_chap[i] <= 'D48': 
            labels_cid_chap[i] = 2
        elif labels_cid_chap[i] >= 'D50' and labels_cid_chap[i] <= 'D89': 
            labels_cid_chap[i] = 3
        elif labels_cid_chap[i] >= 'E00' and labels_cid_chap[i] <= 'E90': 
            labels_cid_chap[i] = 4
        elif labels_cid_chap[i] >= 'F00' and labels_cid_chap[i] <= 'F99': 
            labels_cid_chap[i] = 5
        elif labels_cid_chap[i] >= 'G00' and labels_cid_chap[i] <= 'G99': 
            labels_cid_chap[i] = 6
        elif labels_cid_chap[i] >= 'H00' and labels_cid_chap[i] <= 'H59': 
            labels_cid_chap[i] = 7
        elif labels_cid_chap[i] >= 'H60' and labels_cid_chap[i] <= 'H95': 
            labels_cid_chap[i] = 8
        elif labels_cid_chap[i] >= 'I00' and labels_cid_chap[i] <= 'I99': 
            labels_cid_chap[i] = 9
        elif labels_cid_chap[i] >= 'J00' and labels_cid_chap[i] <= 'J99': 
            labels_cid_chap[i] = 10
        elif labels_cid_chap[i] >= 'K00' and labels_cid_chap[i] <= 'K93': 
            labels_cid_chap[i] = 11
        elif labels_cid_chap[i] >= 'L00' and labels_cid_chap[i] <= 'L99': 
            labels_cid_chap[i] = 12
        elif labels_cid_chap[i] >= 'M00' and labels_cid_chap[i] <= 'M99': 
            labels_cid_chap[i] = 13
        elif labels_cid_chap[i] >= 'N00' and labels_cid_chap[i] <= 'N99': 
            labels_cid_chap[i] = 14
        elif labels_cid_chap[i] >= 'O00' and labels_cid_chap[i] <= 'O99': 
            labels_cid_chap[i] = 15
        elif labels_cid_chap[i] >= 'P00' and labels_cid_chap[i] <= 'P96': 
            labels_cid_chap[i] = 16
        elif labels_cid_chap[i] >= 'Q00' and labels_cid_chap[i] <= 'Q99': 
            labels_cid_chap[i] = 17
        elif labels_cid_chap[i] >= 'R00' and labels_cid_chap[i] <= 'R99': 
            labels_cid_chap[i] = 18
        elif labels_cid_chap[i] >= 'S00' and labels_cid_chap[i] <= 'T98': 
            labels_cid_chap[i] = 19
        elif labels_cid_chap[i] >= 'V01' and labels_cid_chap[i] <= 'V98': 
            labels_cid_chap[i] = 20
        elif labels_cid_chap[i] >= 'Z00' and labels_cid_chap[i] <= 'Z99': 
            labels_cid_chap[i] = 21
        else: 
            labels_cid_chap[i] = 22
    return labels_cid_chap

c_labels_cid = icd9_chap(c_labels_cid)
labels_cid_ml_c = np.array([icd9_chap(line) for line in labels_cid_ml_b])

c_labels_pred = icd9_chap(c_labels_pred)
labels_pred_ml_c = np.array([ line.replace(',','').replace('[','').replace(']','').replace("'",'').rstrip('\n\r').split(' ') for line in codecs.open('pred_baseline_chap.txt', encoding="iso_8859-1") ])

for i in range(len(labels_pred_ml_c)):
    if labels_pred_ml_c[i] == ['']: labels_pred_ml_c[i] = [str(c_labels_pred[i])]
    labels_pred_ml_c[i] = list(map(int,labels_pred_ml_c[i]))
#%% ACCURACY, MACRO-AVERAGED PRECISION RECALL AND F1-SCORE FOR FULL-CODES, BLOCK AND CHAPTER

p_per_class = metrics.precision_score(c_labels_cid,c_labels_pred,average=None,labels=list(set(c_labels_cid)))
r_per_class = metrics.recall_score(c_labels_cid,c_labels_pred,average=None,labels=list(set(c_labels_cid)))
f1_per_class = metrics.f1_score(c_labels_cid,c_labels_pred,average=None,labels=list(set(c_labels_cid)))

print('\n -> SINGLE-LABEL ACCURACY (FULL-CODE): %s' % metrics.accuracy_score(true_4,pred_4))
print('\n      -> Precision (FULL-CODE): %s' % metrics.precision_score(true_4,pred_4,average='macro'))
print('\n      -> Recall (FULL-CODE): %s' % metrics.recall_score(true_4,pred_4,average='macro'))
print('\n      -> F1 (FULL-CODE): %s' % metrics.f1_score(true_4,pred_4,average='macro'))

print('\n -> SINGLE-LABEL ACCURACY (BLOCKS): %s' % metrics.accuracy_score(true_3,pred_3))
print('\n      -> Precision (BLOCKS): %s' % metrics.precision_score(true_3,pred_3,average='macro'))
print('\n      -> Recall (BLOCKS): %s' % metrics.recall_score(true_3,pred_3,average='macro'))
print('\n      -> F1 (BLOCKS): %s' % metrics.f1_score(true_3,pred_3,average='macro'))

print('\n -> SINGLE-LABEL ACCURACY (CHAPTER): %s' % metrics.accuracy_score(c_labels_cid,c_labels_pred))
print('\n      -> Precision (CHAPTER): %s' % metrics.precision_score(c_labels_cid,c_labels_pred,average='macro'))
print('\n      -> Recall (CHAPTER): %s' % metrics.recall_score(c_labels_cid,c_labels_pred,average='macro'))
print('\n      -> F1 (CHAPTER): %s' % metrics.f1_score(c_labels_cid,c_labels_pred,average='macro'))

print('\n -> PRECISION; RECALL; F1 SCORE: ')
for i in range(len(f1_per_class)):
    print('\n   | CLASS %s: %s  ;  %s  ;  %s' % (list(set(c_labels_cid))[i], p_per_class[i], r_per_class[i], f1_per_class[i]))
    
#%%
print('\n -> MULTI-LABEL PERFORMANCE METRICS (FULL-CODES):')
print('\n      -> Accuracy: %s' % ex_based_acc(labels_cid_ml, labels_pred_ml))
print('\n      -> Precision: %s' % ex_based_precision(labels_cid_ml, labels_pred_ml))
print('\n      -> Recall: %s' % ex_based_recall(labels_cid_ml, labels_pred_ml))
print('\n      -> F1: %s' % ex_based_f1(labels_cid_ml, labels_pred_ml))
print('\n      -> Hamming Loss: %s' % multi_hamming_loss(labels_cid_ml, labels_pred_ml))
print('\n      -> Exact Match: %s' % exact_match(labels_cid_ml, labels_pred_ml))
print('\n      -> One Error: %s' % one_error(labels_cid_ml, labels_pred_ml))

print('\n -> MULTI-LABEL PERFORMANCE METRICS (BLOCKS):')
print('\n      -> Accuracy: %s' % ex_based_acc(labels_cid_ml_b, labels_pred_ml_b))
print('\n      -> Precision: %s' % ex_based_precision(labels_cid_ml_b, labels_pred_ml_b))
print('\n      -> Recall: %s' % ex_based_recall(labels_cid_ml_b, labels_pred_ml_b))
print('\n      -> F1: %s' % ex_based_f1(labels_cid_ml_b, labels_pred_ml_b))
print('\n      -> Hamming Loss: %s' % multi_hamming_loss(labels_cid_ml_b, labels_pred_ml_b))
print('\n      -> Exact Match: %s' % exact_match(labels_cid_ml_b, labels_pred_ml_b))
print('\n      -> One Error: %s' % one_error(labels_cid_ml_b, labels_pred_ml_b))

print('\n -> MULTI-LABEL PERFORMANCE METRICS (CHAPTERS):')
print('\n      -> Accuracy: %s' % ex_based_acc(labels_cid_ml_c, labels_pred_ml_c))
print('\n      -> Precision: %s' % ex_based_precision(labels_cid_ml_c, labels_pred_ml_c))
print('\n      -> Recall: %s' % ex_based_recall(labels_cid_ml_c, labels_pred_ml_c))
print('\n      -> F1: %s' % ex_based_f1(labels_cid_ml_c, labels_pred_ml_c))
print('\n      -> Hamming Loss: %s' % multi_hamming_loss(labels_cid_ml_c, labels_pred_ml_c))
print('\n      -> Exact Match: %s' % exact_match(labels_cid_ml_c, labels_pred_ml_c))
print('\n      -> One Error: %s' % one_error(labels_cid_ml_c, labels_pred_ml_c))