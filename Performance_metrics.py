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
labels_cid = np.array([ line.replace('[','').replace(']','').replace("'",'').rstrip('\n').split(' ') for line in codecs.open('true_baseline.txt', encoding="iso_8859-1") ])  
labels_cid_main = np.array([ line.rstrip('\n') for line in codecs.open('true_baseline_main.txt', encoding="iso_8859-1") ])  
labels_pred = np.array([ line.replace('[','').replace(']','').replace("'",'').rstrip('\n').split(', ') for line in codecs.open('pred_baseline.txt', encoding="iso_8859-1") ])

# CLASS ACCURACY ANALYSIS

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
    c_labels_pred = lst[:]
    for i in range(len(c_labels_pred)):
        if c_labels_pred[i] >= 'A00' and c_labels_pred[i] <= 'B99': 
            c_labels_pred[i] = 1 
        elif c_labels_pred[i] >= 'C00' and c_labels_pred[i] <= 'D48': 
            c_labels_pred[i] = 2
        elif c_labels_pred[i] >= 'D50' and c_labels_pred[i] <= 'D89': 
            c_labels_pred[i] = 3
        elif c_labels_pred[i] >= 'E00' and c_labels_pred[i] <= 'E90': 
            c_labels_pred[i] = 4
        elif c_labels_pred[i] >= 'F00' and c_labels_pred[i] <= 'F99': 
            c_labels_pred[i] = 5
        elif c_labels_pred[i] >= 'G00' and c_labels_pred[i] <= 'G99': 
            c_labels_pred[i] = 6
        elif c_labels_pred[i] >= 'H00' and c_labels_pred[i] <= 'H59': 
            c_labels_pred[i] = 7
        elif c_labels_pred[i] >= 'H60' and c_labels_pred[i] <= 'H95': 
            c_labels_pred[i] = 8
        elif c_labels_pred[i] >= 'I00' and c_labels_pred[i] <= 'I99': 
            c_labels_pred[i] = 9
        elif c_labels_pred[i] >= 'J00' and c_labels_pred[i] <= 'J99': 
            c_labels_pred[i] = 10
        elif c_labels_pred[i] >= 'K00' and c_labels_pred[i] <= 'K93': 
            c_labels_pred[i] = 11
        elif c_labels_pred[i] >= 'L00' and c_labels_pred[i] <= 'L99': 
            c_labels_pred[i] = 12
        elif c_labels_pred[i] >= 'M00' and c_labels_pred[i] <= 'M99': 
            c_labels_pred[i] = 13
        elif c_labels_pred[i] >= 'N00' and c_labels_pred[i] <= 'N99': 
            c_labels_pred[i] = 14
        elif c_labels_pred[i] >= 'O00' and c_labels_pred[i] <= 'O99': 
            c_labels_pred[i] = 15
        elif c_labels_pred[i] >= 'P00' and c_labels_pred[i] <= 'P96': 
            c_labels_pred[i] = 16
        elif c_labels_pred[i] >= 'Q00' and c_labels_pred[i] <= 'Q99': 
            c_labels_pred[i] = 17
        elif c_labels_pred[i] >= 'R00' and c_labels_pred[i] <= 'R99': 
            c_labels_pred[i] = 18
        elif c_labels_pred[i] >= 'S00' and c_labels_pred[i] <= 'T98': 
            c_labels_pred[i] = 19
        elif c_labels_pred[i] >= 'V01' and c_labels_pred[i] <= 'Y98': 
            c_labels_pred[i] = 20
        elif c_labels_pred[i] >= 'Z00' and c_labels_pred[i] <= 'Z99': 
            c_labels_pred[i] = 21
        else:
            c_labels_pred[i] = 22
    return c_labels_pred

labels_cid_b = np.array([[x[:3] for x in line] for line in labels_cid])
labels_cid_main_b = np.array([x[:3] for x in labels_cid_main])
labels_pred_b = np.array([[x[:3] for x in line] for line in labels_pred])

labels_cid_c = np.array([icd10_chap(line) for line in labels_cid_b])
labels_cid_main_c = np.array(icd10_chap([line for line in labels_cid_main_b]))
labels_pred_c = np.array([icd10_chap(line) for line in labels_pred_b])

#%%
print('\n -> EXAMPLE-BASED PERFORMANCE METRICS (FULL-CODES):')
print('\n      -> Accuracy: %s' % ex_based_acc(labels_cid, labels_pred))
print('\n      -> Precision: %s' % ex_based_precision(labels_cid, labels_pred))
print('\n      -> Recall: %s' % ex_based_recall(labels_cid, labels_pred))
print('\n      -> F1: %s' % ex_based_f1(labels_cid, labels_pred))
print('\n      -> Hamming Loss: %s' % multi_hamming_loss(labels_cid, labels_pred))
print('\n      -> Exact Match: %s' % exact_match(labels_cid, labels_pred))
print('\n      -> One Error: %s' % one_error(labels_cid, labels_pred))

print('\n -> EXAMPLE-BASED PERFORMANCE METRICS (BLOCKS):')
print('\n      -> Accuracy: %s' % ex_based_acc(labels_cid_b, labels_pred_b))
print('\n      -> Precision: %s' % ex_based_precision(labels_cid_b, labels_pred_b))
print('\n      -> Recall: %s' % ex_based_recall(labels_cid_b, labels_pred_b))
print('\n      -> F1: %s' % ex_based_f1(labels_cid_b, labels_pred_b))
print('\n      -> Hamming Loss: %s' % multi_hamming_loss(labels_cid_b, labels_pred_b))
print('\n      -> Exact Match: %s' % exact_match(labels_cid_b, labels_pred_b))
print('\n      -> One Error: %s' % one_error(labels_cid_b, labels_pred_b))

print('\n -> EXAMPLE-BASED PERFORMANCE METRICS (CHAPTERS):')
print('\n      -> Accuracy: %s' % ex_based_acc(labels_cid_c, labels_pred_c))
print('\n      -> Precision: %s' % ex_based_precision(labels_cid_c, labels_pred_c))
print('\n      -> Recall: %s' % ex_based_recall(labels_cid_c, labels_pred_c))
print('\n      -> F1: %s' % ex_based_f1(labels_cid_c, labels_pred_c))
print('\n      -> Hamming Loss: %s' % multi_hamming_loss(labels_cid_c, labels_pred_c))
print('\n      -> Exact Match: %s' % exact_match(labels_cid_c, labels_pred_c))
print('\n      -> One Error: %s' % one_error(labels_cid_c, labels_pred_c))